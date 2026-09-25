# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt, pow

import cython
import numpy as np
cimport numpy as np
import random

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
MIN_NOISE_ALPHA = 0.1
_DRAW_VALUE = 0.5

np.seterr(all='raise')

cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public object e
    cdef public bint is_win
    cdef public float q
    cdef public float v
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, int action):
        self._children = []
        self.a = action
        self.e = None
        self.is_win = False
        self.q = 0.0
        self.v = 0.0
        self.n = 0
        self.p = 0.0
        self.player = 0

    def __repr__(self):
        return 'Node(a={}, e={}, is_win={}, q={}, v={}, n={}, p={}, player={})' \
            .format(self.a, self.e, self.is_win, self.q, self.v, self.n, self.p, self.player)

    cdef void add_children(self, object valids):
        cdef int a
        cdef int length = len(valids)
        for a in range(length):
            if valids[a]:
                self._children.append(Node(a))
        # shuffle children to prevent directional bias
        random.shuffle(self._children)

    @cython.cdivision(True)
    cdef inline float uct(self, float sqrt_parent_n, float fpu_value, float cpuct):
        cdef float q_val
        if self.n == 0:
            q_val = fpu_value
        else:
            q_val = self.q
        return q_val + cpuct * self.p * sqrt_parent_n / (1.0 + self.n)

    cdef Node best_child(self, float fpu_reduction, float cpuct):
        cdef Node c
        cdef float seen_policy = 0.0
        cdef float parent_q = 0.0
        cdef int parent_n = 0
        cdef float fpu_value

        for c in self._children:
            if c.n > 0:
                seen_policy += c.p
                parent_q += c.n * c.q
                parent_n += c.n

        if parent_n > 0:
            parent_q /= parent_n
            fpu_value = parent_q - fpu_reduction * sqrt(seen_policy)
            # Clamp to 0.0 for probability bounds
            if fpu_value < 0.0:
                fpu_value = 0.0
        else:
            fpu_value = self.v

        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self.n)
        cdef float uct_val
        cdef Node child = None

        for c in self._children:
            uct_val = c.uct(sqrt_n, fpu_value, cpuct)
            if uct_val > cur_best:
                cur_best = uct_val
                child = c

        return child

cdef class MCTS:
    cdef public float root_noise_frac
    cdef public float root_temp
    cdef public float min_discount
    cdef public float fpu_reduction
    cdef public float cpuct
    cdef public int winstate_size
    cdef public Node _root
    cdef public Node _curnode
    cdef public list _path
    cdef public int depth
    cdef public int max_depth
    cdef public int _discount_max_depth

    def __init__(self, float root_noise_frac, float root_temp, float min_discount,
                 float fpu_reduction, float cpuct, int winstate_size):
        self.root_noise_frac = root_noise_frac
        self.root_temp = root_temp
        self.min_discount = min_discount
        self.fpu_reduction = fpu_reduction
        self.cpuct = cpuct
        self.winstate_size = winstate_size
        self._root = Node(-1)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    cpdef void reset(self):
        self._root = Node(-1)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    cpdef void search(self, object gs, object nn, int sims, bint add_root_noise, bint add_root_temp):
        cdef float[:] v
        cdef float[:] p
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            p, v = nn(leaf.observation())
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void raw_search(self, object gs, int sims, bint add_root_noise, bint add_root_temp):
        cdef Py_ssize_t policy_size = gs.action_size()
        cdef float[:] v = np.zeros(gs.num_players() + 1, dtype=np.float32)
        cdef float[:] p = np.full(policy_size, 1.0, dtype=np.float32)
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void update_root(self, object gs, int a):
        if not self._root._children:
            self._root.add_children(gs.valid_moves())

        cdef Node c
        for c in self._root._children:
            if c.a == a:
                self._root = c
                return

        raise ValueError(f'Invalid action encountered while updating root: {a}')

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        if num_valid_moves == 0:
            return

        cdef float[:] noise = np.array(np.random.dirichlet(
            [max(NOISE_ALPHA_RATIO / num_valid_moves, MIN_NOISE_ALPHA)] * num_valid_moves
        ), dtype=np.float32)
        cdef Node c
        cdef float n

        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1.0 - self.root_noise_frac) + self.root_noise_frac * n

    cpdef object find_leaf(self, object gs):
        self.depth = 0
        self._curnode = self._root
        cdef object leaf = gs.clone()

        # Fast C-boolean check avoids Python/NumPy array overhead
        while self._curnode.n > 0 and not self._curnode.is_win:
            self._path.append(self._curnode)
            self._curnode = self._curnode.best_child(self.fpu_reduction, self.cpuct)
            leaf.play_action(self._curnode.a)
            self.depth += 1

        if self.depth > self.max_depth:
            self.max_depth = self.depth
            self._discount_max_depth = self.depth

        if self._curnode.n == 0:
            self._curnode.player = leaf.player
            self._curnode.e = leaf.win_state()
            self._curnode.is_win = True if np.any(self._curnode.e) else False
            self._curnode.add_children(leaf.valid_moves())

        return leaf

    cpdef void process_results(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        cdef Node c
        cdef float pi_sum = 0.0

        if self._curnode.is_win:
            value = np.array(self._curnode.e, dtype=np.float32)
        else:
            # We don't allocate a mask array anymore! We just extract
            # the network probabilities directly into the valid children
            # and normalize them in pure C.
            for c in self._curnode._children:
                pi_sum += pi[c.a]

            if pi_sum > 0:
                for c in self._curnode._children:
                    c.p = pi[c.a] / pi_sum
            else:
                pi_sum = len(self._curnode._children)
                if pi_sum > 0:
                    for c in self._curnode._children:
                        c.p = 1.0 / pi_sum

            if self._curnode == self._root:
                # Fast math exponentiation for temperature
                if add_root_temp:
                    pi_sum = 0.0
                    for c in self._curnode._children:
                        if c.p > 0:
                            c.p = pow(c.p, 1.0 / self.root_temp)
                            pi_sum += c.p
                    if pi_sum > 0:
                        for c in self._curnode._children:
                            c.p /= pi_sum

                if add_root_noise:
                    self._add_root_noise()

        cdef Py_ssize_t num_players = gs.num_players()
        cdef Node parent
        cdef float v
        cdef float discounted_v
        cdef float curr_discount = 1.0
        cdef float step_decay = 1.0

        # Protected against ZeroDivisionError when depth == 0
        if self._discount_max_depth > 0:
            step_decay = pow(self.min_discount, 1.0 / self._discount_max_depth)

        if self._curnode.n == 0:
            self._curnode.v = self._get_value(value, self._curnode.player, num_players)

        while self._path:
            parent = self._path.pop()
            v = self._get_value(value, parent.player, num_players)

            # Pull the value towards the draw value based on discount factor
            discounted_v = _DRAW_VALUE + (v - _DRAW_VALUE) * curr_discount

            self._curnode.q = (self._curnode.q * self._curnode.n + discounted_v) / (self._curnode.n + 1)
            self._curnode.n += 1
            self._curnode = parent

            curr_discount *= step_decay

        self._root.n += 1

    cpdef float _get_value(self, float[:] value, Py_ssize_t player, Py_ssize_t num_players):
        if value.size > num_players:
            return value[player] + value[num_players] / num_players
        else:
            return value[player]

    cpdef int[:] counts(self, object gs):
        cdef int[:] counts = np.zeros(gs.action_size(), dtype=np.int32)
        cdef Node c

        for c in self._root._children:
            counts[c.a] = c.n
        return np.asarray(counts)

    cpdef int best_action(self, object gs):
        return np.argmax(self.counts(gs))

    cpdef np.ndarray probs(self, object gs, float temp=1.0):
        cdef np.ndarray counts_arr = np.array(self.counts(gs), dtype=np.float32)
        cdef float counts_sum = np.sum(counts_arr)
        cdef np.ndarray[dtype=np.float32_t, ndim=1] probs
        cdef Py_ssize_t best_action

        # Protect against empty arrays
        if counts_sum == 0.0:
            probs = np.full(gs.action_size(), 1.0 / gs.action_size(), dtype=np.float32)
            return probs

        if temp == 0:
            best_action = np.argmax(counts_arr)
            probs = np.zeros_like(counts_arr)
            probs[best_action] = 1.0
            return probs

        try:
            probs = (counts_arr / counts_sum) ** (1.0 / temp)
            probs /= np.sum(probs)
            return probs
        except (OverflowError, FloatingPointError):
            best_action = np.argmax(counts_arr)
            probs = np.zeros_like(counts_arr)
            probs[best_action] = 1.0
            return probs

    cpdef float value(self, bint average=False):
        cdef float value = 0.0
        cdef Node c

        if average:
            cdef int count = 0
            for c in self._root._children:
                if c.n > 0:
                    value += c.q
                    count += 1
            if count > 0:
                value /= count
        else:
            for c in self._root._children:
                if c.q > value and c.n > 0:
                    value = c.q

        return value