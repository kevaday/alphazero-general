# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt

import numpy as np
cimport numpy as np
from alphazero.utils import dotdict


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
_DRAW_VALUE = 0.5

np.seterr(all='raise')


"""
def rebuild_node(children, a, cpuct, num_players, e, q, n, p, player):
    childs = []
    for child_state in children:
        rebuild, args = child_state[0], child_state[1:]
        child = rebuild(*args)
        childs.append(child)

    node = Node(a, cpuct, num_players)
    node._children = childs
    node.a = a
    node.cpuct = cpuct
    node.e = e
    node.q = q
    node.n = n
    node.p = p
    node.player = player

    return node
"""


# @cython.auto_pickle(True)
cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public np.ndarray e
    cdef public float q
    cdef public float v
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, int action, int num_players):
        self._children = []
        self.a = action
        self.e = np.zeros(num_players + 1, dtype=np.uint8)
        self.q = 0
        self.v = 0
        self.n = 0
        self.p = 0
        self.player = 0

    def __repr__(self):
        return 'Node(a={}, e={}, q={}, v={}, n={}, p={}, player={})' \
            .format(self.a, self.e, self.q, self.v, self.n, self.p, self.player)

    # def __reduce__(self):
    #    return rebuild_node, ([n.__reduce__() for n in self._children], self.a, self.cpuct, self._players, self.e, self.q, self.n, self.p, self.player)

    cdef void add_children(self, np.ndarray v, int num_players):
        self._children.extend([Node(a, num_players) for a, valid in enumerate(v) if valid])
        # shuffle children
        np.random.shuffle(self._children)

    cdef void update_policy(self, float[:] pi):
        cdef Node c
        for c in self._children:
            c.p = pi[c.a]

    cdef float uct(self, float sqrt_parent_n, float fpu_value, float cpuct):
        return (fpu_value if self.n == 0 else self.q) + cpuct * self.p * sqrt_parent_n / (1 + self.n)

    cdef Node best_child(self, float fpu_reduction, float cpuct):
        cdef Node c
        cdef float seen_policy = sum([c.p for c in self._children if c.n > 0])
        cdef float fpu_value = self.v - fpu_reduction * sqrt(seen_policy)
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self.n)
        cdef float uct
        child = None

        for c in self._children:
            uct = c.uct(sqrt_n, fpu_value, cpuct)
            if uct > cur_best:
                cur_best = uct
                child = c

        return child


"""
def rebuild_mcts(num_players, cpuct, root, curnode, path):
    mcts = MCTS(num_players, cpuct)
    mcts.cpuct = cpuct
    mcts._root = root
    mcts._curnode = curnode
    mcts.path = path
    return mcts
"""


# @cython.auto_pickle(True)
cdef class MCTS:
    cdef public float root_noise_frac
    cdef public float root_temp
    cdef public float min_discount
    cdef public float fpu_reduction
    cdef public float cpuct
    cdef public int _num_players
    cdef public Node _root
    cdef public Node _curnode
    cdef public list _path
    cdef public int depth
    cdef public int max_depth
    cdef public int _discount_max_depth

    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct
        self._num_players = args._num_players
        self._root = Node(-1, self._num_players)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    def __repr__(self):
        return 'MCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, _num_players={}, ' \
               '_root={}, _curnode={}, _path={}, depth={}, max_depth={})' \
            .format(self.root_noise_frac, self.root_temp, self.min_discount,
                    self.fpu_reduction,self.cpuct, self._num_players, self._root,
                    self._curnode, self._path, self.depth, self.max_depth)

    cpdef void reset(self):
        self._root = Node(-1, self._num_players)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    # def __reduce__(self):
    #   return rebuild_mcts, (self._root._players, self.cpuct, self._root, self._curnode, self._path)

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
        cdef float[:] v = np.zeros(gs.num_players() + 1, dtype=np.float32)  #np.full((value_size,), 1 / value_size, dtype=np.float32)
        cdef float[:] p = np.full(policy_size, 1, dtype=np.float32)
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void update_root(self, object gs, int a):
        if not self._root._children:
            self._root.add_children(gs.valid_moves(), self._num_players)

        cdef Node c
        for c in self._root._children:
            if c.a == a:
                self._root = c
                return

        raise ValueError(f'Invalid action encountered while updating root: {c.a}')

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        cdef float[:] noise = np.array(np.random.dirichlet(
            [NOISE_ALPHA_RATIO / num_valid_moves] * num_valid_moves
        ), dtype=np.float32)
        cdef Node c
        cdef float n

        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1 - self.root_noise_frac) + self.root_noise_frac * n

    cpdef object find_leaf(self, object gs):
        self.depth = 0
        self._curnode = self._root
        cdef object leaf = gs.clone()

        while self._curnode.n > 0 and not self._curnode.e.any():
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
            self._curnode.add_children(leaf.valid_moves(), self._num_players)

        return leaf

    cpdef void process_results(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        cdef float[:] valids
        cdef Node c
        
        if self._curnode.e.any():
            value = np.array(self._curnode.e, dtype=np.float32)
        else:
            # reconstruct valid moves based on children of current node
            # instead of recalculating with gs.valid_moves() -> expensive
            valids = np.zeros(gs.action_size(), dtype=np.float32)
            for c in self._curnode._children:
                valids[c.a] = 1

            # mask invalid moves and rescale
            pi *= np.array(valids, dtype=np.float32)
            pi /= np.sum(pi)

            if self._curnode == self._root:
                # add root temperature
                if add_root_temp:
                    pi = np.asarray(pi) ** (1.0 / self.root_temp)
                    # re-normalize
                    pi /= np.sum(pi)

                self._curnode.update_policy(pi)
                if add_root_noise:
                    self._add_root_noise()
            else:
                self._curnode.update_policy(pi)

        cdef Py_ssize_t num_players = gs.num_players()
        cdef Node parent
        cdef float v
        cdef float discount
        cdef int i = 0
        while self._path:
            parent = self._path.pop()
            v = self._get_value(value, parent.player, num_players)

            # apply discount only to current node's Q value
            discount = (self.min_discount ** (i / self._discount_max_depth))
            if v < _DRAW_VALUE:
                # (1 - discount) + 1 to invert it because bad values far away
                # are better than bad values close to root
                discount = 2 - discount
            elif v == _DRAW_VALUE:
                # don't discount value in the rare case that it is a precise draw (0.5)
                discount = 1

            # scale value to the range [-1, 1]
            # v = 2 * v * discount - 1

            self._curnode.q = (self._curnode.q * self._curnode.n + v * discount) / (self._curnode.n + 1)
            if self._curnode.n == 0:
                self._curnode.v = self._get_value(value, self._curnode.player, num_players)  # * 2 - 1
            self._curnode.n += 1
            self._curnode = parent
            i += 1

        self._root.n += 1

    cpdef float _get_value(self, float[:] value, Py_ssize_t player, Py_ssize_t num_players):
        return value[player] + value[num_players] / num_players

    cpdef int[:] counts(self, object gs):
        cdef int[:] counts = np.zeros(gs.action_size(), dtype=np.int32)
        cdef Node c

        for c in self._root._children:
            counts[c.a] = c.n
        return np.asarray(counts)

    cpdef int best_action(self, object gs):
        return np.argmax(self.counts(gs))

    cpdef np.ndarray probs(self, object gs, float temp=1.0):
        cdef float[:] counts = np.array(self.counts(gs), dtype=np.float32)
        cdef np.ndarray[dtype=np.float32_t, ndim=1] probs
        cdef Py_ssize_t best_action

        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

        try:
            probs = (counts / np.sum(counts)) ** (1.0 / temp)
            probs /= np.sum(probs)
            return probs
        except OverflowError:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

    cpdef float value(self, bint average=False):
        """Get the value of the current root node in the range [0, 1]
        by looking at the max value of child nodes (or averaging them).
        """
        cdef float value = 0
        cdef Node c
        
        if average:
            value = sum([c.q for c in self._root._children if c.n > 0]) / len(self._root._children)
        
        else:
            for c in self._root._children:
                if c.q > value and c.n > 0:
                    value = c.q
            
        return value
