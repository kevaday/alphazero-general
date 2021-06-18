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
import cython

np.seterr(all='raise')


def rebuild_node(children, a, cpuct, e, q, n, p, player):
    childs = []
    for child_state in children:
        rebuild, args = child_state[0], child_state[1:]
        child = rebuild(*args)
        childs.append(child)

    node = Node(a, cpuct)
    node._children = childs
    node.a = a
    node.cpuct = cpuct
    node.e = e
    node.q = q
    node.n = n
    node.p = p
    node.player = player

    return node


@cython.auto_pickle(True)
cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public float cpuct
    cdef public (bint, int) e
    cdef public float q
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, int action, float cpuct):
        self._children = []
        self.a = action
        self.cpuct = cpuct
        self.e = (False, 0)
        self.q = 0
        self.n = 0
        self.p = 0
        self.player = 0

    def __reduce__(self):
        return rebuild_node, ([n.__reduce__() for n in self._children], self.a, self.cpuct, self.e, self.q, self.n, self.p, self.player)

    cdef add_children(self, int[:] v):
        cdef Py_ssize_t a
        for a in range(len(v)):
            if v[a] == 1:
                self._children.append(Node(a, self.cpuct))
        # shuffle

    cdef update_policy(self, float[:] pi):
        cdef Node c
        for c in self._children:
            c.p = pi[c.a]

    cdef uct(self, float sqrtParentN):
        uct = self.q + self.cpuct * self.p * sqrtParentN/(1+self.n)
        return uct

    cdef best_child(self):
        child = None
        curBest = -float('inf')
        sqrtN = sqrt(self.n)
        cdef Node c
        for c in self._children:
            uct = c.uct(sqrtN)
            if uct > curBest:
                curBest = uct
                child = c
        return child


def rebuild_mcts(cpuct, root, curnode, path):
    mcts = MCTS()
    mcts.cpuct = cpuct
    mcts._root = root
    mcts._curnode = curnode
    mcts.path = path
    return mcts


@cython.auto_pickle(True)
cdef class MCTS:
    cdef public float cpuct
    cdef public Node _root
    cdef public Node _curnode
    cdef public list path
    def __init__(self, float cpuct=2.0):
        self.cpuct = cpuct
        self._root = Node(-1, cpuct)
        self._curnode = self._root
        self.path = []

    def __reduce__(self):
        return rebuild_mcts, (self.cpuct, self._root, self._curnode, self.path)

    cpdef search(self, gs, nn, sims):
        cdef float v
        cdef float[:] p
        for _ in range(sims):
            leaf = self.find_leaf(gs)
            p, v = nn(leaf.observation()) if not leaf.win_state() else np.array([], dtype=np.float32), 0
            self.process_results(leaf, v, p)

    cpdef update_root(self, gs, int a):
        if self._root._children == []:
            self._root.add_children(gs.valid_moves())
        cdef Node c
        for c in self._root._children:
            if c.a == a:
                self._root = c
                return

        raise ValueError(f'Invalid action while updating root: {c.a}')

    cpdef find_leaf(self, gs):
        self._curnode = self._root
        gs = gs.clone()
        while self._curnode.n > 0 and not self._curnode.e[0]:
            self.path.append(self._curnode)
            self._curnode = self._curnode.best_child()
            gs.play_action(self._curnode.a)

        if self._curnode.n == 0:
            ws = gs.win_state()
            self._curnode.player = gs.current_player()
            self._curnode.e = (ws[0], ws[1]*self._curnode.player)
            self._curnode.add_children(gs.valid_moves())

        return gs

    cpdef process_results(self, gs, float value, float[:] pi):
        if self._curnode.e[0]:
            value = self._curnode.e[1]
        else:
            self._curnode.update_policy(pi)

        player = gs.current_player()
        while self.path:
            parent = self.path.pop()
            v = value if parent.player == player else -value
            self._curnode.q = (self._curnode.q * self._curnode.n + v) / (self._curnode.n + 1)
            self._curnode.n += 1
            self._curnode = parent

        self._root.n += 1

    cpdef counts(self, gs):
        cdef int[:] counts = np.zeros(gs.action_size(), dtype=np.intc)
        cdef Node c
        for c in self._root._children:
            counts[c.a] = c.n
        return counts

    cpdef probs(self, gs, temp=1):
        counts = np.zeros(gs.action_size())
        cdef Node c
        for c in self._root._children:
            counts[c.a] = c.n

        if temp == 0:
            bestA = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            return probs

        try:
            probs = counts ** (1.0/temp)
            probs /= np.sum(probs)
            return probs
        except OverflowError:
            bestA = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            return probs

    cpdef value(self):
        value = None
        cdef Node c
        for c in self._root._children:
            if value == None or c.q > value:
                value = c.q
        return value
