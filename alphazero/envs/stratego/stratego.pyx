# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import numpy as np
cimport numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from boardgame import Square
from boardgame.board cimport Square
from alphazero.envs.stratego.engine import Board
from alphazero.envs.stratego.engine cimport Board

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef int other_team_offset = 20
cdef int visible_offset = 100
cdef int RED_TEAM_COLOUR = 1
cdef tuple ALL_RED_PIECES = tuple(range(1, 13))
cdef tuple ALL_BLUE_PIECES = tuple((p + other_team_offset for p in ALL_RED_PIECES))

cdef int NUM_PLAYERS = 2
cdef int NUM_STACKED_OBSERVATIONS = 1
cdef int NUM_BASE_CHANNELS = 30
cdef int NUM_CHANNELS = NUM_BASE_CHANNELS * NUM_STACKED_OBSERVATIONS
cdef int NUM_PIECES = len(ALL_RED_PIECES)
cdef int DRAW_MOVE_COUNT = 512

cdef Board b = Board()
cdef int ACTION_SIZE = max(
    # size is 1048 for width 8, height 10, num_pieces 12
    b.width + b.height * b.width + NUM_PIECES * b.width * b.height,
    # 1280 for width 8, height 10
    b.width * b.height * (b.width + b.height - 2)
)
cdef tuple OBS_SIZE = (NUM_CHANNELS, b.height, b.width)


cpdef int get_action(Board board, tuple move):
    cdef int x
    cdef int y
    cdef int new_x = move[1].x
    cdef int new_y = move[1].y
    cdef int move_type

    if isinstance(move[0], int):
        # phase 1 of game
        return new_x + new_y * board.width + (move[0] - other_team_offset * super(Board, board).to_play()) * board.width * board.height

    else:
        # phase 2
        x = move[0].x
        y = move[0].y

        if (x - new_x) == 0:
            move_type = new_y if new_y < y else new_y - 1
        else:
            move_type = board.height + new_x - 1
            if new_x >= x: move_type -= 1

        return (board.width + board.height - 2) * (x + y * board.width) + move_type

cpdef tuple get_move(Board board, int action):
    cdef int a, size
    cdef int move_type
    cdef int start_x, start_y
    cdef int new_x, new_y

    if board.play_phase:
        size = board.width + board.height - 2
        move_type = action % size
        a = action // size
        start_x = a % board.width
        start_y = a // board.width

        if move_type < board.height - 1:
            new_x = start_x
            new_y = move_type
            if move_type >= start_y: new_y += 1
        else:
            new_x = move_type - board.height + 1
            if new_x >= start_x: new_x += 1
            new_y = start_y

        return Square(int(start_x), int(start_y)), Square(int(new_x), int(new_y))

    else:
        size = board.width * board.height
        a = action % size
        return (action // size) + other_team_offset * super(Board, board).to_play(), Square(a % board.width, a // board.width)


cpdef list _add_obs(Board b, int const_max_player, int const_max_turns):
    cdef list obs_planes = []
    cdef np.ndarray[DTYPE_t, ndim=2] red_bombs = np.zeros_like(b._state, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] blue_bombs = red_bombs.copy()
    cdef Square bomb_square
    cdef int i

    # add binary planes for the positions of pieces on both teams
    obs_planes.extend((
        np.array(np.where(np.in1d(b._state % visible_offset, ALL_RED_PIECES), 1., 0.), dtype=DTYPE).reshape((b._state.shape[0], b._state.shape[1])),
        np.array(np.where(np.in1d(b._state % visible_offset, ALL_BLUE_PIECES), 1., 0.), dtype=DTYPE).reshape((b._state.shape[0], b._state.shape[1]))
    ))

    # add binary planes for the visible pieces of both teams
    for i in range(1, NUM_PIECES + 1):
        obs_planes.extend((
            np.array(np.where(b.get_mask((i + visible_offset,)), 1., 0.), dtype=DTYPE),
            np.array(np.where(b.get_mask((i + visible_offset + other_team_offset,)), 1., 0.), dtype=DTYPE)
        ))

    # add binary planes for the locations of exploded bombs on both teams
    for bomb_square in b.red_exploded_bombs:
        red_bombs[bomb_square.y, bomb_square.x] = 1.

    for bomb_square in b.blue_exploded_bombs:
        blue_bombs[bomb_square.y, bomb_square.x] = 1.

    obs_planes.extend((red_bombs, blue_bombs))

    # add binary plane for the current player and relative plane for the current turn number
    obs_planes.extend((
        np.full_like(b._state,
            super(Board, b).to_play() / (const_max_player - 1) if const_max_player > 1 else 0., dtype=DTYPE
        ),
        np.full_like(b._state,
            <float>b.num_turns / <float>const_max_turns if const_max_turns else 0., dtype=DTYPE
        )
    ))

    return obs_planes


cpdef list _add_empty(Board board):
    return [np.zeros_like(board._state, dtype=DTYPE)] * NUM_BASE_CHANNELS


cpdef np.ndarray _get_observation(Board board, int const_max_players, int const_max_turns, int past_obs=1, list past_states=[]):
    cdef list past, obs = []
    cdef Py_ssize_t i

    if past_states:
        past = past_states.copy()
        past.insert(0, board)
        for i in range(past_obs):
            if board.num_turns < i:
                obs.extend(_add_empty(board))
            else:
                obs.extend(_add_obs(past[i], const_max_players, const_max_turns))
    else:
        obs = _add_obs(board, const_max_players, const_max_turns)

    return np.array(obs, dtype=DTYPE)

cdef class Game:  #(GameState):
    cdef public Board _board

    def __init__(self, _board=None):
        self._board = _board or Board()

    def __eq__(self, other: 'Game') -> bool:
        return self._board == other._board

    def __str__(self):
        return str(self._board) + '\n'

    @property
    def player(self) -> int:
        return super(Board, self._board).to_play()

    @property
    def turns(self):
        return self._board.num_turns

    cpdef Game clone(self):
        return Game(self._board.copy())

    @staticmethod
    def num_players():
        return NUM_PLAYERS

    @staticmethod
    def action_size():
        return ACTION_SIZE

    @staticmethod
    def observation_size():
        return OBS_SIZE

    cpdef int _next_player(self, int player, int turns=1):
        return (player + turns) % Game.num_players()

    cpdef np.ndarray valid_moves(self):
        cdef list valids = [0] * ACTION_SIZE
        cdef tuple move

        for move in self._board.legal_moves(pieces=(), piece_type=self._board.to_play()):
            valids[get_action(self._board, move)] = 1

        return np.array(valids, dtype=np.uint8)

    cpdef void play_action(self, int action):
        cdef tuple move = get_move(self._board, action)
        self._board.move(move[0], move[1], check_turn=False, _check_valid=False, _check_win=False)

    cpdef np.ndarray win_state(self):
        cdef np.ndarray[dtype=np.uint8_t, ndim=1] result = np.zeros(NUM_PLAYERS + 1, dtype=np.uint8)
        cdef int winner

        # Check if maximum moves have been exceeded
        if self.turns >= DRAW_MOVE_COUNT:
            result[NUM_PLAYERS] = 1
        else:
            winner = self._board.get_winner()
            if winner != 0:
                result[0 if winner == RED_TEAM_COLOUR else 1] = 1

        return result

    cpdef np.ndarray observation(self):
        return _get_observation(
            self._board,
            NUM_PLAYERS,
            DRAW_MOVE_COUNT,
            NUM_STACKED_OBSERVATIONS
        )

    cpdef list symmetries(self, np.ndarray pi):
        cdef np.ndarray[DTYPE_t, ndim=2] new_state = np.fliplr(self._board._state.astype(DTYPE))
        cdef np.ndarray[DTYPE_t, ndim=1] new_pi = np.zeros(ACTION_SIZE, dtype=DTYPE)
        cdef Board new_b = self._board.copy()
        cdef Square dest
        cdef tuple new_move

        new_b._state = new_state
        for action, prob in enumerate(pi):
            move = get_move(self._board, action)
            dest = Square(self._board.width - 1 - move[1].x, move[1].y)

            if self._board.play_phase:
                new_move = (Square(self._board.width - 1 - move[0].x, move[0].y), dest)
            else:
                new_move = (move[0], dest)

            new_pi[get_action(new_b, new_move)] = prob

        return [(self.clone(), pi), (Game(new_b), new_pi)]


cpdef void display(Game g, int action=-1):
        if action != -1: print(f'Action: {action}, Move: {get_move(g._board, action)}')
        print(g)
