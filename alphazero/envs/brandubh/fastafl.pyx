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

#from alphazero.cGame import GameState
#from alphazero.cGame cimport GameState
from boardgame import Square
from boardgame.board cimport Square
from fastafl.cengine import Board
from fastafl.cengine cimport Board
from fastafl import variants


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cpdef Board _get_board():
    return Board(*GAME_VARIANT)


cdef tuple GAME_VARIANT = variants.brandubh_args
cdef int NUM_PLAYERS = 2
cdef int NUM_STACKED_OBSERVATIONS = 1
cdef int NUM_BASE_CHANNELS = 5
cdef int NUM_CHANNELS = NUM_BASE_CHANNELS * NUM_STACKED_OBSERVATIONS

cdef Board b = _get_board()
cdef int ACTION_SIZE = b.width * b.height * (b.width + b.height - 2)
cdef tuple OBS_SIZE = (NUM_CHANNELS, b.width, b.height)

cdef int DRAW_MOVE_COUNT = 100


cpdef tuple get_move(Board board, int action):
    cdef int size = board.width + board.height - 2
    cdef int move_type = action % size
    cdef int a = action // size
    cdef int start_x = a % board.width
    cdef int start_y = a // board.width
    cdef int new_x, new_y

    if move_type < board.height - 1:
        new_x = start_x
        new_y = move_type
        if move_type >= start_y: new_y += 1
    else:
        new_x = move_type - board.height + 1
        if new_x >= start_x: new_x += 1
        new_y = start_y

    return Square(int(start_x), int(start_y)), Square(int(new_x), int(new_y))


cpdef int get_action(Board board, tuple move):
    cdef int x = move[0].x
    cdef int y = move[0].y
    cdef int new_x = move[1].x
    cdef int new_y = move[1].y
    cdef int move_type

    if (x - new_x) == 0:
        move_type = new_y if new_y < y else new_y - 1
    else:
        move_type = board.height + new_x - 1
        if new_x >= x: move_type -= 1

    return (board.width + board.height - 2) * (x + y * board.width) + move_type


cpdef list _add_obs(Board b, int const_max_player, int const_max_turns):
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] black_mask = b.get_mask((2,))
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] white_mask = b.get_mask((1,))
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] king_mask = b.get_mask((3, 7, 8))

    cdef np.ndarray[np.float32_t, ndim=2] black = np.array(np.where(black_mask, 1., 0.), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] white = np.array(np.where(white_mask, 1., 0.), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] king =  np.array(np.where(king_mask, 1., 0.), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] turn_colour = np.full_like(
        b._state, 2 - b.to_play() / (const_max_player - 1) if const_max_player > 1 else 0, dtype=np.float32
    )
    cdef np.ndarray[np.float32_t, ndim=2] turn_number = np.full_like(
        b._state, b.num_turns / const_max_turns if const_max_turns else 0, dtype=np.float32
    )

    return [black, white, king, turn_colour, turn_number]


cpdef list _add_empty(Board board):
    return [[[0] * board.width] * board.height] * NUM_BASE_CHANNELS


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
        obs.extend(_add_obs(board, const_max_players, const_max_turns))

    return np.array(obs, dtype=np.float32)


cdef class Game:#(GameState):
    cdef public Board _board
    cdef public int _player
    cdef public int _turns
    
    def __init__(self, _board=None):
        self._board = _board or _get_board()
        self._player = 0
        self._turns = 0

    def __eq__(self, other: 'Game') -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self._board) + '\n'
    
    @property
    def player(self) -> int:
        return self._player

    @property
    def turns(self):
        return self._turns

    @staticmethod
    cdef int _get_player_int(int player):
        return (1, -1)[2 - player]

    cpdef Game clone(self):
        cdef Game g = Game(self._board.copy())
        g._player = self._player
        g._turns = self.turns
        return g

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

    cpdef void _update_turn(self):
        """Should be called at the end of play_action"""
        self._player = self._next_player(self._player)
        self._turns += 1

    cpdef np.ndarray valid_moves(self):
        cdef list valids = [0] * ACTION_SIZE
        cdef tuple move

        for move in self._board.legal_moves(pieces=(), piece_type=self._board.to_play()):
            valids[get_action(self._board, move)] = 1

        return np.array(valids, dtype=np.uint8)

    cpdef void play_action(self, int action):
        cdef tuple move = get_move(self._board, action)
        self._board.move(move[0], move[1], check_turn=False, _check_valid=False, _check_win=False)
        self._update_turn()
        

    cpdef np.ndarray win_state(self):
        cdef np.ndarray[dtype=np.uint8_t, ndim=1] result = np.zeros(NUM_PLAYERS + 1, dtype=np.uint8)
        cdef int winner

        # Check if maximum moves have been exceeded
        if self.turns >= DRAW_MOVE_COUNT:
            result[NUM_PLAYERS] = 1
        else:
            winner = self._board.get_winner()
            if winner != 0:
                result[2 - winner] = 1

        return result

    cpdef np.ndarray observation(self):
        return _get_observation(
            self._board,
            NUM_PLAYERS,
            DRAW_MOVE_COUNT,
            NUM_STACKED_OBSERVATIONS
        )

    cpdef list symmetries(self, np.ndarray pi):
        cdef list syms = [None] * 8
        cdef int i
        cdef bint flip
        cdef np.ndarray[np.float32_t, ndim=2] state
        cdef np.ndarray[np.float32_t, ndim=1] new_pi
        cdef Board new_b
        cdef Game new_state

        for i in range(1, 5):
            for flip in (False, True):
                state = np.rot90(np.array(self._board._state, dtype=np.float32), i)
                if flip:
                    state = np.fliplr(state)

                new_b = self._board.copy()
                new_b._state = state
                new_pi = np.zeros(ACTION_SIZE, dtype=np.float32)
                for action, prob in enumerate(pi):
                    move = get_move(self._board, action)
                    x = move[0].x
                    new_x = move[1].x
                    y = move[0].y
                    new_y = move[1].y

                    for _ in range(i):
                        temp_x = x
                        temp_new_x = new_x
                        x = self._board.width - 1 - y
                        new_x = self._board.width - 1 - new_y
                        y = temp_x
                        new_y = temp_new_x
                    if flip:
                        x = self._board.width - 1 - x
                        new_x = self._board.width - 1 - new_x

                    new_action = get_action(new_b, (Square(x, y), Square(new_x, new_y)))
                    new_pi[new_action] = prob

                new_state = self.clone()
                new_state._board = new_b
                syms[(i - 1) * 2 + int(flip)] = (new_state, new_pi)

        return syms

    """
    cpdef int crude_value(self):
        cdef int[:] result = self.win_state()
        white_pieces = len(list(filter(lambda p: p.is_white, self._board.pieces)))
        black_pieces = len(list(filter(lambda p: p.is_black, self._board.pieces)))
        return self.player * (1000 * result[1] + black_pieces - white_pieces)
    """


cpdef void display(Game state, int action=-1):
    if action != -1: print(f'Action: {action}, Move: {get_move(state._board, action)}')
    print(state)
