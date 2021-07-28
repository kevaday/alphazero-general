# cython: language_level=3

import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from alphazero.Game import GameState
from fastafl.cengine import Board, Square
from fastafl import variants
from typing import List, Tuple, Any


def _get_board():
    return Board(*GAME_VARIANT)


GAME_VARIANT = variants.brandubh_args
NUM_PLAYERS = 2
NUM_STACKED_OBSERVATIONS = 1
NUM_BASE_CHANNELS = 5
NUM_CHANNELS = NUM_BASE_CHANNELS * NUM_STACKED_OBSERVATIONS
Move_T = Tuple[Square, Square]

b = _get_board()
ACTION_SIZE = b.width * b.height * (b.width + b.height - 2)
OBS_SIZE = (NUM_CHANNELS, b.width, b.height)
del b

DRAW_MOVE_COUNT = 100


def get_move(board: Board, action: int) -> Move_T:
    size = (board.width + board.height - 2)
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


def get_action(board: Board, move: Move_T) -> int:
    x = move[0].x
    y = move[0].y
    new_x = move[1].x
    new_y = move[1].y

    if (x - new_x) == 0:
        move_type = new_y if new_y < y else new_y - 1
    else:
        move_type = board.height + new_x - 1
        if new_x >= x: move_type -= 1

    return (board.width + board.height - 2) * (x + y * board.width) + move_type


def _get_observation(board: Board, const_max_player: int, const_max_turns: int, past_obs: int = 1, past_states: list = None):
    obs = []

    def add_obs(b):
        game_board = b._state
        black = np.where(game_board == 2, 1., 0.)
        white = np.where(game_board == 1, 1., 0.)
        king = np.where((game_board == 3) | (game_board == 7) | (game_board == 8), 1., 0.)
        turn_colour = np.full_like(
            game_board,
            2 - b.to_play() / (const_max_player - 1) if const_max_player > 1 else 0
        )
        turn_number = np.full_like(
            game_board,
            b.num_turns / const_max_turns if const_max_turns else 0, dtype=np.float32
        )
        obs.extend([black, white, king, turn_colour, turn_number])

    def add_empty():
        obs.extend([[[0] * board.width] * board.height] * NUM_BASE_CHANNELS)

    if past_states:
        past = past_states.copy()
        past.insert(0, board)
        for i in range(past_obs):
            if board.num_turns < i:
                add_empty()
            else:
                add_obs(past[i])
    else:
        add_obs(board)

    return np.array(obs, dtype=np.float32)


class TaflGame(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or _get_board())

    def __eq__(self, other: 'TaflGame') -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self._board) + '\n'

    @staticmethod
    def _get_player_int(player: int) -> int:
        return (1, -1)[2 - player]

    def clone(self) -> 'GameState':
        g = TaflGame(self._board.copy())
        g._player = self._player
        g._turns = self.turns
        return g

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBS_SIZE

    def valid_moves(self):
        valids = [0] * self.action_size()

        for move in self._board.legal_moves(piece_type=self._board.to_play()):
            valids[get_action(self._board, move)] = 1

        return np.array(valids, dtype=np.intc)

    def play_action(self, action: int) -> None:
        move = get_move(self._board, action)
        self._board.move(*move, _check_valid=False, _check_win=False)
        self._update_turn()

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * (NUM_PLAYERS + 1)

        # Check if maximum moves have been exceeded
        if self.turns >= DRAW_MOVE_COUNT:
            result[-1] = True
        else:
            winner = self._board.get_winner()
            if winner:
                result[2 - winner] = True

        return tuple(result)

    def observation(self):
        return _get_observation(
            self._board,
            NUM_PLAYERS,
            DRAW_MOVE_COUNT,
            NUM_STACKED_OBSERVATIONS
        )

    def symmetries(self, pi: np.ndarray) -> List[Tuple[Any, int]]:
        action_size = self.action_size()
        assert (len(pi) == action_size)
        syms = [None] * 8

        for i in range(1, 5):
            for flip in (False, True):
                state = np.rot90(np.array(self._board._state), i)
                if flip:
                    state = np.fliplr(state)

                new_b = self._board.copy()
                new_b._state = state
                new_pi = [0] * action_size
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
                syms[(i - 1) * 2 + int(flip)] = (new_state, np.array(new_pi, dtype=np.float32))

        return syms

    def crude_value(self) -> int:
        _, result = self.win_state()
        white_pieces = len(list(filter(lambda p: p.is_white, self._board.pieces)))
        black_pieces = len(list(filter(lambda p: p.is_black, self._board.pieces)))
        return self.player * (1000 * result + black_pieces - white_pieces)


def display(state: TaflGame, action: int = None):
    print(f'Action: {action}, Move: {get_move(state._board, action)}')
    print(state)
