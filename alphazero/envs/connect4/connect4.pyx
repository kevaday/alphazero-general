# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.connect4.Connect4Logic import Board

import numpy as np

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4
NUM_PLAYERS = 2
MAX_TURNS = 42
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 4 if MULTI_PLANE_OBSERVATION else 1


class Game(GameState):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board(DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_WIN_LENGTH)

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns

    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game._turns = self.turns
        game.last_action = self.last_action
        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return DEFAULT_WIDTH

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves())

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.add_stone(action, (1, -1)[self.player])
        self._update_turn()

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * 3
        game_over, player = self._board.get_win_state()

        if game_over:
            index = -1
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True

        return np.array(result, dtype=np.uint8)

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == 1, 1, 0)
            player2 = np.where(pieces == -1, 1, 0)
            colour = np.full_like(pieces, self.player)
            turn = np.full_like(pieces, self.turns / MAX_TURNS, dtype=np.float32)
            return np.array([player1, player2, colour, turn], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[:, ::-1]
        return [(self.clone(), pi), (new_state, pi[::-1])]


def display(board, action=None):
    if action:
        print(f'Action: {action}, Move: {action + 1}')
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
