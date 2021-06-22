# cython: language_level=3
# cython: auto_pickle=True
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


class Connect4Game(GameState):
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

    def __eq__(self, other: 'Connect4Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns

    def clone(self) -> 'Connect4Game':
        game = Connect4Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game.turns = self.turns
        return game

    @staticmethod
    def action_size() -> int:
        return DEFAULT_WIDTH

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves())

    def play_action(self, action: int) -> None:
        self._board.add_stone(action, self.current_player())
        self._update_turn()

    def win_state(self) -> Tuple[bool, int]:
        return self._board.get_win_state()

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == self.get_players()[0], 1, 0)
            player2 = np.where(pieces == self.get_players()[1], 1, 0)
            colour = np.full_like(pieces, self.get_players().index(self.current_player()))
            turn = np.full_like(pieces, self.turns / MAX_TURNS)
            return np.array([player1, player2, colour, turn], dtype=np.intc)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[:, ::-1]
        return [(self.clone(), pi), (new_state, pi[::-1])]


def display(board, player=None):
    if player is not None:
        b = board * [1, -1][player]
    else:
        b = board
    
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    print(b)
    print(" -----------------------")
