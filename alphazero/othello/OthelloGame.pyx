# cython: language_level=3
from typing import List, Tuple, Any

from alphazero.othello.OthelloLogic import Board
from alphazero.Game import GameState

import numpy as np

NUM_PLAYERS = 2
NUM_CHANNELS = 1
BOARD_SIZE = 8
ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class OthelloGame(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'GameState') -> bool:
        return (
            np.asarray(self._board.pieces) == np.asarray(other._board.pieces)
            and self._player == other._player
            and self.turns == other.turns
        )

    @staticmethod
    def _get_board():
        return Board(BOARD_SIZE)

    def clone(self) -> 'OthelloGame':
        game = OthelloGame()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game.turns = self.turns
        return game

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBSERVATION_SIZE

    def valid_moves(self):
        # return a fixed size binary vector
        valids = [0] * self.action_size()

        for x, y in self._board.get_legal_moves(self.current_player()):
            valids[self._board.n * x + y] = 1

        return np.array(valids, dtype=np.intc)

    def play_action(self, action: int) -> None:
        move = (action // self._board.n, action % self._board.n)
        self._board.execute_move(move, self.current_player())
        self._update_turn()

    def win_state(self) -> Tuple[bool, int]:
        if self._board.has_legal_moves(self.current_player()):
            return False, 0
        if self._board.has_legal_moves(-self.current_player()):
            return False, 0
        if self._board.count_diff(self.current_player()) > 0:
            return True, self.current_player()
        return True, -self.current_player()

    def observation(self):
        return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        # mirror, rotational
        assert (len(pi) == self._board.n ** 2)

        pi_board = np.reshape(pi[:-1], (self._board.n, self._board.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces), i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                gs = self.clone()
                gs._board = new_b
                l.append((gs, new_pi.ravel()))

        return l


def display(board):
    n = board.shape[0]

    for y in range(n):
        print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1:
                print("b ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("   -----------------------")
