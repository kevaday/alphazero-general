# cython: language_level=3
from typing import List, Tuple, Any

from alphazero.envs.othello.OthelloLogic import Board
from alphazero.Game import GameState

import numpy as np

NUM_PLAYERS = 2
NUM_CHANNELS = 1
BOARD_SIZE = 8
ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class Game(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or self._get_board())

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'GameState') -> bool:
        return (
            np.asarray(self._board.pieces) == np.asarray(other._board.pieces)
            and self._player == other._player
            and self.turns == other.turns
        )

    def display(self):
        display(self._board.pieces)

    @staticmethod
    def _get_board(*args, **kwargs):
        return Board(BOARD_SIZE, *args, **kwargs)

    def clone(self) -> 'Game':
        board = self._get_board(_pieces=np.copy(np.asarray(self._board.pieces)))
        game = Game(_board=board)
        game._player = self._player
        game._turns = self.turns
        return game

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBSERVATION_SIZE

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    def _player_range(self):
        return (1, -1)[self.player]

    def valid_moves(self):
        # return a fixed size binary vector
        valids = [0] * self.action_size()

        for x, y in self._board.get_legal_moves(self._player_range()):
            valids[self._board.n * x + y] = 1

        return np.array(valids, dtype=np.intc)

    def play_action(self, action: int) -> None:
        move = (action // self._board.n, action % self._board.n)
        self._board.execute_move(move, self._player_range())
        self._update_turn()

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * (NUM_PLAYERS + 1)
        player = self._player_range()

        if self._board.has_legal_moves(player):
            return tuple(result)
        elif self._board.has_legal_moves(-player):
            return tuple(result)
        elif self._board.count_diff(player) > 0:
            result[self.player] = True
        else:
            result[self._next_player(self.player)] = True

        return tuple(result)

    def observation(self):
        return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        # mirror, rotational
        assert (len(pi) == self._board.n ** 2)

        pi_board = np.reshape(pi, (self._board.n, self._board.n))
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces), i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                gs = self.clone()
                gs._board.pieces = new_b
                result.append((gs, new_pi.ravel()))

        return result


def display(board: np.ndarray):
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
