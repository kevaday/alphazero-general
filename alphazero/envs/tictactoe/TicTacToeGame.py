from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.tictactoe.TicTacToeLogic import Board

import numpy as np

NUM_PLAYERS = 2
PLAYERS = list(range(NUM_PLAYERS))
NUM_CHANNELS = 1
BOARD_SIZE = 3
ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class TicTacToeGame(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board(BOARD_SIZE)

    def __eq__(self, other: 'TicTacToeGame') -> bool:
        return (
            self._board.pieces == other._board.pieces
            and self._board.n == other._board.n
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self) -> 'TicTacToeGame':
        g = TicTacToeGame()
        g._board = self._board
        g._player = self._player
        g.turns = self.turns
        return g

    @staticmethod
    def get_players() -> List[int]:
        return PLAYERS

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

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * (NUM_PLAYERS + 1)
        player = (1, -1)[self.current_player()]

        if self._board.is_win(player):
            result[self.current_player()] = True
        elif self._board.is_win(-player):
            result[self._next_player(self.current_player())] = True
        elif not self._board.has_legal_moves():
            result[-1] = True

        return tuple(result)

    def observation(self):
        return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi: np.ndarray) -> List[Tuple[Any, int]]:
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

    print("   ", end="")
    for y in range(n):
        print(y, "", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|", end="")  # print the row #
        for x in range(n):
            piece = board[y][x]  # get the piece to print
            if piece == -1:
                print("X ", end="")
            elif piece == 1:
                print("O ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
