from alphazero.Game import GameState
from alphazero.envs.gobang.GobangLogic import Board

from typing import List, Tuple, Any

import numpy as np

NUM_PLAYERS = 2
NUM_CHANNELS = 1

BOARD_SIZE = 15
NUM_IN_ROW = 5

ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def get_move(action: int, n: int) -> Tuple[int, int]:
    return action // n, action % n


def get_action(move: Tuple[int, int], n: int) -> int:
    return n * move[0] + move[1]


class GobangGame(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or self._get_board())

    @staticmethod
    def _get_board(*args, **kwargs) -> Board:
        return Board(BOARD_SIZE, NUM_IN_ROW, *args, **kwargs)

    def __eq__(self, other: 'GobangGame') -> bool:
        return (
            self._board.pieces == other._board.pieces
            and self._board.n == other._board.n
            and self._board.n_in_row == other._board.n_in_row
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self) -> 'GobangGame':
        board = self._get_board(_pieces=np.copy(self._board.pieces))
        g = GobangGame(_board=board)
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
        return OBSERVATION_SIZE

    def valid_moves(self):
        # return a fixed size binary vector
        valids = [0] * self.action_size()

        for move in self._board.get_legal_moves():
            valids[get_action(move, self._board.n)] = 1

        return np.array(valids, dtype=np.intc)

    def play_action(self, action: int) -> None:
        move = get_move(action, self._board.n)
        self._board.execute_move(move, (1, -1)[self.player])
        self._update_turn()

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * (NUM_PLAYERS + 1)
        game_over, player = self._board.get_win_state()

        if game_over:
            index = -1
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True

        return tuple(result)

    def observation(self):
        return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi: np.ndarray) -> List[Tuple[Any, int]]:
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


def display(board, action=None):
    n = board.shape[0]

    if action:
        print(f'Action: {action}, Move: {get_move(action, n)}')

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
