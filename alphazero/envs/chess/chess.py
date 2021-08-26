from alphazero.Game import GameState
from typing import List, Tuple, Any

import chess
import string

DIGS = string.digits + string.ascii_letters

NUM_PLAYERS = 2
BOARD_SIZE = 8
ACTION_SIZE = BOARD_SIZE ^ 4
NUM_CHANNELS = 1 #placeholder
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


# TODO: Use https://github.com/Unimax/alpha-zero-general-chess-and-battlesnake/blob/master/chesspy/ChessGame.py for
#  implementation


def _int2base(x, base, length):
    if x < 0:
        sign = -1
    elif x == 0:
        return [DIGS[0]]*length
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(DIGS[int(x % base)])
        x //= base

    if sign < 0:
        digits.append('-')

    while len(digits) < length: digits.append('0')
    return list(map(lambda x: int(x, base), digits))


class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return chess.Board()

    def __eq__(self, other) -> bool:
        return (
            self._board == other._board
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self):
        g = ChessGame()
        g._board = self._board.copy()
        g._player = self._player
        g._turns = self.turns
        return g

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBSERVATION_SIZE

    def valid_moves(self):
        valids = [0] * self.action_size()
        for move in self._board.legal_moves:
            valids[
                move.tile.x
                + move.tile.y * BOARD_SIZE
                + move.new_tile.x * BOARD_SIZE ** 2
                + move.new_tile.y * BOARD_SIZE ** 3
            ] = 1

    def play_action(self, action: int) -> None:
        pass

    def win_state(self) -> Tuple[bool, int]:
        pass

    def observation(self):
        pass

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        pass
