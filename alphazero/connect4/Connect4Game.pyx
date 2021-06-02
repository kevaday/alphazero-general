# cython: language_level=3
from alphazero.Game import Game
from alphazero.connect4.Connect4Logic import Board

import numpy as np

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4
NUM_PLAYERS = 2
NUM_CHANNELS = 1


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, win_length=None):
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH

    def getInitBoard(self):
        b = Board(self.height, self.width, self.win_length)
        return np.asarray(b.pieces)

    def getBoardSize(self):
        return self.height, self.width

    def getActionSize(self):
        return self.width

    def getObservationSize(self):
        return NUM_CHANNELS, self.height, self.width

    def getPlayers(self):
        return list(range(NUM_PLAYERS))

    def getPlayerToPlay(self, board) -> int:
        return np.count_nonzero(board.pieces)

    def _player_range(self, player):
        return 1 if player == self.getPlayers()[0] else -1

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        b.add_stone(action, self._player_range(player))
        return np.asarray(b.pieces), self.getNextPlayer(player)

    def getValidMoves(self, board, player):
        """Any zero value in top row in a valid move"""
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        return np.asarray(b.get_valid_moves())

    def getGameEnded(self, board, player):
        player = self._player_range(player)
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        is_ended, winner = b.get_win_state()
        if is_ended:
            if winner is None:
                # draw has very little value.
                return 1e-4
            elif winner == player:
                return +1
            elif winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', is_ended, winner)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * self._player_range(player)

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board):
        return board.tostring()


def display(board):
    print(" -----------------------")
    print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
