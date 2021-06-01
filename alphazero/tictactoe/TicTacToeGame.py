from alphazero.Game import Game
from alphazero.tictactoe.TicTacToeLogic import Board

import numpy as np

NUM_PLAYERS = 2
NUM_CHANNELS = 1


class TicTacToeGame(Game):
    """
    Game class implementation for the game of TicTacToe.
    Based on the OthelloGame then getGameEnded() was adapted to new rules.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Based on the OthelloGame by Surag Nair.
    """

    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1

    def getObservationSize(self):
        return NUM_CHANNELS, *self.getBoardSize()

    def getPlayers(self):
        return list(range(NUM_PLAYERS))

    def _player_range(self, player):
        return 1 if player == self.getPlayers()[0] else -1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        b = Board(self.n)
        b.pieces = np.copy(board)

        # action must be a valid move
        if action == self.n * self.n + 1:
            return b, self.getNextPlayer(player)

        move = (action // self.n, action % self.n)
        b.execute_move(move, player)
        return b.pieces, self.getNextPlayer(player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)

        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)

        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        player = self._player_range(player)
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return board * self._player_range(player)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()


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