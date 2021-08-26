import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI
from alphazero.envs.connect4.connect4 import Game
from game2dboard import Board
from pathlib import Path


class GUI(CustomGUI):
    def __init__(self):
        _, self.height, self.width = Game.observation_size()
        self.board = Board(self.height, self.width, Path('alphazero') / 'envs' / 'connect4' / 'img')

    def update_state(self, state):
        for x in range(self.width):
            for y in range(self.height):
                piece = state._board.pieces[y][x]
                if piece == -1:
                    piece = 2
                self.board[y][x] = piece
