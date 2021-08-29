import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI
from alphazero.envs.connect4.connect4 import Game
from game2dboard import Board
from pathlib import Path

import numpy as np


class GUI(CustomGUI):
    def __init__(self, state=None):
        _, self.height, self.width = Game.observation_size()
        self.board = Board(self.height, self.width, Path('alphazero') / 'envs' / 'connect4' / 'img')
        self.board.on_mouse_click = self._mouse_click
        self.board.cell_size = 100
        self.turns = 0
        self.player = 0
        if not state:
            state = Game()
        self.update_state(state)
        self.board.show()

    def _mouse_click(self, btn, row, col):
        state = self.get_state()
        try:
            state.play_action(col)
        except ValueError:
            return
        self.update_state(state)

    def update_state(self, state):
        for x in range(self.width):
            for y in range(self.height):
                piece = state._board.pieces[y][x]
                if piece == -1:
                    piece = 2
                self.board[y][x] = piece if piece else None

        self.turns = state.turns
        self.player = state.player

    def get_state(self):
        gs = Game()
        gs._turns = self.turns
        gs._player = self.player
        gs._board.pieces = np.array([list(map(lambda x: x if x else 0, row)) for row in self.board], dtype=np.intc)
        return gs
