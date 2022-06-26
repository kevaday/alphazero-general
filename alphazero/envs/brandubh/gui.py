import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI, GameWindow, GameBoardWidget, NUM_BEST_ACTIONS
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.brandubh.fastafl import Game, Square, get_action, get_move
from PySide2 import QtCore, QtGui, QtWidgets
from typing import List

import numpy as np

BACKGROUND_COLOR = QtGui.QColor(153, 76, 0)
ARROWHEAD_ANGLE = 30


class CustomGameBoardWidget(GameBoardWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = None
        self._best_action = None
        self._last_best_action = None
        self._worst_action = None
        self._last_worst_action = None

    def set_state(self, state: Game) -> None:
        self._state = state
        self.update()

    def set_actions(self, actions: List[int]) -> None:
        best_action = actions[0]
        worst_action = actions[-1]
        if best_action == self._last_best_action and worst_action == self._last_worst_action:
            return
        self._last_best_action = self._best_action
        self._last_worst_action = self._worst_action
        self._best_action = best_action
        self._worst_action = worst_action
        self.update()

    def draw_arrow(self, colour: QtGui.QColor, action: int) -> None:
        move = get_move(self._state._board, action)
        x1, y1 = move[0]
        x2, y2 = move[1]
        line_width = self.cell_size // 20

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(colour, line_width))
        painter.drawLine(
            x1 * self.cell_size + self.cell_size // 2,
            y1 * self.cell_size + self.cell_size // 2,
            x2 * self.cell_size + self.cell_size // 2,
            y2 * self.cell_size + self.cell_size // 2
        )

        # draw a circle on both ends of the line
        painter.setBrush(colour)
        painter.drawEllipse(
            x1 * self.cell_size + self.cell_size // 2 - line_width,
            y1 * self.cell_size + self.cell_size // 2 - line_width,
            line_width * 2,
            line_width * 2
        )
        painter.drawEllipse(
            x2 * self.cell_size + self.cell_size // 2 - line_width,
            y2 * self.cell_size + self.cell_size // 2 - line_width,
            line_width * 2,
            line_width * 2
        )
        painter.end()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self._state is None:
            return

        if self._worst_action is not None:
            self.draw_arrow(QtCore.Qt.red, self._worst_action)
        if self._best_action is not None:
            self.draw_arrow(QtCore.Qt.green, self._best_action)
        
        # draw the arrow head using two lines
        """
        to_point = np.array([tile_to_x, tile_to_y], dtype=np.float32)
        from_point = np.array([tile_from_x, tile_from_y], dtype=np.float32)
        line_vector = to_point - from_point
        line_length = np.linalg.norm(line_vector)
        point_on_line = to_point - (line_width / np.tan(np.deg2rad(ARROWHEAD_ANGLE))) * line_length * line_vector
        perpendicular_vector = np.array([-line_vector[1], line_vector[0]], dtype=np.float32)
        normal = line_width / (2 * line_length)
        left_point = point_on_line + normal * perpendicular_vector
        right_point = point_on_line - normal * perpendicular_vector

        painter.drawLine(
            tile_to_x * self.cell_size + self.cell_size // 2,
            tile_to_y * self.cell_size + self.cell_size // 2,
            left_point[0],
            left_point[1]
        )
        painter.drawLine(
            tile_to_x * self.cell_size + self.cell_size // 2,
            tile_to_y * self.cell_size + self.cell_size // 2,
            right_point[0],
            right_point[1]
        )
        """


class GUI(CustomGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(Game, *args, **kwargs)
        _, self.height, self.width = Game.observation_size()

        self.board = CustomGameBoardWidget(
            self.width,
            self.height,
            cell_size=100,
            title=self.title,
            background_colour=BACKGROUND_COLOR
        )
        self.window = GameWindow(
            game_board=self.board,
            title=self.title,
            evaluator=self.evaluator,
            verbose=True,
            num_best_actions=NUM_BEST_ACTIONS if self.show_hints else 0,
            use_evaluator=(self.evaluator is not None),
            action_to_move=self._action_to_move
        )
        if self.evaluator and not self.evaluator.model:
            self.evaluator.model = self._crude_model
        if self.show_hints:
            self.window.eval_stats_timer.timeout.connect(self._update_draw_actions)
        self.board.tileClicked.connect(self._tile_click)
        self.board.closing.connect(self.on_window_close)

        self.board.add_circle_pixmap(1, QtCore.Qt.white)
        self.board.add_circle_pixmap(2, QtCore.Qt.black)
        self.board.add_circle_pixmap(3, QtCore.Qt.lightGray)
        self.board.add_cross_pixmap(4, QtCore.Qt.black)
        self.board.add_cross_pixmap(5, QtCore.Qt.red)

        self.update_state(self._state)

    def _crude_model(self, state: Game):
        value = state.crude_value()
        return (
            np.full(state.action_size(), 1, dtype=np.float32),
            np.array([value, 1 - value, 0], dtype=np.float32)
        )

    @staticmethod
    def _action_to_move(state: Game, action: int) -> str:
        move = get_move(state._board, action)
        return f'{move[0]._get_tuple()} -> {move[1]._get_tuple()}'

    def _update_draw_actions(self):
        if self.evaluator is None or not self.evaluator.is_running:
            return
        self.board.set_actions(self.evaluator.get_best_actions())

    def _tile_click(self, x, y):
        print('[DEBUG] Tile clicked: {} {}'.format(x, y))
        if not self.user_input:
            self.board.clear_selection()
            return

        board = self._state._board

        def highlight_legals(square: Square) -> bool:
            if not board.is_turn(square):
                return False
            legals = board.legal_moves(pieces=(square,))
            if not legals:
                return False

            self.board.remove_highlights()
            for (_, end_square) in legals:
                self.board.highlight_tile(*end_square)

            self.board.update()
            return True

        def remove_selection():
            print('[DEBUG] Removing selection')
            self.board.clear_selection()
            self.board.remove_highlights()
            self.board.update()

        if self.board.last_selected_tile and self.board.selected_tile:
            from_square = Square(*self.board.last_selected_tile)
            to_square = Square(*self.board.selected_tile)
            move = (from_square, to_square)
            print('[DEBUG] Move: {}'.format(move))

            if move in board.legal_moves(pieces=(from_square,)):
                action = get_action(board, move)
                print('[DEBUG] Move is legal, action: {}'.format(action))
                remove_selection()
                self.on_player_move(action)

        elif self.board.selected_tile and highlight_legals(Square(*self.board.selected_tile)):
            print('[DEBUG] Legals highlighted')
            return
        else:
            remove_selection()

    def show(self):
        self.window.show()

    def close(self):
        self.window.close()

    def undo(self):
        raise NotImplementedError

    def update_state(self, state):
        self.board.set_state(state)

        for x in range(self.width):
            for y in range(self.height):
                tile = state._board[Square(x, y)]
                if tile == 7:
                    self.board.set_tile(x, y, 4)
                    self.board.set_tile(x, y, 3, overwrite=False)
                elif tile == 8:
                    self.board.set_tile(x, y, 5)
                    self.board.set_tile(x, y, 3, overwrite=False)
                else:
                    self.board.set_tile(x, y, tile if tile else None)

        if state.last_action is not None and state.last_action != -1:
            self.board.remove_highlights()
            from_square, to_square = get_move(state._board, state.last_action)
            self.board.clear_fills()
            self.board.fill_tile(*from_square, QtGui.Qt.darkGreen)
            self.board.fill_tile(*to_square, QtGui.Qt.darkGreen)
            self.board.clear_selection()

        if state.win_state().any():
            self.user_input = False
            self.window.stop_evaluator()
        else:
            self.window.side_menu.update_turn(state.player + 1)
            self.window.run_evaluator(state, block=False)

        self.window.update()
        super().update_state(state)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    eval_model = NNetWrapper.from_checkpoint(
        Game, '', r'D:\Projects\Python\alphazero-general\checkpoint\brandubh\brandubh_fastafl_iteration-0048.pkl'
    )
    gui = GUI(title='Hnefatafl', eval_model=eval_model)
    gui.show()
    app.exec_()
