import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI, GameWindow, NUM_BEST_ACTIONS
from alphazero.envs.connect4.connect4 import Game
from PySide2.QtCore import Qt


class GUI(CustomGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(Game, *args, **kwargs)
        _, self.height, self.width = Game.observation_size()
        self.window = GameWindow(
            self.width,
            self.height,
            cell_size=100,
            title=self.title,
            # image_dir=str(Path(__file__).parent / 'img'),
            evaluator=self.evaluator,
            verbose=True,
            num_best_actions=NUM_BEST_ACTIONS if self.show_hints else 0,
            use_evaluator=(self.evaluator is not None),
            action_to_move=lambda state, action: str(action + 1)
        )
        if self.show_hints:
            self.window.eval_stats_timer.timeout.connect(self._update_draw_actions)
        self.board = self.window.game_board
        self.board.tileClicked.connect(self._mouse_click)
        self.board.closing.connect(self.on_window_close)

        self.board.add_circle_pixmap(1, Qt.black)
        self.board.add_circle_pixmap(2, Qt.white)

        self.update_state(self._state)

    def _update_draw_actions(self):
        if self.evaluator is None or not self.evaluator.is_running:
            return

        actions = self.evaluator.get_best_actions()
        if not actions:
            return

        self.board.clear_fills()
        self.board.fill_tile(actions[0], 0, Qt.green)
        self.board.fill_tile(actions[-1], 0, Qt.red)

        self.board.update()

    def _mouse_click(self, action, _):
        if (
            self.user_input
            and self._state.valid_moves()[action]
            and callable(self.on_player_move)
        ):
            self.on_player_move(action)

    def show(self):
        self.window.show()

    def close(self):
        self.window.close()
        super().close()

    def undo(self):
        raise NotImplementedError

    def update_state(self, state):
        for x in range(self.width):
            for y in range(self.height):
                piece = state._board.pieces[y][x]
                if piece == -1:
                    piece = 2
                self.board.set_tile(x, y, piece if piece else None)

        if state.last_action is not None:
            # remove previous highlight
            self.board.remove_highlights()
            # highlight the tile where the piece landed
            for y in range(self.height):
                if state._board.pieces[y][state.last_action] != 0:
                    self.board.highlight_tile(state.last_action, y)
                    break

        if state.win_state().any():
            self.user_input = False
            self.window.stop_evaluator()
        else:
            self.window.side_menu.update_turn(state.player + 1)
            self.window.run_evaluator(state, block=False)

        self.window.update()
        super().update_state(state)


if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    gui = GUI(title='Connect 4')
    gui.show()
    sys.exit(app.exec_())

