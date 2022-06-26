from alphazero import MCTSEvaluator, BaseEvaluator
from alphazero.Game import GameState
from alphazero.utils import map_value
from PySide2 import QtCore, QtGui, QtWidgets
from typing import Callable, Optional, Union, List
from abc import ABC, abstractmethod
from pathlib import Path

EVAL_BAR_ANIMATION_UPDATE = 33
STATS_UPDATE_INTERVAL = 1000
MIN_EVAL_BAR_UPDATE_INCREMENT = 0.005
MAX_EVAL_BAR_UPDATE_INCREMENT = 0.1
EVAL_BAR_START_VALUE = 0.5
NUM_BEST_ACTIONS = 3


def _update_board(func: Callable):
    """Decorator to update widget if argument update=True.
    All other arguments are passed to the function.
    """
    def wrapper(self, *args, **kwargs):
        update = kwargs.pop('update', False)
        result = func(self, *args, **kwargs)
        if update:
            self.update()
        return result
    return wrapper


class _EvalBarWidget(QtWidgets.QWidget):
    def __init__(self, width: int = 140, height: int = 20, parent: QtWidgets.QWidget = None,
                 animation_speed: int = 50,
                 background_colour: QtGui.QColor = QtCore.Qt.white,
                 foreground_colour: QtGui.QColor = QtCore.Qt.black,
                 font: QtGui.QFont = QtGui.QFont('Arial', 10)):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setPalette(QtGui.QPalette(background_colour))
        self.setAutoFillBackground(True)
        self.setFont(font)

        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.timeout.connect(self.update)
        self._animation_timer.setInterval(EVAL_BAR_ANIMATION_UPDATE)

        self.foreground_colour = foreground_colour
        self.current_player = 0
        self.value = 0
        self._new_value = 0
        self.value_increment = map_value(
            animation_speed, 0, 100, MIN_EVAL_BAR_UPDATE_INCREMENT, MAX_EVAL_BAR_UPDATE_INCREMENT
        )
        self.set_value(EVAL_BAR_START_VALUE)

    def set_value(self, value: float):
        self._new_value = value if self.current_player == 0 else 1 - value
        self._animation_timer.start()
        # print('[DEBUG] EvalBarWidget.set_value: {}'.format(self._new_value))

    def update_turn(self, player: int):
        self.current_player = player
        print('[DEBUG] EvalBarWidget.update_turn: {}'.format(player))

    def next_turn(self):
        self.update_turn(1 - self.current_player)

    def update(self) -> None:
        if self.value != self._new_value:
            if self.value_increment == MAX_EVAL_BAR_UPDATE_INCREMENT:
                self.value = self._new_value
            elif self.value < self._new_value:
                self.value += self.value_increment
            else:
                self.value -= self.value_increment

            if abs(self.value - self._new_value) < self.value_increment:
                self.value = self._new_value
                # if threading.current_thread() is threading.main_thread():
                self._animation_timer.stop()

        elif self._animation_timer.isActive():
            self._animation_timer.stop()
            # self.value = self._new_value

        super().update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1, QtCore.Qt.SolidLine))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.fillRect(0, 0, int(self.width() * self.value), self.height(), QtGui.QBrush(self.foreground_colour))
        painter.end()


class EvalBar(QtWidgets.QWidget):
    def __init__(self, *args, players: List[str] = None, font: QtGui.QFont = QtGui.QFont('Arial', 12),
                 parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        self.setLayout(self.layout)
        self.setFont(font)

        if not players:
            players = ['Player 1', 'Player 2']
        elif len(players) != 2:
            raise ValueError('Only 2 players supported for evaluation bar')

        self._player1_label = QtWidgets.QLabel(players[0], self)
        self._player1_label.setFont(self.font())
        self.layout.addWidget(self._player1_label)

        self._bar = _EvalBarWidget(*args, parent=parent, font=font, **kwargs)
        self.layout.addWidget(self._bar)

        self._player2_label = QtWidgets.QLabel(players[1], self)
        self._player2_label.setFont(self.font())
        self.layout.addWidget(self._player2_label)

    def set_value(self, value: float):
        self._bar.set_value(value)

    def update_turn(self, player: int):
        self._bar.update_turn(player)

    def next_turn(self):
        self._bar.next_turn()


class SideMenuWidget(QtWidgets.QWidget):
    """A side menu with a vertical layout to attach
    to the right side of the provided board widget.
    """
    def __init__(self, parent: QtWidgets.QWidget, width: int = 300, border_width: int = 5, spacing: int = 25,
                 background_colour: QtGui.QColor = QtCore.Qt.white, font: QtGui.QFont = QtGui.QFont('Arial', 26)):
        super().__init__(parent)
        self.setPalette(QtGui.QPalette(background_colour))
        self.setFont(font)
        self.border_width = border_width
        self.menu_width = width
        self.turn_label = None
        self.eval_bar = None

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(spacing)
        self.layout.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)

    def update(self) -> None:
        self.setFixedSize(self.menu_width, self.parent().height())
        super().update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, self.border_width, QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, self.height())
        painter.end()

    def add_widget(self, *args, **kwargs):
        self.layout.addWidget(*args, **kwargs)

    def add_text(self, text: str, font: QtGui.QFont = None, **kwargs) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setFont(font or self.font())
        self.layout.addWidget(label, **kwargs)
        return label

    def add_button(self, text: str, callback: Callable, font: QtGui.QFont = None, **kwargs) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text, self)
        button.setFont(font or self.font())
        button.clicked.connect(callback)
        self.layout.addWidget(button, **kwargs)
        return button

    def add_turn_label(self, text: str = '', *args, **kwargs):
        self.turn_label = self.add_text(text, *args, **kwargs)

    def update_turn(self, player: Union[int, str], update_eval_bar: bool = False):
        if not self.turn_label:
            self.add_turn_label()
        self.turn_label.setText(f'Player {player} to move')
        if self.eval_bar and update_eval_bar:
            self.eval_bar.next_turn()

    def update_turn_label(self, text: str):
        if not self.turn_label:
            self.add_turn_label()
        self.turn_label.setText(text)

    def add_eval_bar(self, *args, **kwargs):
        if self.eval_bar is not None:
            raise ValueError('Evaluation bar can only be added once')

        kwargs['parent'] = self
        self.eval_bar = EvalBar(*args, **kwargs)
        self.layout.addWidget(self.eval_bar)

    def update_eval_bar(self, value: float):
        if self.eval_bar is None:
            raise ValueError('Evaluation bar has not been added')
        self.eval_bar.set_value(value)


class GameBoardWidget(QtWidgets.QWidget):
    closing = QtCore.Signal()
    tileClicked = QtCore.Signal(int, int)
    keyPressed = QtCore.Signal(QtGui.QKeyEvent)
    paintBoard = QtCore.Signal(QtGui.QPaintEvent)

    def __init__(self, board_width: int, board_height: int, cell_size: int = 100, grid_line_width: int = 2,
                 title: str = None, background_colour: QtGui.QColor = QtGui.QColor(QtCore.Qt.white),
                 highlight_colour: QtGui.QColor = QtGui.QColor(QtCore.Qt.green), image_dir: str = None,
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.board_width = board_width
        self.board_height = board_height
        self.cell_size = cell_size
        self.grid_line_width = grid_line_width

        self.highlight_colour = highlight_colour
        self.highlighted_tiles = []
        self._fill_tiles = {}
        self.selected_tile = None
        self.last_selected_tile = None

        self._pixmaps = {}
        self._tile_map = {}
        if image_dir is None:
            image_dir = Path(__file__).parent / 'img'
        else:
            image_dir = Path(image_dir)
        for file in image_dir.glob('*.png'):
            self._pixmaps[file.stem] = QtGui.QPixmap(str(file))

        self.setAutoFillBackground(True)
        self.setPalette(QtGui.QPalette(background_colour))

        self.setWindowTitle(title)
        self.setFixedSize(self.board_width * self.cell_size, self.board_height * self.cell_size)

    def __getitem__(self, item: tuple) -> Optional[str]:
        return self.get_tile(item[0], item[1])

    def __setitem__(self, key: tuple, value: Optional[str]) -> None:
        self.set_tile(key[0], key[1], value)

    def __iter__(self):
        """Return each row of the board as a list"""
        for y in range(self.board_height):
            yield [self[x, y] for x in range(self.board_width)]

    def add_circle_pixmap(self, name: str, colour: QtCore.Qt.GlobalColor, border_width: int = 2) -> None:
        pixmap = QtGui.QPixmap(self.cell_size, self.cell_size)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        painter.setBrush(QtGui.QBrush(colour, QtCore.Qt.SolidPattern))
        painter.drawEllipse(0, 0, self.cell_size, self.cell_size)
        if border_width > 0:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, border_width, QtCore.Qt.SolidLine))
            painter.drawEllipse(0, 0, self.cell_size, self.cell_size)
        painter.end()

        self.register_pixmap(name, pixmap)

    def add_cross_pixmap(self, name: str, colour: QtCore.Qt.GlobalColor, line_width: int = 2) -> None:
        pixmap = QtGui.QPixmap(self.cell_size, self.cell_size)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(colour, line_width, QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, self.cell_size, self.cell_size)
        painter.drawLine(0, self.cell_size, self.cell_size, 0)
        painter.end()

        self.register_pixmap(name, pixmap)

    def add_filled_pixmap(self, name: str, colour: QtCore.Qt.GlobalColor) -> None:
        pixmap = QtGui.QPixmap(self.cell_size, self.cell_size)
        pixmap.fill(colour)
        self.register_pixmap(name, pixmap)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self.keyPressed.emit(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = event.x() // self.cell_size
        y = event.y() // self.cell_size

        self.last_selected_tile = self.selected_tile
        if self.selected_tile == (x, y):
            self.selected_tile = None
        else:
            self.selected_tile = (x, y)
        self.tileClicked.emit(x, y)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closing.emit()
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtCore.Qt.black, self.grid_line_width, QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw filled tiles
        for x, y in self._fill_tiles:
            painter.fillRect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size, self._fill_tiles[(x, y)]
            )

        # Draw highlighted tiles
        for x, y in self.highlighted_tiles:
            painter.fillRect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size, self.highlight_colour
            )

        # Draw images
        for x, y in self._tile_map:
            for tile in self._tile_map[(x, y)]:
                try:
                    painter.drawPixmap(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size,
                                       self._pixmaps[tile])
                except KeyError:
                    raise ValueError(f'No registered image found for tile `{tile}`')

        # Draw tile borders
        for x in range(self.board_width + 1):
            painter.drawLine(x * self.cell_size, 0, x * self.cell_size, self.board_height * self.cell_size)
        for y in range(self.board_height + 1):
            painter.drawLine(0, y * self.cell_size, self.board_width * self.cell_size, y * self.cell_size)

        painter.end()
        self.paintBoard.emit(event)

    def clear_selection(self):
        self.selected_tile = None
        self.last_selected_tile = None

    def register_pixmap(self, name, pixmap: QtGui.QPixmap):
        self._pixmaps[str(name)] = pixmap

    @_update_board
    def highlight_tile(self, x: int, y: int) -> None:
        self.highlighted_tiles.append((x, y))

    @_update_board
    def remove_highlight(self, x: int, y: int) -> None:
        self.highlighted_tiles.remove((x, y))

    @_update_board
    def remove_highlights(self) -> None:
        self.highlighted_tiles = []

    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.board_width and 0 <= y < self.board_height

    @_update_board
    def set_tile(self, x: int, y: int, tile: Optional[str], overwrite=True) -> None:
        if not self._is_in_bounds(x, y):
            raise ValueError(f'Coordinates {x}, {y} are out of bounds')

        if tile is None:
            self.remove_tile(x, y)
        elif str(tile) not in self._pixmaps:
            raise ValueError(f'No image found for tile `{tile}`')
        elif overwrite or (x, y) not in self._tile_map:
            self._tile_map[(x, y)] = [str(tile)]
        else:
            self._tile_map[(x, y)].append(str(tile))

    def get_tiles(self, x: int, y: int) -> List[str]:
        if not self._is_in_bounds(x, y):
            raise ValueError(f'Coordinates {x}, {y} are out of bounds')

        return self._tile_map.get((x, y), [])

    @_update_board
    def remove_tile(self, x: int, y: int, first_only=True) -> None:
        if (x, y) not in self._tile_map:
            return

        if first_only:
            self._tile_map[(x, y)] = self._tile_map[(x, y)][1:]
        else:
            del self._tile_map[(x, y)]

    @_update_board
    def clear_board(self) -> None:
        self._tile_map = {}
        self.remove_highlights()
        self.clear_selection()

    @_update_board
    def fill_tile(self, x: int, y: int, colour: QtCore.Qt.GlobalColor) -> None:
        self._fill_tiles[(x, y)] = colour

    @_update_board
    def clear_fill(self, x: int, y: int) -> None:
        self._fill_tiles.pop((x, y), None)

    @_update_board
    def clear_fills(self) -> None:
        self._fill_tiles = {}


class GameWindow(QtWidgets.QWidget):
    """A window that contains a game board and a side menu"""

    def __init__(self, *game_board_args, game_board: GameBoardWidget = None, side_menu: SideMenuWidget = None,
                 title: str = None, parent: QtWidgets.QWidget = None, evaluator: BaseEvaluator = None,
                 use_evaluator=True, verbose=False, action_to_move: Callable[[GameState, int], str] = None,
                 num_best_actions: int = NUM_BEST_ACTIONS, **game_board_kwargs):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setLayout(self.layout)

        side_menu_given = side_menu is not None
        game_board_kwargs['parent'] = self
        self.game_board = game_board or GameBoardWidget(*game_board_args, **game_board_kwargs)
        self.side_menu = side_menu or SideMenuWidget(self)
        self.evaluator = evaluator or MCTSEvaluator(max_search_depth=12, max_search_time=10)
        self.use_evaluator = use_evaluator
        self.verbose = verbose
        self.action_to_move = action_to_move
        self.num_best_actions = num_best_actions

        self.layout.addWidget(self.game_board, 0, 0)
        self.layout.addWidget(self.side_menu, 0, 1)
        self.setFixedSize(self.game_board.width() + self.side_menu.menu_width, self.game_board.height())
        self.side_menu.update()

        if not side_menu_given:
            self.side_menu.add_turn_label()
            if self.use_evaluator:
                self.side_menu.add_eval_bar(animation_speed=40)

        if self.verbose and self.use_evaluator:
            self.side_menu.add_text('Evaluator stats:', font=QtGui.QFont('Arial', 14, QtGui.QFont.Bold))
            font = QtGui.QFont('Arial', 12)
            self.running_label = self.side_menu.add_text('', font=font)
            self.value_label = self.side_menu.add_text('', font=font)
            self.depth_label = self.side_menu.add_text('', font=font)
            self.action_label = self.side_menu.add_text('', font=font)

        if self.use_evaluator:
            self.eval_bar_timer = QtCore.QTimer(self)
            self.eval_bar_timer.setInterval(EVAL_BAR_ANIMATION_UPDATE)
            self.eval_bar_timer.timeout.connect(self.__update_eval_bar)
            self.eval_bar_timer.start()

            self.eval_stats_timer = QtCore.QTimer(self)
            self.eval_stats_timer.setInterval(STATS_UPDATE_INTERVAL)
            self.eval_stats_timer.timeout.connect(self.__update_eval_stats)
            self.eval_stats_timer.start()

    def __update_eval_bar(self):
        if not self.use_evaluator:
            return

        value = self.evaluator.get_value(
            self.evaluator.current_state.player if self.evaluator.current_state is not None else 0
        )
        if value is not None:
            self.side_menu.update_eval_bar(value)

            if self.verbose:
                self.running_label.setText(f'State: {"running" if self.evaluator.is_running else "idle"}')
                self.value_label.setText('Value: %.4f' % value)
                if hasattr(self.evaluator, '_mcts'):
                    self.depth_label.setText('Depth: %d' % self.evaluator._mcts.max_depth)

    def __update_eval_stats(self):
        if not self.use_evaluator or not self.num_best_actions:
            return

        def show_best_actions(actions: list, text: str, keep_text=False):
            previous_text = '' if not keep_text else self.action_label.text() + '\n'
            self.action_label.setText(f'{previous_text}{text}:\n' + '\n'.join(
                str(action) for action in actions[:self.num_best_actions]
            ))

        best_actions = self.evaluator.get_best_actions()
        if best_actions:
            if hasattr(self.evaluator, '_mcts'):
                probs = self.evaluator._mcts.probs(self.evaluator.current_state,
                                                   temp=self.evaluator.best_actions_temp)
            else:
                probs = None

            if callable(self.action_to_move):
                best_actions = [
                    str(self.action_to_move(self.evaluator.current_state, action))
                    + '\t%.4f' % probs[action] if probs is not None else ''
                    for action in best_actions
                ]
            else:
                best_actions = [
                    str(action) + '\t%.4f' % probs[action] if probs is not None else ''
                    for action in best_actions
                ]
            show_best_actions(best_actions, f'Best {self.num_best_actions} moves')
            show_best_actions(best_actions[::-1], f'Worst {self.num_best_actions} moves', keep_text=True)

    def run_evaluator(self, state: GameState, block=False):
        if not self.use_evaluator:
            return
        self.evaluator.stop()
        self.evaluator.run(state, block)

    # def player_moved(self, state: GameState, action: int = None):
    #     print('[DEBUG] player to move:', 2 - state.player, state.player + 1)
    #     self.side_menu.update_turn(state.player + 1)
    #     if self.use_evaluator and action is not None:
    #         self.evaluator.update(state, action)

    def stop_evaluator(self):
        if not self.use_evaluator:
            return
        self.evaluator.stop()
        self.eval_bar_timer.stop()
        self.eval_stats_timer.stop()

    def update(self) -> None:
        self.game_board.update()
        self.side_menu.update()
        super().update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.use_evaluator:
            self.eval_bar_timer.stop()
            self.eval_stats_timer.stop()
            self.evaluator.stop()
        event.accept()


class CustomGUI(ABC):
    def __init__(self, game_cls: GameState, on_player_move: Callable[[int], None] = None,
                 on_window_close: Callable = None, user_input=True, title: str = None,
                 init_state: GameState = None, evaluator: BaseEvaluator = None,
                 show_hints: bool = False):
        self.on_player_move = on_player_move  # Must call this when player moves with the action
        self.on_window_close = on_window_close  # Must call this when window is closed
        self.user_input = user_input
        self.title = title
        self._state = init_state.clone() if init_state else game_cls()
        self.evaluator = evaluator
        self.show_hints = show_hints

    @abstractmethod
    def show(self):
        """Open the custom GUI window"""
        pass

    @abstractmethod
    def close(self):
        """Close the GUI window"""
        pass

    @abstractmethod
    def undo(self):
        """Restore the previous state of the game"""
        pass

    @abstractmethod
    def update_state(self, state: GameState):
        """Update the state of the GUI based on the given game state."""
        self._state = state.clone()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = GameWindow(board_width=7, board_height=6, cell_size=100, grid_line_width=2, title='Connect 4')
    window.game_board.tileClicked.connect(lambda x, y: print(f'Clicked {x}, {y}'))
    window.show()
    app.exec_()
