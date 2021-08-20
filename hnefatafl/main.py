import pyximport
pyximport.install()

import os
import socket
import sys
import random

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QListWidgetItem, QFileDialog

from boardgame.net import Message, byte
from hnefatafl.gui import Ui_MainWindow, Ui_FrmLocalGame, Ui_FrmOnlineConnect, \
    Ui_FrmOnlineGame, Ui_DialogMessage, Ui_FrmListGames, Ui_FrmGameType, show_dialog
from hnefatafl.engine import Board, Game, bool_to_colour, is_turn, variants
from hnefatafl.bots import AlphaZeroBot
from hnefatafl.net import SERVER_ADDR, PORT, TIMEOUT
from hnefatafl.net.client import Client

from functools import partial
from threading import Thread, Lock


class SinglePlayerWindow(Ui_FrmLocalGame):
    def __init__(self, game):
        self.bot_white = True#bool(random.randint(0, 1))
        super().__init__(game=game, playable=True, is_white=not self.bot_white)

        from alphazero.envs.tafl.train_fastafl import args
        args.numMCTSSims = 2000
        #args.temp_scaling_fn = lambda x, y, z: 0.25
        args.add_root_noise = args.add_root_temp = False
        args.cuda = False
        self.bot = AlphaZeroBot(
            game.white if self.bot_white else game.black,
            game.board.to_string(),
            use_default_args=False,
            verbose=True,
            args=args
        )

        self.bot_timer = QtCore.QTimer(self)
        self.bot_timer.setInterval(200)
        self.bot_timer.timeout.connect(self._check_bot_moved)
        self.bot_thread = None
        self.bot_lock = Lock()
        self.init_gameboard(update=True)
        self.gameboard.pieceMoved.connect(self.do_bot_move)
        self.btnSave.clicked.connect(partial(LocalGameWindow.save_game, self))
        self.btnUndo.clicked.connect(self.btn_undo_clicked)
        self.btnExit.clicked.connect(self.close)
        if is_turn(self.bot_white, self.gameboard.game): self.do_bot_move()

    def do_bot_move(self):
        if self.gameboard.game.board.num_turns:
            try:
                self.bot.update(self.gameboard.game.board, self.gameboard.game.board.get_last_move())
            except ValueError as e:
                print(e)

        if not is_turn(self.bot_white, self.gameboard.game): return
        self.bot_thread = Thread(target=self.bot.get_move, args=(self.gameboard.game.board,))
        self.bot_thread.start()
        self.bot_timer.start()

    def stop_bot_thread(self):
        self.bot_timer.stop()
        self.bot_thread.join()
        self.bot_thread = None

    def _check_bot_moved(self):
        self.bot_lock.acquire()
        if self.bot.result is not None:
            self.stop_bot_thread()
            self.gameboard.move_piece(move=self.bot.result, emit_move=False)
            self.bot.update(self.gameboard.game.board, self.bot.result)
            self.gameboard.remove_highlights(remove_killed=True)
            self.gameboard.update()
        self.bot_lock.release()

    def btn_undo_clicked(self):
        self.gameboard.game.undo()
        if self.bot_thread:
            self.stop_bot_thread()
        elif self.gameboard.game.board.num_turns:
            self.gameboard.game.undo()

        self.gameboard.remove_highlights()
        self.gameboard.update()
        self.bot.reset()

        white = self.gameboard.game.white
        black = self.gameboard.game.black
        if white.is_turn and white.bot or black.is_turn and black.bot: self.do_bot_move()


class LocalGameWindow(Ui_FrmLocalGame):
    def __init__(self, game, playable=True):
        super().__init__(game=game, playable=playable)
        self.init_gameboard(update=True)
        if self.gameboard.playable:
            self.btnUndo.clicked.connect(self.btn_undo_clicked)
            self.btnSave.clicked.connect(self.save_game)
        else:
            self.setWindowTitle('Hnefatafl - Game Preview')
            self.btnUndo.setText('Play')
            self.btnSave.deleteLater()
            self.btnSave = None
            self.lblTurn.setVisible(False)
            self.lblBlackPieces.setVisible(False)
            self.lblWhitePieces.setVisible(False)
        self.btnExit.clicked.connect(self.close)

    def save_game(self):
        name = QFileDialog.getSaveFileName(self, 'Save Board', directory=os.path.abspath(os.getcwd()))[0]
        if name:
            try:
                save_game(self.gameboard.game, name)
            except IOError as e:
                show_dialog(f'Error saving board: {e}', self, 'Error', error=True)

    def btn_undo_clicked(self):
        self.gameboard.game.undo()
        self.gameboard.remove_highlights()
        self.gameboard.update()


def save_game(game: Game, path: str):
    with open(os.path.abspath(path), 'wb') as f:
        f.write(game.serialize())


def load_game(path: str) -> Game:
    with open(os.path.abspath(path), 'rb') as f:
        return Game.from_serial(f.read())


def file_dialog(parent) -> str or None:
    dialog = QFileDialog(parent, directory=os.path.abspath(os.getcwd()))
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setViewMode(QFileDialog.Detail)
    if dialog.exec_() == QFileDialog.Accepted:
        return dialog.selectedFiles()[0]


def custom_game(parent):
    path = file_dialog(parent)
    if path:
        try:
            return Game(Board(load_file=path, custom=True))
        except Exception as e:
            show_dialog(f'The file selected is not a valid board file. ({e})', parent, 'Error', error=True)


class GameTypeWindow(Ui_FrmGameType):
    def __init__(self):
        super().__init__()
        self.btnDefault.clicked.connect(self.close)
        self.btnCustom.clicked.connect(self.close)
        self.btnLoad.clicked.connect(self.close)
        self.btnBack.clicked.connect(self.close)


def server_error_dialog(error: str, parent):
    show_dialog(f'Failed to send to server:\n{error}', parent, 'Error', error=True)


class OnlineGameWindow(Ui_FrmOnlineGame):
    def __init__(self, client: Client):
        super().__init__(client, game=client.game)
        if client.in_game:
            self.init_gameboard()
            self.gameboard.update()
        self.btnSend.clicked.connect(self.btn_send_clicked)
        self.txtChat.returnPressed.connect(self.btn_send_clicked)
        self.btnExit.clicked.connect(self.close)
        self.__timer = QtCore.QTimer(self)
        self.__timer.setInterval(100)
        self.__timer.timeout.connect(self.__msg_update)
        self.__timer.start()

    def __msg_update(self):
        try:
            msg = self.client.recv_msg()
        except (BlockingIOError, BrokenPipeError):
            return

        if msg.startswith(Message.Exit.value):
            show_dialog('Server closed.', self, 'Error', error=True)
            self.close()

        elif msg.startswith(Message.Colour.value):
            self.gameboard.is_white = self.client.is_white
            self.display_colour()

        elif msg.startswith(Message.Error.value):
            error = Message(byte(msg[1]))
            if error == Message.ErrorPlayerLeft:
                show_dialog('The other player left. Waiting for someone to join.', self, 'Player Left')
            else:
                show_dialog(f'Error: {error}', self, 'Error', error=True)
                self.close()

        elif msg.startswith(Message.GameUpdate.value) or msg.startswith(Message.Game.value):
            self.gameboard.game = self.client.game
            self.gameboard.remove_highlights()
            if not len(self.gameboard.buttons): self.init_gameboard()
            self.gameboard.update()

        elif msg.startswith(Message.Chat.value):
            self.display_chat(msg[1:].decode())

    def display_chat(self, text: str):
        item = QListWidgetItem(text)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        item.setFont(font)
        self.lstChat.addItem(item)
        self.lstChat.scrollToBottom()

    def display_colour(self):
        show_dialog(f"You are {bool_to_colour(self.client.is_white)}.", self, 'Game Start', modal=False)

    def btn_send_clicked(self):
        try:
            text = self.txtChat.text()
            if text: self.client.send_chat(text)
            self.txtChat.setText('')
        except socket.error as e:
            server_error_dialog(e, self)

    def close(self) -> bool:
        self.__timer.stop()
        self.client.exit()
        return super().close()


class ListGamesWindow(Ui_FrmListGames):
    playGame = pyqtSignal(Game)

    def __init__(self, client: Client):
        super().__init__()
        self.client = client
        self.game = None
        self.preview_window = None
        self.lstGames.itemClicked.connect(self.item_clicked)
        self.btnRefresh.clicked.connect(self.show_games)
        self.btnBack.clicked.connect(self.close)
        self.show_games()

    def show_games(self):
        self.lstGames.clear()
        try:
            games = list(map(lambda x: (x[0], Game.from_serial(x[1])), self.client.search_games()))
        except socket.error as e:
            server_error_dialog(e, self)
        else:
            font = QtGui.QFont()
            font.setFamily("Trebuchet MS")
            font.setPointSize(16)
            for username, game in games:
                text = f"{username}'s {'custom' if game.is_custom else 'default'} game"
                item = QListWidgetItem(text, self.parent())
                item.username = username
                item.game = game
                item.setFont(font)
                self.lstGames.addItem(item)

    def item_clicked(self, item):
        self.game = item.game
        self.preview_window = LocalGameWindow(self.game, playable=False)
        self.preview_window.btnUndo.clicked.connect(self.play_clicked)
        self.preview_window.btnExit.clicked.connect(self.show)
        self.preview_window.lblTurn.setText(item.username)
        self.preview_window.show()
        self.hide()

    def play_clicked(self):
        self.preview_window.close()
        self.playGame.emit(self.game)
        self.close()


class WaitWindow(Ui_DialogMessage):
    ready = pyqtSignal(Game)
    showPrevious = pyqtSignal()

    def __init__(self, client: Client):
        super().__init__()
        self.client = client
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(10, 100, 381, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.rejected.connect(self.cancel_clicked)
        QtCore.QMetaObject.connectSlotsByName(self)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(16)
        self.lblMsg.setFont(font)
        self.lblMsg.setText('Waiting for player to join...')
        self.setWindowTitle('Waiting')
        self.__timer = QtCore.QTimer(self)
        self.__timer.setInterval(500)
        self.__timer.timeout.connect(self.__await_game_start)
        self.__timer.start()

    def __await_game_start(self):
        try:
            msg = self.client.recv_msg()
        except (BlockingIOError, BrokenPipeError):
            return

        if msg.startswith(Message.Exit.value):
            server_error_dialog('Server closed.', self)
            self.showPrevious.emit()
            self.close()
        elif msg.startswith(Message.Game.value):
            self.ready.emit(self.client.game)
            self.close()

    def cancel_clicked(self):
        self.__timer.stop()
        self.client.cancel_game()
        self.showPrevious.emit()
        self.close()

    def close(self) -> bool:
        self.__timer.stop()
        return super().close()


class OnlineConnectWindow(Ui_FrmOnlineConnect):
    openMenu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.client = None
        self.gamelist_window = None
        self.wait_window = None
        self.game_window = None
        self.btnSearch.clicked.connect(self.btn_search_clicked)
        self.btnDefault.clicked.connect(self.create_game)
        self.btnCustom.clicked.connect(self.btn_custom_clicked)
        self.btnQuit.clicked.connect(self.close)

    def create_client(self) -> bool:
        username = self.txtUsername.text()
        if not username:
            show_dialog('Invalid username entered. Username must not be empty', self, 'Error', error=True)
            return False

        if self.client: self.client.exit()
        self.client = Client(username)
        self.client.settimeout(TIMEOUT)
        addr = self.txtAddress.text()
        port = self.txtPort.text()
        if not addr:
            addr = SERVER_ADDR
        if not port:
            port = PORT
        try:
            port = int(port)
        except ValueError:
            show_dialog('Invalid port entered.', self, 'Error', error=True)
            return False

        try:
            self.client.connect((addr, port))
        except BlockingIOError:
            pass
        except socket.error as e:
            server_error_dialog(e, self)
            return False

        try:
            msg = self.client.recv_msg(True)
        except socket.error as e:
            server_error_dialog(e, self)
            return False

        if msg.startswith(Message.Error.value):
            if msg.endswith(Message.ErrorUserExists.value):
                show_dialog('Invalid username entered, that user already exists.', self, 'Error', error=True)
            else:
                server_error_dialog(str(Message(byte(msg[1]))), self)
            return False
        else:
            return True

    def create_game(self, game=None):
        if not self.create_client(): return
        try:
            self.client.create_game(game)
        except socket.error as e:
            server_error_dialog(e, self)
        else:
            self.wait_window = WaitWindow(self.client)
            self.wait_window.ready.connect(self.start_game)
            self.wait_window.showPrevious.connect(self.show)
            self.wait_window.show()
            self.hide()

    def start_game(self, game: Game):
        if not self.client.in_game: self.client.game = game
        self.game_window = OnlineGameWindow(self.client)
        self.game_window.closing.connect(self.show)
        self.game_window.show()
        self.hide()
        if self.client.is_white is not None: self.game_window.display_colour()

    def join_game(self, game: Game):
        if not self.client.in_game:
            try:
                self.client.join_game(game)
            except socket.error as e:
                server_error_dialog(e, self)
            else:
                self.start_game(game)

    def btn_search_clicked(self):
        if not self.create_client(): return
        self.gamelist_window = ListGamesWindow(self.client)
        self.gamelist_window.btnBack.clicked.connect(self.show)
        self.gamelist_window.playGame.connect(self.join_game)
        self.gamelist_window.show()
        self.hide()

    def btn_custom_clicked(self):
        game = custom_game(self)
        if game: self.create_game(game)

    def close(self) -> bool:
        self.openMenu.emit()
        if self.client: self.client.exit()
        return super().close()


class MainWindow(Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_type_window = None
        self.online_conn_window = None
        self.game_window = None
        self.btnSingle.clicked.connect(self.btn_single_clicked)
        self.btnLocal.clicked.connect(self.btn_local_clicked)
        self.btnOnline.clicked.connect(self.btn_online_clicked)
        self.btnQuit.clicked.connect(self.close)

    def btn_single_clicked(self):
        try:
            self.game_window = SinglePlayerWindow(Game(Board(variants.brandubh)))
        except IOError as e:
            show_dialog(f'Failed to start game: {e}', self, 'Error', error=True)
        else:
            self.game_window.closing.connect(self.show)
            self.game_window.show()
            self.hide()

    def btn_local_clicked(self):
        self.game_type_window = GameTypeWindow()
        self.game_type_window.btnDefault.clicked.connect(self.local_default_game)
        self.game_type_window.btnCustom.clicked.connect(self.local_custom_game)
        self.game_type_window.btnLoad.clicked.connect(self.local_load_game)
        self.game_type_window.btnBack.clicked.connect(self.show)
        self.game_type_window.show()
        self.hide()

    def btn_online_clicked(self):
        self.online_conn_window = OnlineConnectWindow()
        self.online_conn_window.openMenu.connect(self.show)
        self.online_conn_window.show()
        self.hide()

    def start_game(self, game):
        self.game_window = LocalGameWindow(game)
        self.game_window.closing.connect(self.show)
        self.game_window.show()
        self.hide()

    def local_default_game(self):
        try:
            self.start_game(Game())
        except IOError as e:
            show_dialog(f'Failed to start game: {e}', self, 'Error', error=True)
            self.game_type_window.show()

    def local_custom_game(self):
        self.show()
        game = custom_game(self)
        if game: self.start_game(game)

    def local_load_game(self):
        self.show()
        path = file_dialog(self)
        if path: game = load_game(path)
        else: return
        if game: self.start_game(game)


def main():
    # Handle high resolution displays:
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
