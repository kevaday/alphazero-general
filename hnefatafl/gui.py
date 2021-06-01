# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal

from boardgame import Win, Move
from boardgame.errors import BoardGameException

from hnefatafl import ICON, PNG_ICON, PNG_ICON2
from hnefatafl.engine.board import Tile
from hnefatafl.engine.game import Game, is_turn, bool_to_colour
from hnefatafl.net.client import Client


# Form implementation generated from reading ui file 'mainmenu.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(QtGui.QIcon(ICON))
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setupUi(self)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.drawPixmap(QtCore.QRect(15, 20, 120, 176), QtGui.QPixmap(PNG_ICON))
        # painter.drawPixmap(QtCore.QRect(350, 20, 135, 148), QtGui.QPixmap(PNG_ICON2))

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(496, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        btnfont = QtGui.QFont()
        btnfont.setFamily("Arial Black")
        btnfont.setPointSize(14)
        btnfont.setBold(True)
        btnfont.setWeight(75)
        self.btnSingle = QtWidgets.QPushButton(self.centralwidget)
        self.btnSingle.setGeometry(QtCore.QRect(160, 170, 161, 81))
        self.btnSingle.setFont(btnfont)
        self.btnSingle.setObjectName('btnSingle')
        self.btnLocal = QtWidgets.QPushButton(self.centralwidget)
        self.btnLocal.setGeometry(QtCore.QRect(160, 270, 161, 81))
        self.btnLocal.setFont(btnfont)
        self.btnLocal.setObjectName("btnLocal")
        self.btnOnline = QtWidgets.QPushButton(self.centralwidget)
        self.btnOnline.setGeometry(QtCore.QRect(160, 370, 161, 81))
        self.btnOnline.setFont(btnfont)
        self.btnOnline.setObjectName("btnOnline")
        self.btnHelp = QtWidgets.QPushButton(self.centralwidget)
        self.btnHelp.setGeometry(QtCore.QRect(180, 510, 111, 41))
        self.btnHelp.setFont(btnfont)
        self.btnHelp.setObjectName('btnHelp')
        self.btnQuit = QtWidgets.QPushButton(self.centralwidget)
        self.btnQuit.setGeometry(QtCore.QRect(180, 570, 111, 41))
        self.btnQuit.setFont(btnfont)
        self.btnQuit.setObjectName("btnQuit")
        self.lblTitle = QtWidgets.QLabel(self.centralwidget)
        self.lblTitle.setGeometry(QtCore.QRect(135, 20, 221, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.lblTitle.setFont(font)
        self.lblTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTitle.setObjectName("lblTitle")
        self.lblSubtitle = QtWidgets.QLabel(self.centralwidget)
        self.lblSubtitle.setGeometry(QtCore.QRect(135, 80, 221, 35))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(20)
        self.lblSubtitle.setFont(font)
        self.lblSubtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSubtitle.setObjectName("lblSubtitle")
        self.lblAuthor = QtWidgets.QLabel(self.centralwidget)
        self.lblAuthor.setGeometry(QtCore.QRect(180, 125, 121, 21))
        font.setPointSize(16)
        self.lblAuthor.setFont(font)
        self.lblSubtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSubtitle.setObjectName("lblAuthor")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 496, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hnefatafl - Main Menu"))
        self.btnSingle.setText(_translate('MainWindow', 'Single Player'))
        self.btnLocal.setText(_translate("MainWindow", "Local Game"))
        self.lblTitle.setText(_translate("MainWindow", "Hnefatafl"))
        self.lblSubtitle.setText(_translate("MainWindow", "Viking Chess"))
        self.lblAuthor.setText(_translate('MainWindow', 'By Kevi Aday'))
        self.btnOnline.setText(_translate("MainWindow", "Online Game"))
        self.btnQuit.setText(_translate("MainWindow", "Quit"))
        self.btnHelp.setText(_translate('MainWindow', 'Help'))


class GameWidget(QtWidgets.QWidget):
    closing = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(QtGui.QIcon(ICON))
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closing.emit()
        a0.accept()


class BoardButton(QtWidgets.QPushButton):
    tilePressed = pyqtSignal(Tile)

    def __init__(self, tile: Tile = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(True)
        self.setCheckable(False)
        self.setDefault(False)
        self.setFlat(True)
        self.raise_()
        self.tile = tile
        self.default_background(tile)
        self.highlight = False
        self.move_highlight = False
        self.piece_killed = False
        self.is_clicked = False

    def __change_bgd(self, colour: QtGui.QColor):
        p = self.palette()
        p.setColor(self.backgroundRole(), colour)
        self.setPalette(p)

        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtGui.QBrush(colour))
        painter.fillRect(self.rect(), painter.brush())
        painter.end()
        """
        # self.setStyleSheet(f'background-color: rgb({colour.red()}, {colour.green()}, {colour.blue()});')

    def default_background(self, tile: Tile = None):
        if (tile == self.tile and self.tile is not None) or tile is None:
            self.__change_bgd(QtGui.QColor(153, 76, 0))

    def highlight_background(self, tile: Tile = None):
        if (tile == self.tile and self.tile is not None) or tile is None:
            self.__change_bgd(QtGui.QColor(QtCore.Qt.green))

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self.tilePressed.emit(self.tile)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        if self.highlight:
            self.highlight_background()
        elif self.move_highlight:
            self.__change_bgd(QtGui.QColor(0, 130, 0))
        else:
            self.default_background()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtCore.Qt.black, QtCore.Qt.SolidLine)
        pen.setWidth(1)
        painter.setPen(pen)
        if self.piece_killed:
            width_dist = self.width() // 8
            height_dist = self.height() // 8
            painter.drawLine(width_dist, height_dist, self.width() - width_dist, self.height() - height_dist)
            painter.drawLine(width_dist, self.height() - height_dist, self.width() - width_dist, height_dist)

        # pen.setWidth(1)
        # painter.setPen(pen)
        painter.drawRect(0, 0, self.width(), self.height())
        if self.tile:
            if self.tile.is_special:
                first_line = self.width() // 3
                second_line = first_line * 2
                painter.drawLine(first_line, 0, first_line, self.height())
                painter.drawLine(second_line, 0, second_line, self.height())
                painter.drawLine(0, first_line, self.width(), first_line)
                painter.drawLine(0, second_line, self.width(), second_line)
            if self.tile.is_exit:
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(self.width() // 4, self.height() // 4, self.width() // 2, self.height() // 2)
            if self.tile.piece is not None:
                if self.tile.piece.is_white:
                    colour = QtCore.Qt.white
                else:
                    colour = QtCore.Qt.black
                painter.setBrush(QtGui.QBrush(colour, QtCore.Qt.SolidPattern))
                painter.drawEllipse(0, 0, self.width(), self.height())
                if self.tile.piece.is_king:
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.SolidPattern))
                    painter.drawEllipse(self.width() // 4, self.height() // 4, self.width() // 2, self.height() // 2)


def display_board(board, parent: QtWidgets.QWidget = None):
    gameboard = GameBoard(Game(board), playable=False, parent=parent)
    gameboard.init_buttons()
    gameboard.show()

    return gameboard


class GameBoard(QtWidgets.QWidget):
    boardUpdate = pyqtSignal()
    tilePressed = pyqtSignal(Tile)
    pieceMoved = pyqtSignal(Move)

    def __init__(self, game: Game = None, is_white: bool = None, playable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setGeometry(QtCore.QRect(20, 20, 650, 650))
        self.setObjectName("boardWidget")
        self.game = game
        self.is_white = is_white
        self.playable = playable
        self.buttons = []
        self.__selected = None
        self.tilePressed.connect(self.__tile_pressed)

    def init_buttons(self):
        def pos_to_name(x, y):
            return f'btn_{x}{y}'

        self.buttons = []
        board = self.game.board
        board_size = board.width
        if board.width > board.height:
            board_size = board.width
        elif board.width < board.height:
            board_size = board.height
        btn_size = self.width() / board_size
        start_x = self.width() / 2 - btn_size * (board.width / 2)
        start_y = self.height() / 2 - btn_size * (board.height / 2)

        for y in range(board.height):
            row = []
            for x in range(board.width):
                button = BoardButton(board[y][x], parent=self)
                button.setGeometry(QtCore.QRect(start_x + x * btn_size, start_y + y * btn_size, btn_size, btn_size))
                button.setObjectName(pos_to_name(x, y))
                button.tilePressed.connect(self.tilePressed.emit)
                row.append(button)
            self.buttons.append(row)

    def __get_button(self, tile: Tile) -> BoardButton:
        return self.buttons[tile.y][tile.x]

    def update(self) -> None:
        self.boardUpdate.emit()
        board = self.game.board
        last_move = board.get_last_move()
        tile = new_tile = None
        if last_move: tile, new_tile = last_move.tile, last_move.new_tile
        for y, row in enumerate(self.buttons):
            for x, button in enumerate(row):
                button.tile = board[y][x]
                if last_move:
                    button.move_highlight = (button.tile == tile or button.tile == new_tile)

                for piece in board.killed_pieces:
                    button.piece_killed = (board[piece.y][piece.x] == button.tile)

                button.update()
        super().update()

    """
    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        board = self.game.board
        board_size = board.width
        if board.width > board.height:
            board_size = board.width
        elif board.width < board.height:
            board_size = board.height
        btn_size = self.width() / board_size
        start_x = self.width() / 2 - btn_size * (board.width / 2)
        start_y = self.height() / 2 - btn_size * (board.height / 2)

        for y in range(board.height):
            for x in range(board.width):
                tile_rect = QtCore.QRect(start_x + x * btn_size, start_y + y * btn_size, btn_size, btn_size)
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                pen = QtGui.QPen(QtCore.Qt.black, QtCore.Qt.SolidLine)
                pen.setWidth(1)
                painter.setPen(pen)

                painter.fillRect(tile_rect, )

                if self.piece_killed:
                    width_dist = self.width() // 8
                    height_dist = self.height() // 8
                    painter.drawLine(width_dist, height_dist, self.width() - width_dist, self.height() - height_dist)
                    painter.drawLine(width_dist, self.height() - height_dist, self.width() - width_dist, height_dist)

                # pen.setWidth(1)
                # painter.setPen(pen)
                painter.drawRect(0, 0, self.width(), self.height())
                if self.tile:
                    if self.tile.is_special:
                        first_line = self.width() // 3
                        second_line = first_line * 2
                        painter.drawLine(first_line, 0, first_line, self.height())
                        painter.drawLine(second_line, 0, second_line, self.height())
                        painter.drawLine(0, first_line, self.width(), first_line)
                        painter.drawLine(0, second_line, self.width(), second_line)
                    if self.tile.is_exit:
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.drawEllipse(self.width() // 4, self.height() // 4, self.width() // 2,
                                            self.height() // 2)
                    if self.tile.piece is not None:
                        if self.tile.piece.is_white:
                            colour = QtCore.Qt.white
                        else:
                            colour = QtCore.Qt.black
                        painter.setBrush(QtGui.QBrush(colour, QtCore.Qt.SolidPattern))
                        painter.drawEllipse(0, 0, self.width(), self.height())
                        if self.tile.piece.is_king:
                            painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.SolidPattern))
                            painter.drawEllipse(self.width() // 4, self.height() // 4, self.width() // 2,
                                                self.height() // 2)
    """

    def remove_highlights(self):
        self.__selected = None
        for row in self.buttons:
            for button in row:
                button.highlight = False
                button.move_highlight = False
                button.piece_killed = False
                button.is_clicked = False

    def highlight_buttons(self, source_tile: Tile):
        self.__selected = source_tile
        for move in self.game.board.valid_moves(source_tile):
            self.buttons[move.new_tile.y][move.new_tile.x].highlight = True

    def move_piece(self, target_tile: Tile = None, move: Move = None, emit_move=True):
        assert target_tile is not None or move is not None
        assert not (target_tile is not None and move is not None)
        if target_tile and not self.__selected:
            raise ValueError('Cannot move piece because no piece is selected.')
        if target_tile:
            move = Move(self.game.board, self.__selected, target_tile)
        try: self.game.move(move)
        except Win: pass
        if emit_move: self.pieceMoved.emit(move)

    def is_highlight(self, tile: Tile) -> bool:
        return self.__get_button(tile).highlight

    def is_clicked(self, tile: Tile) -> bool:
        return self.__get_button(tile).is_clicked

    def set_clicked(self, tile: Tile, value: bool):
        self.__get_button(tile).is_clicked = value

    def __tile_pressed(self, tile: Tile):
        if not tile or self.game.game_over or not self.playable: return
        if self.is_white is not None and not is_turn(self.is_white, self.game): return

        if not tile.piece and self.is_highlight(tile):
            try: self.move_piece(tile)
            except (ValueError, BoardGameException): pass
            self.remove_highlights()

        elif not tile.piece and not self.is_highlight(tile):
            self.remove_highlights()

        elif tile.piece:
            if self.is_clicked(tile):
                self.remove_highlights()
            elif self.game.is_turn(tile):
                self.remove_highlights()
                self.highlight_buttons(tile)
                self.set_clicked(tile, True)

        self.update()


class _GameBoardWindow(GameWidget):
    def __init__(self, client: Client = None, *args, **kwargs):
        super().__init__()
        self.client = client
        if client: self.gameboard = GameBoard(is_white=client.is_white, *args, **kwargs, parent=self)
        else: self.gameboard = GameBoard(*args, **kwargs, parent=self)
        self.gameboard.boardUpdate.connect(self._update_labels)
        self.gameboard.pieceMoved.connect(self.__piece_moved)

    def init_gameboard(self, game: Game = None, update=False):
        if game: self.gameboard.game = game
        self.gameboard.init_buttons()
        if update: self.gameboard.update()

    def _update_labels(self):
        game = self.gameboard.game
        if not game: return
        if not game.game_over: text = f"{bool_to_colour(game.white.is_turn).capitalize()}'s Turn"
        else:
            winner = bool_to_colour(game.white.won).capitalize()
            text = f'{winner} Won!'
            show_dialog(f'{winner} has won the game!', self, 'Game Over!')
        self.lblTurn.setText(text)
        self.lblBlackPieces.setText(f'Black: {game.board.num_black}/{game.board.num_start_black}')
        self.lblWhitePieces.setText(f'White: {game.board.num_white}/{game.board.num_start_white}')

    def __piece_moved(self, move: Move):
        if self.client: self.client.send_update(move)


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'localgame.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_FrmLocalGame(_GameBoardWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, FrmLocalGame):
        FrmLocalGame.setObjectName("FrmLocalGame")
        FrmLocalGame.resize(700, 800)
        self.btnUndo = QtWidgets.QPushButton(FrmLocalGame)
        self.btnUndo.setGeometry(QtCore.QRect(20, 690, 161, 81))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnUndo.setFont(font)
        self.btnUndo.setObjectName("btnUndo")
        self.btnSave = QtWidgets.QPushButton(FrmLocalGame)
        self.btnSave.setGeometry(QtCore.QRect(460, 690, 100, 81))
        self.btnSave.setFont(font)
        self.btnSave.setObjectName('btnSave')
        self.btnExit = QtWidgets.QPushButton(FrmLocalGame)
        self.btnExit.setGeometry(QtCore.QRect(571, 690, 100, 81))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnExit.setFont(font)
        self.btnExit.setObjectName("btnExit")
        self.verticalLayoutWidget = QtWidgets.QWidget(FrmLocalGame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(260, 680, 160, 99))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.lblVerticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.lblVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lblVerticalLayout.setObjectName("lblVerticalLayout")
        self.lblTurn = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(22)
        self.lblTurn.setFont(font)
        self.lblTurn.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTurn.setObjectName("lblTurn")
        self.lblVerticalLayout.addWidget(self.lblTurn)
        self.lblBlackPieces = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.lblBlackPieces.setFont(font)
        self.lblBlackPieces.setAlignment(QtCore.Qt.AlignCenter)
        self.lblBlackPieces.setObjectName("lblBlackPieces")
        self.lblVerticalLayout.addWidget(self.lblBlackPieces)
        self.lblWhitePieces = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.lblWhitePieces.setFont(font)
        self.lblWhitePieces.setAlignment(QtCore.Qt.AlignCenter)
        self.lblWhitePieces.setObjectName("lblWhitePieces")
        self.lblVerticalLayout.addWidget(self.lblWhitePieces)

        self.retranslateUi(FrmLocalGame)
        QtCore.QMetaObject.connectSlotsByName(FrmLocalGame)

    def retranslateUi(self, FrmLocalGame):
        _translate = QtCore.QCoreApplication.translate
        FrmLocalGame.setWindowTitle(_translate("FrmLocalGame", "Hnefatafl - Local Game"))
        self.btnUndo.setText(_translate("FrmLocalGame", "Undo"))
        self.btnSave.setText(_translate('FrmLocalGame', 'Save'))
        self.btnExit.setText(_translate("FrmLocalGame", "Quit"))
        self.lblTurn.setText(_translate("FrmLocalGame", "Black\'s Turn"))
        self.lblBlackPieces.setText(_translate("FrmLocalGame", "<html><head/><body><p>Black: 24/24</p></body></html>"))
        self.lblWhitePieces.setText(_translate("FrmLocalGame", "<html><head/><body><p>White: 13/13</p></body></html>"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'onlinegameconnect.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_FrmOnlineConnect(GameWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, FrmOnlineConnect):
        FrmOnlineConnect.setObjectName("FrmOnlineConnect")
        FrmOnlineConnect.resize(480, 496)
        self.lblTitle = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblTitle.setGeometry(QtCore.QRect(80, 30, 301, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.lblTitle.setFont(font)
        self.lblTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTitle.setObjectName("lblTitle")
        self.lblServerAddr = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblServerAddr.setGeometry(QtCore.QRect(30, 120, 391, 21))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(16)
        self.lblServerAddr.setFont(font)
        self.lblServerAddr.setAlignment(QtCore.Qt.AlignCenter)
        self.lblServerAddr.setObjectName("lblServerAddr")
        self.txtAddress = QtWidgets.QLineEdit(FrmOnlineConnect)
        self.txtAddress.setGeometry(QtCore.QRect(110, 150, 161, 20))
        self.txtAddress.setObjectName("txtAddress")
        self.lblAddress = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblAddress.setGeometry(QtCore.QRect(40, 150, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblAddress.setFont(font)
        self.lblAddress.setObjectName("lblAddress")
        self.lblPort = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblPort.setGeometry(QtCore.QRect(280, 150, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblPort.setFont(font)
        self.lblPort.setObjectName("lblPort")
        self.txtPort = QtWidgets.QLineEdit(FrmOnlineConnect)
        self.txtPort.setGeometry(QtCore.QRect(320, 150, 81, 20))
        self.txtPort.setObjectName("txtPort")
        self.btnCustom = QtWidgets.QPushButton(FrmOnlineConnect)
        self.btnCustom.setGeometry(QtCore.QRect(150, 430, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnCustom.setFont(font)
        self.btnCustom.setObjectName("btnCustom")
        self.btnDefault = QtWidgets.QPushButton(FrmOnlineConnect)
        self.btnDefault.setGeometry(QtCore.QRect(150, 370, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnDefault.setFont(font)
        self.btnDefault.setObjectName("btnDefault")
        self.btnQuit = QtWidgets.QPushButton(FrmOnlineConnect)
        self.btnQuit.setGeometry(QtCore.QRect(380, 450, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnQuit.setFont(font)
        self.btnQuit.setObjectName("btnQuit")
        self.lblCreateGame = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblCreateGame.setGeometry(QtCore.QRect(140, 330, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(16)
        self.lblCreateGame.setFont(font)
        self.lblCreateGame.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCreateGame.setObjectName("lblCreateGame")
        self.btnSearch = QtWidgets.QPushButton(FrmOnlineConnect)
        self.btnSearch.setGeometry(QtCore.QRect(150, 250, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnSearch.setFont(font)
        self.btnSearch.setObjectName("btnSearch")
        self.lblUsername = QtWidgets.QLabel(FrmOnlineConnect)
        self.lblUsername.setGeometry(QtCore.QRect(120, 200, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblUsername.setFont(font)
        self.lblUsername.setObjectName("lblUsername")
        self.txtUsername = QtWidgets.QLineEdit(FrmOnlineConnect)
        self.txtUsername.setGeometry(QtCore.QRect(210, 200, 131, 20))
        self.txtUsername.setObjectName("txtUsername")

        self.retranslateUi(FrmOnlineConnect)
        QtCore.QMetaObject.connectSlotsByName(FrmOnlineConnect)

    def retranslateUi(self, FrmOnlineConnect):
        _translate = QtCore.QCoreApplication.translate
        FrmOnlineConnect.setWindowTitle(_translate("FrmOnlineConnect", "Hnefatafl - Connect to Game"))
        self.lblTitle.setText(_translate("FrmOnlineConnect", "Online Game"))
        self.lblServerAddr.setText(_translate("FrmOnlineConnect", "Server Address (leave blank for default):"))
        self.lblAddress.setText(_translate("FrmOnlineConnect", "Address:"))
        self.lblPort.setText(_translate("FrmOnlineConnect", "Port:"))
        self.btnCustom.setText(_translate("FrmOnlineConnect", "Custom Game"))
        self.btnDefault.setText(_translate("FrmOnlineConnect", "Default Game"))
        self.btnQuit.setText(_translate("FrmOnlineConnect", "Cancel"))
        self.lblCreateGame.setText(_translate("FrmOnlineConnect", "Create Game:"))
        self.btnSearch.setText(_translate("FrmOnlineConnect", "Search Games"))
        self.lblUsername.setText(_translate("FrmOnlineConnect", "Username:"))
        # self.txtUsername.setText(_translate("FrmOnlineConnect", "Player"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'listgames.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_FrmListGames(GameWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, FrmListGames):
        FrmListGames.setObjectName("FrmListGames")
        FrmListGames.resize(480, 640)
        self.lstGames = QtWidgets.QListWidget(FrmListGames)
        self.lstGames.setGeometry(QtCore.QRect(10, 10, 461, 571))
        self.lstGames.setObjectName("lstGames")
        self.btnBack = QtWidgets.QPushButton(FrmListGames)
        self.btnBack.setGeometry(QtCore.QRect(250, 590, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnBack.setFont(font)
        self.btnBack.setObjectName("btnBack")
        self.btnRefresh = QtWidgets.QPushButton(FrmListGames)
        self.btnRefresh.setGeometry(QtCore.QRect(150, 590, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnRefresh.setFont(font)
        self.btnRefresh.setObjectName("btnRefresh")

        self.retranslateUi(FrmListGames)
        QtCore.QMetaObject.connectSlotsByName(FrmListGames)

    def retranslateUi(self, FrmListGames):
        _translate = QtCore.QCoreApplication.translate
        FrmListGames.setWindowTitle(_translate("FrmListGames", "Hnefatafl - Available Games"))
        self.btnBack.setText(_translate("FrmListGames", "Back"))
        self.btnRefresh.setText(_translate("FrmListGames", "Refresh"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'onlinegame.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_FrmOnlineGame(_GameBoardWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, FrmOnlineGame):
        FrmOnlineGame.setObjectName("FrmOnlineGame")
        FrmOnlineGame.resize(1000, 800)
        self.verticalLayoutWidget = QtWidgets.QWidget(FrmOnlineGame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(240, 680, 160, 99))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.lblVerticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.lblVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lblVerticalLayout.setObjectName("lblVerticalLayout")
        self.lblTurn = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(22)
        self.lblTurn.setFont(font)
        self.lblTurn.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTurn.setObjectName("lblTurn")
        self.lblVerticalLayout.addWidget(self.lblTurn)
        self.lblBlackPieces = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.lblBlackPieces.setFont(font)
        self.lblBlackPieces.setAlignment(QtCore.Qt.AlignCenter)
        self.lblBlackPieces.setObjectName("lblBlackPieces")
        self.lblVerticalLayout.addWidget(self.lblBlackPieces)
        self.lblWhitePieces = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.lblWhitePieces.setFont(font)
        self.lblWhitePieces.setAlignment(QtCore.Qt.AlignCenter)
        self.lblWhitePieces.setObjectName("lblWhitePieces")
        self.lblVerticalLayout.addWidget(self.lblWhitePieces)
        self.btnExit = QtWidgets.QPushButton(FrmOnlineGame)
        self.btnExit.setGeometry(QtCore.QRect(20, 690, 161, 81))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnExit.setFont(font)
        self.btnExit.setObjectName("btnExit")
        self.lstChat = QtWidgets.QListWidget(FrmOnlineGame)
        self.lstChat.setGeometry(QtCore.QRect(690, 20, 281, 651))
        # self.lstChat.setWrapping(True)
        self.lstChat.setObjectName("lstChat")
        self.txtChat = QtWidgets.QLineEdit(FrmOnlineGame)
        self.txtChat.setGeometry(QtCore.QRect(690, 680, 281, 31))
        self.txtChat.setObjectName("txtChat")
        self.btnSend = QtWidgets.QPushButton(FrmOnlineGame)
        self.btnSend.setGeometry(QtCore.QRect(770, 720, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnSend.setFont(font)
        self.btnSend.setObjectName("btnSend")

        self.retranslateUi(FrmOnlineGame)
        QtCore.QMetaObject.connectSlotsByName(FrmOnlineGame)

    def retranslateUi(self, FrmOnlineGame):
        _translate = QtCore.QCoreApplication.translate
        FrmOnlineGame.setWindowTitle(_translate("FrmOnlineGame", "Hnefatafl - Online Game"))
        self.lblTurn.setText(_translate("FrmOnlineGame", "Black\'s Turn"))
        self.lblBlackPieces.setText(_translate("FrmOnlineGame", "<html><head/><body><p>Black: 24/24</p></body></html>"))
        self.lblWhitePieces.setText(_translate("FrmOnlineGame", "<html><head/><body><p>White: 13/13</p></body></html>"))
        self.btnExit.setText(_translate("FrmOnlineGame", "Quit to Menu"))
        self.btnSend.setText(_translate("FrmOnlineGame", "Send Chat"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'msgdialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_DialogMessage(GameWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, DialogMessage):
        DialogMessage.setObjectName("DialogMessage")
        DialogMessage.resize(409, 142)
        '''
        self.buttonBox = QtWidgets.QDialogButtonBox(DialogMessage)
        self.buttonBox.setGeometry(QtCore.QRect(10, 100, 381, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        '''
        self.lblMsg = QtWidgets.QLabel(DialogMessage)
        self.lblMsg.setGeometry(QtCore.QRect(30, 20, 351, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(10)
        self.lblMsg.setFont(font)
        self.lblMsg.setText("")
        self.lblMsg.setAlignment(QtCore.Qt.AlignCenter)
        self.lblMsg.setWordWrap(True)
        self.lblMsg.setObjectName("lblMsg")

        self.retranslateUi(DialogMessage)
        # self.buttonBox.accepted.connect(DialogMessage.accept)
        # QtCore.QMetaObject.connectSlotsByName(DialogMessage)

    def retranslateUi(self, DialogMessage):
        _translate = QtCore.QCoreApplication.translate
        DialogMessage.setWindowTitle(_translate("DialogMessage", "Dialog"))


def show_dialog(txt: str, parent, title: str = None, error=False, modal=True):
    dialog = QtWidgets.QMessageBox(parent)
    dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
    dialog.setText(txt)
    dialog.setModal(modal)
    # dialog.setWindowModality(QtCore.Qt.WindowModal)
    if error:
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
    if not title:
        title = 'Dialog'
    dialog.setWindowTitle(title)
    dialog.show()


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gametype.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


class Ui_FrmGameType(GameWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def setupUi(self, FrmGameType):
        FrmGameType.setObjectName("FrmGameType")
        FrmGameType.resize(480, 315)
        self.lblTitle = QtWidgets.QLabel(FrmGameType)
        self.lblTitle.setGeometry(QtCore.QRect(80, 20, 301, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.lblTitle.setFont(font)
        self.lblTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTitle.setObjectName("lblTitle")
        self.btnDefault = QtWidgets.QPushButton(FrmGameType)
        self.btnDefault.setGeometry(QtCore.QRect(150, 120, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnDefault.setFont(font)
        self.btnDefault.setObjectName("btnDefault")
        self.btnCustom = QtWidgets.QPushButton(FrmGameType)
        self.btnCustom.setGeometry(QtCore.QRect(150, 160, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnCustom.setFont(font)
        self.btnCustom.setObjectName("btnCustom")
        self.btnLoad = QtWidgets.QPushButton(FrmGameType)
        self.btnLoad.setGeometry(QtCore.QRect(150, 200, 161, 41))
        self.btnLoad.setFont(font)
        self.btnLoad.setObjectName('btnLoad')
        self.btnBack = QtWidgets.QPushButton(FrmGameType)
        self.btnBack.setGeometry(QtCore.QRect(180, 250, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnBack.setFont(font)
        self.btnBack.setObjectName("btnBack")

        self.retranslateUi(FrmGameType)
        QtCore.QMetaObject.connectSlotsByName(FrmGameType)

    def retranslateUi(self, FrmGameType):
        _translate = QtCore.QCoreApplication.translate
        FrmGameType.setWindowTitle(_translate("FrmGameType", "Hnefatafl - Game Type"))
        self.lblTitle.setText(_translate("FrmGameType", "Game Type"))
        self.btnDefault.setText(_translate("FrmGameType", "Default Game"))
        self.btnCustom.setText(_translate("FrmGameType", "Custom Game"))
        self.btnLoad.setText(_translate('FrmGameType', 'Load Game'))
        self.btnBack.setText(_translate("FrmGameType", "Back"))
