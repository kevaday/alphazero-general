# cython: language_level=3
import sys

from boardgame.errors import TurnError

sys.path.append('..')
sys.path.append('../..')

from Game import Game as AlphaGame
from hnefatafl.engine.board import Board, Move, PieceType
from hnefatafl.engine.game import Game, bool_to_colour

import numpy as np
import string


DIGS = string.digits + string.ascii_letters
DRAW_MOVE_COUNT = 256
OBS_CHANNELS = 4

BOARD = """50022222005
00000200000
00000000000
20000100002
20001110002
22011711022
20001110002
20000100002
00000000000
00000200000
50022222005"""


def _board_from_numpy(np_board: np.ndarray) -> Board:
    return Board(custom_board='\n'.join([''.join([str(i) for i in row]) for row in np_board.tolist()]))


def _board_to_numpy(board: Board) -> np.ndarray:
    return np.array([[int(tile) for tile in row] for row in board])


def get_move(board: Board, action: int) -> Move:
    size = (board.width + board.height - 2)
    move_type = action % size
    a = action // size
    start_x = a % board.width
    start_y = a // board.width

    if move_type < board.height - 1:
        new_x = start_x
        new_y = move_type
        if move_type >= start_y: new_y += 1
    else:
        new_x = move_type - board.height + 1
        if new_x >= start_x: new_x += 1
        new_y = start_y

    return Move(board, int(start_x), int(start_y), int(new_x), int(new_y))


def get_action(board: Board, move: Move) -> int:
    new_x = move.new_tile.x
    new_y = move.new_tile.y

    if move.is_vertical:
        move_type = new_y if new_y < move.tile.y else new_y - 1
    else:
        move_type = board.height + new_x - 1
        if new_x >= move.tile.x: move_type -= 1

    return (board.width + board.height - 2) * (move.tile.x + move.tile.y * board.width) + move_type


def _get_observation(board: Board, player_turn: int):
    game_board = _board_to_numpy(board)
    black = np.where(game_board == PieceType.black.value, 1., 0.)
    white = np.where((game_board == PieceType.white.value) | (game_board == PieceType.king.value), 1., 0.)
    king = np.where(game_board == PieceType.king.value, 1., 0.)
    turn = np.full_like(game_board, player_turn)

    return black, white, king, turn


class CustomBoard(Board):
    def astype(self, t):
        return np.array(_get_observation(self, 2 - self.to_play().value), dtype=t)


class HnefataflGame(AlphaGame):
    def __init__(self):
        self.board = CustomBoard(custom_board=BOARD, _store_past_states=True)

    @staticmethod
    def __get_piece_type(player: int) -> PieceType:
        return PieceType.black if player == 1 else PieceType.white

    @staticmethod
    def __get_player_int(player: PieceType) -> int:
        return 1 if player == PieceType.black else -1

    def getInitBoard(self) -> CustomBoard:
        return self.board.copy()

    def getBoardSize(self):
        return self.board.width, self.board.height

    def getActionSize(self) -> int:
        return self.board.width * self.board.height * (self.board.width + self.board.height - 2)

    def getObservationSize(self) -> tuple:
        # channels x width x height
        return OBS_CHANNELS, self.board.width, self.board.height

    def getNextState(self, board: CustomBoard, player: int, action: int):
        b = board.copy()
        move = get_move(b, action)
        
        """
        if b.current_player != self.__get_player_int(b.to_play()):
            vms = self.getValidMoves(b, player)
            print(
                f"move: {str(move)}, action: {action}, turn: {bool_to_colour(b.to_play() == PieceType.white)}\n"
                f"action in valids: {action in vms}, len of valids: {len(list(filter(lambda x: x == 1, vms)))}\n"
                f"move in valids: {move in b.all_valid_moves(b.to_play())}\n"
                f"current player: {b.current_player}, num_turns: {b.num_turns}\n"
                f"past moves: {[str(state[1]) for state in b._past_states]}"
            )
            print(b.to_string(add_spaces=True))
            print()
            exit()
        """

        b.move(move)

        return b, -player

    def getValidMoves(self, board: CustomBoard, player: int):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valids = [0] * self.getActionSize()
        legal_moves = board.all_valid_moves(board.to_play())

        for move in legal_moves:
            valids[get_action(board, move)] = 1

        return np.array(valids, dtype=np.float32)

    def getGameEnded(self, board: CustomBoard, player: int):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if board.num_turns >= DRAW_MOVE_COUNT: return 1e-4

        winner = board.get_winner()
        if not winner: return 0
        
        player *= self.current_player
        
        winner = self.__get_player_int(winner)
        reward = int(winner == player)
        reward -= int(winner == -player)
        
        return reward

    def getCanonicalForm(self, board: CustomBoard, player: int):
        b = board.copy()
        b.current_player = player
        return b

    def getSymmetries(self, board: CustomBoard, pi: list):
        return [(board, pi)]

    def stringRepresentation(self, board: CustomBoard):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.to_string()
