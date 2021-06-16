# cython: language_level=3

import pyximport; pyximport.install()

from alphazero.Game import Game
from hnefatafl.engine import Board, Move, PieceType
from typing import List, Tuple

import numpy as np


NUM_PLAYERS = 2
DRAW_MOVE_COUNT = 200
NUM_STACKED_OBSERVATIONS = 8
NUM_BASE_CHANNELS = 5
NUM_CHANNELS = NUM_BASE_CHANNELS * NUM_STACKED_OBSERVATIONS
MAX_REPEATS = 3  # N-fold repetition loss
# Used for checking pairs of moves from same player
# Changing this breaks repeat check
_ONE_MOVE = 2
_TWO_MOVES = _ONE_MOVE * 2


def _board_from_numpy(np_board: np.ndarray) -> Board:
    return Board(custom_board='\n'.join([''.join([str(i) for i in row]) for row in np_board]))


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

    return Move(board, int(start_x), int(start_y), int(new_x), int(new_y), _check_in_bounds=False)


def get_action(board: Board, move: Move) -> int:
    new_x = move.new_tile.x
    new_y = move.new_tile.y

    if move.is_vertical:
        move_type = new_y if new_y < move.tile.y else new_y - 1
    else:
        move_type = board.height + new_x - 1
        if new_x >= move.tile.x: move_type -= 1

    return (board.width + board.height - 2) * (move.tile.x + move.tile.y * board.width) + move_type


def _get_observation(board: Board, player_turn: int, const_max_player: int, const_max_turns: int, past_obs: int = 1):
    obs = []

    def add_obs(b, turn_num):
        game_board = _board_to_numpy(b)
        black = np.where(game_board == PieceType.black.value, 1., 0.)
        white = np.where((game_board == PieceType.white.value) | (game_board == PieceType.king.value), 1., 0.)
        king = np.where(game_board == PieceType.king.value, 1., 0.)
        turn_colour = np.full_like(
            game_board,
            player_turn / (const_max_player - 1) if const_max_player > 1 else 0
        )
        turn_number = np.full_like(game_board, turn_num / const_max_turns if const_max_turns else 0, dtype=np.float32)
        obs.extend([black, white, king, turn_colour, turn_number])

    def add_empty():
        obs.extend([[[0]*board.width]*board.height]*NUM_BASE_CHANNELS)

    past = board._past_states.copy()
    past.append((board, None))
    past_len = len(past)
    for i in range(past_obs):
        if past_len < i + 1:
            add_empty()
        else:
            add_obs(past[i][0], past_len-i-1)

    return np.array(obs, dtype=np.float32)


class CustomBoard(Board):
    def __init__(self, *args, max_moves=DRAW_MOVE_COUNT, num_stacked_obs=NUM_STACKED_OBSERVATIONS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_moves = max_moves if max_moves else 0
        self.num_stacked_obs = num_stacked_obs
        self.current_player = 0
        self.winner = None
    
    def copy(self, *args, **kwargs):
        b = super().copy(*args, **kwargs)
        b.max_moves = self.max_moves
        b.num_stacked_obs = self.num_stacked_obs
        b.current_player = self.current_player
        return b

    def astype(self, t):
        return _get_observation(
            self,
            2 - self.to_play().value,
            NUM_PLAYERS,
            self.max_moves,
            self.num_stacked_obs
        ).astype(t)


class TaflGame(Game):
    def __init__(self, game_variant, max_moves: int = DRAW_MOVE_COUNT, num_stacked_obs: int = NUM_STACKED_OBSERVATIONS):
        super().__init__()
        self.board = CustomBoard(
            max_moves=max_moves,
            num_stacked_obs=num_stacked_obs,
            board=game_variant,
            _store_past_states=True,
            _max_past_states=max(num_stacked_obs, _TWO_MOVES * MAX_REPEATS + _ONE_MOVE)
        )
        self.board_size = (self.board.width, self.board.height)
        self.action_size = self.board.width * self.board.height * (self.board.width + self.board.height - 2)
        self.players = list(range(NUM_PLAYERS))
        self.obs_size = (NUM_CHANNELS, *self.board_size)

    @staticmethod
    def _get_piece_type(player: int) -> PieceType:
        return PieceType(2 - player)

    @staticmethod
    def _get_player_int(player: PieceType) -> int:
        return 2 - player.value

    def getInitBoard(self) -> CustomBoard:
        return self.board.copy()

    def getBoardSize(self):
        return self.board_size

    def getActionSize(self) -> int:
        return self.action_size

    def getPlayers(self) -> List[int]:
        return self.players

    def getPlayerToPlay(self, board: CustomBoard) -> int:
        return 2 - board.to_play().value

    def getObservationSize(self) -> tuple:
        # channels x width x height
        return self.obs_size

    def getNextState(self, board: CustomBoard, player: int, action: int, copy=True):
        b = board.copy() if copy else board
        move = get_move(b, action)
        try:
            b.move(move, _check_game_end=False, _check_valid=False)
        except Exception as e:
            print()
            print(e)
            print(b)
            print(str(move), action)
            print(list(reversed(board._past_states)))
            print(player, board.num_turns, board.get_winner())
            print(bool(self.getValidMoves(board, player)[action]))
            print(move in b.all_valid_moves(b.to_play()))
            exit()

        return b, self.getNextPlayer(player)

    def getValidMoves(self, board: CustomBoard, player: int) -> np.ndarray:
        valids = [0] * self.getActionSize()
        legal_moves = board.all_valid_moves(
            self._get_piece_type(self.getNextPlayer(board.current_player, player))
        )

        for move in legal_moves:
            valids[get_action(board, move)] = 1

        return np.array(valids, dtype=np.float32)

    def getGameEnded(self, board: CustomBoard, player: int):
        # Check if maximum moves have been exceeded
        if self.board.max_moves and board.num_turns >= self.board.max_moves:
            return -1e-4

        winner = board.get_winner()
        if not winner: return 0

        player = self.getNextPlayer(board.current_player, player)

        winner = self._get_player_int(winner)
        reward = int(winner == player)
        reward -= int(winner == self.getNextPlayer(player))
        
        return reward

    def getCanonicalForm(self, board: CustomBoard, player: int, copy=True):
        b = board.copy() if copy else board
        b.current_player = self.getNextPlayer(b.current_player, player)
        return b

    def getSymmetries(self, board: CustomBoard, pi: np.ndarray) -> List[Tuple[CustomBoard, np.ndarray]]:
        action_size = self.getActionSize()
        assert (len(pi) == action_size)
        syms = [None] * 8

        for i in range(1, 5):
            for flip in (False, True):
                state = np.rot90(np.array(board._board), i)
                if flip:
                    state = np.fliplr(state)

                num_past_states = min(self.board.num_stacked_obs, len(board._past_states))
                past_states = [None] * num_past_states
                for idx in range(num_past_states):
                    past = board._past_states[idx]
                    b = np.rot90(np.array(past[0]._board), i)
                    if flip:
                        b = np.fliplr(b)
                    past_states[idx] = (board.copy(store_past_states=False, state=b.tolist()), past[1])

                new_b = board.copy(store_past_states=True, state=state.tolist(), past_states=past_states)

                new_pi = [0]*action_size
                for action, prob in enumerate(pi):
                    move = get_move(board, action)

                    x = move.tile.x
                    new_x = move.new_tile.x
                    y = move.tile.y
                    new_y = move.new_tile.y

                    for _ in range(i):
                        temp_x = x
                        temp_new_x = new_x
                        x = self.board.width - 1 - y
                        new_x = self.board.width - 1 - new_y
                        y = temp_x
                        new_y = temp_new_x
                    if flip:
                        x = self.board.width - 1 - x
                        new_x = self.board.width - 1 - new_x

                    move = Move(new_b, x, y, new_x, new_y)
                    new_action = get_action(new_b, move)
                    new_pi[new_action] = prob

                syms[(i-1)*2 + int(flip)] = (new_b, np.array(new_pi, dtype=np.float32))

        return syms

    def stringRepresentation(self, board: CustomBoard):
        return board.to_string() + str(board.current_player) + str(board.num_turns) + \
               str(board.num_repeats(PieceType.black)) + str(board.num_repeats(PieceType.white))

    def getScore(self, board: CustomBoard, player: int) -> int:
        result = self.getGameEnded(board, player)
        white_pieces = len(list(filter(lambda p: p.is_white, board.pieces)))
        black_pieces = len(list(filter(lambda p: p.is_black, board.pieces)))
        return [1, -1][player] * (1000 * result + black_pieces - white_pieces)


if __name__ == '__main__':
    from hnefatafl.engine import *

    g = TaflGame(variants.brandubh)
    # g.board[0][0].piece = Piece(PieceType(3), 0, 0, 0)
    board = g.getInitBoard()
    for _ in range(6):
        board.move(Move(board, 3, 0, 2, 0))
        board.move(Move(board, 3, 2, 2, 2))
        board.move(Move(board, 2, 0, 3, 0))
        board.move(Move(board, 2, 2, 3, 2))
    print(g.getGameEnded(board, 0), g.getGameEnded(board, 1))
