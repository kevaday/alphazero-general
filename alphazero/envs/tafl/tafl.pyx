# cython: language_level=3

import pyximport; pyximport.install()

from alphazero.Game import GameState
from hnefatafl.engine import Board, Move, PieceType, variants
from typing import List, Tuple, Any

import numpy as np


def _get_board():
    return Board(
        GAME_VARIANT,
        max_repeats=MAX_REPEATS,
        _store_past_states=False,
        _max_past_states=min((MAX_REPEATS + 1) * NUM_PLAYERS, NUM_STACKED_OBSERVATIONS - 1)
    )


GAME_VARIANT = variants.hnefatafl
MAX_REPEATS = 0  # N-fold repetition loss
NUM_PLAYERS = 2
NUM_STACKED_OBSERVATIONS = 1
NUM_BASE_CHANNELS = 5
NUM_CHANNELS = NUM_BASE_CHANNELS * NUM_STACKED_OBSERVATIONS

b = _get_board()
ACTION_SIZE = b.width * b.height * (b.width + b.height - 2)
OBS_SIZE = (NUM_CHANNELS, b.width, b.height)
del b

DRAW_MOVE_COUNT = 800


def _board_from_numpy(np_board: np.ndarray) -> Board:
    return Board(custom_board='\n'.join([''.join([str(i) for i in row]) for row in np_board]))


def _board_to_numpy(board: Board) -> np.ndarray:
    return np.array([[int(tile) for tile in row] for row in board._board])


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


def _get_observation(board: Board, const_max_player: int, const_max_turns: int, past_obs: int = 1):
    obs = []

    def add_obs(b):
        game_board = _board_to_numpy(b)
        black = np.where(game_board == PieceType.black.value, 1., 0.)
        white = np.where((game_board == PieceType.white.value) | (game_board == PieceType.king.value), 1., 0.)
        king = np.where(game_board == PieceType.king.value, 1., 0.)
        turn_colour = np.full_like(
            game_board,
            2 - b.to_play().value / (const_max_player - 1) if const_max_player > 1 else 0
        )
        turn_number = np.full_like(
            game_board,
            b.num_turns / const_max_turns if const_max_turns else 0, dtype=np.float32
        )
        obs.extend([black, white, king, turn_colour, turn_number])

    def add_empty():
        obs.extend([[[0]*board.width]*board.height]*NUM_BASE_CHANNELS)

    if board._store_past_states:
        past = board._past_states.copy()
        past.insert(0, (board, None))
        for i in range(past_obs):
            if board.num_turns < i:
                add_empty()
            else:
                add_obs(past[i][0])
    else:
        add_obs(board)

    return np.array(obs, dtype=np.float32)


class TaflGame(GameState):
    def __init__(self, _board=None):
        super().__init__(_board or _get_board())

    def __eq__(self, other: 'TaflGame') -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self._board) + '\n'

    @staticmethod
    def _get_piece_type(player: int) -> PieceType:
        return PieceType(2 - player)

    @staticmethod
    def _get_player_int(player: PieceType) -> int:
        return (1, -1)[2 - player.value]

    def clone(self) -> 'GameState':
        g = TaflGame(self._board.copy(store_past_states=self._board._store_past_states))
        g._player = self._player
        g._turns = self.turns
        return g

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return OBS_SIZE

    def valid_moves(self):
        valids = [0] * self.action_size()
        legal_moves = self._board.all_valid_moves(self._board.to_play())

        for move in legal_moves:
            valids[get_action(self._board, move)] = 1

        return np.array(valids, dtype=np.intc)

    def play_action(self, action: int) -> None:
        move = get_move(self._board, action)
        self._board.move(move, _check_game_end=False, _check_valid=False)
        self._update_turn()

    def win_state(self) -> Tuple[bool, ...]:
        result = [False] * (NUM_PLAYERS + 1)

        # Check if maximum moves have been exceeded
        if self.turns >= DRAW_MOVE_COUNT:
            result[-1] = True
        else:
            winner: PieceType = self._board.get_winner()
            if winner:
                result[2 - winner.value] = True

        return tuple(result)

    def observation(self):
        return _get_observation(
            self._board,
            NUM_PLAYERS,
            DRAW_MOVE_COUNT,
            NUM_STACKED_OBSERVATIONS
        )

    def symmetries(self, pi: np.ndarray) -> List[Tuple[Any, int]]:
        action_size = self.action_size()
        assert (len(pi) == action_size)
        syms = [None] * 8

        for i in range(1, 5):
            for flip in (False, True):
                state = np.rot90(np.array(self._board._board), i)
                if flip:
                    state = np.fliplr(state)
                
                if self._board._store_past_states:
                    num_past_states = min(
                        NUM_STACKED_OBSERVATIONS - 1,
                        len(self._board._past_states)
                    )
                    past_states = [None] * num_past_states
                    for idx in range(num_past_states):
                        past = self._board._past_states[idx]
                        b = np.rot90(np.array(past[0]._board), i)
                        if flip:
                            b = np.fliplr(b)
                        past_states[idx] = (self._board.copy(store_past_states=False, state=b.tolist()), past[1])
                else:
                    past_states = None

                new_b = self._board.copy(
                    store_past_states=self._board._store_past_states,
                    state=state.tolist(),
                    past_states=past_states
                )
                if not past_states:
                    new_b._past_states = [s.copy(new_b) for s in new_b._past_states]

                new_pi = [0] * action_size
                for action, prob in enumerate(pi):
                    move = get_move(self._board, action)

                    x = move.tile.x
                    new_x = move.new_tile.x
                    y = move.tile.y
                    new_y = move.new_tile.y

                    for _ in range(i):
                        temp_x = x
                        temp_new_x = new_x
                        x = self._board.width - 1 - y
                        new_x = self._board.width - 1 - new_y
                        y = temp_x
                        new_y = temp_new_x
                    if flip:
                        x = self._board.width - 1 - x
                        new_x = self._board.width - 1 - new_x

                    move = Move(new_b, x, y, new_x, new_y)
                    new_action = get_action(new_b, move)
                    new_pi[new_action] = prob

                new_state = self.clone()
                new_state._board = new_b
                syms[(i - 1) * 2 + int(flip)] = (new_state, np.array(new_pi, dtype=np.float32))

        return syms

    def crude_value(self) -> int:
        _, result = self.win_state()
        white_pieces = len(list(filter(lambda p: p.is_white, self._board.pieces)))
        black_pieces = len(list(filter(lambda p: p.is_black, self._board.pieces)))
        return self.player * (1000 * result + black_pieces - white_pieces)


def display(state: TaflGame, action: int = None):
    print(f'Action: {action}, Move: {get_move(state._board, action)}')
    print(state)


def test_repeat(n):
    global GAME_VARIANT
    GAME_VARIANT = variants.hnefatafl
    g = TaflGame()
    # g.board[0][0].piece = Piece(PieceType(3), 0, 0, 0)
    board = _get_board()
    for _ in range(n):
        board.move(Move(board, 3, 0, 2, 0))
        board.move(Move(board, 5, 3, 5, 2))
        board.move(Move(board, 2, 0, 3, 0))
        board.move(Move(board, 5, 2, 5, 3))
        print(board.num_repeats(PieceType.black), board.num_repeats(PieceType.white))
    g._board = board
    print(g.win_state())
