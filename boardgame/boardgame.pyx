# cython: language_level=3

from boardgame.errors import InvalidMoveError
from typing import Set, List, Callable, Union, Tuple, Optional
from abc import ABC, abstractmethod

import uuid

_TWO_MOVES_BACK = 4


class Win(Exception):
    pass


class BlackWin(Win):
    pass


class WhiteWin(Win):
    pass


class BasePlayer(object):
    def __init__(self, is_turn, white=True, bot=False, raise_win=False):
        self.white = white
        self.bot = bot
        self.raise_win = raise_win
        self.won = False
        self.__is_turn = is_turn

    def __eq__(self, other):
        return other is not None and self.__dict__ == other.__dict__

    @property
    def is_turn(self) -> bool:
        return self.__is_turn

    @is_turn.setter
    def is_turn(self, turn) -> None:
        self.__is_turn = turn

    def win(self):
        self.won = True
        if self.raise_win:
            if self.white:
                raise WhiteWin
            else:
                raise BlackWin

    def reset(self):
        self.__init__(self.white, self.bot)


'''
class BasePieceType(object):
    pass
'''


class BasePiece(object):
    def __init__(self, is_white: bool, piece_type, piece_id: int, pos_x: int, pos_y: int):
        """
        Class for the piece object. Can be black or white, king or not king
        :param is_white: :type bool: is the piece white
        :param piece_type: :type boardgame.BasePieceType: Type of piece
        :param piece_id: :type int: unique integer to identify piece
        :param pos_x: :type int: x position of piece in the board
        :param pos_y: :type int: y position of piece in the board
        """
        self.type = piece_type
        self.x = pos_x
        self.y = pos_y
        self.__id = piece_id
        self.__white = is_white

    def __repr__(self) -> str:
        return str(self.type.value)

    @property
    def id(self) -> int:
        return self.__id

    @property
    def is_white(self) -> bool:
        return self.__white

    @property
    def is_black(self) -> bool:
        return not self.__white


'''
class BaseTileType(object):
    pass
'''


class BaseTile(object):
    def __init__(self, tile_type, pos_x: int, pos_y: int, piece: BasePiece = None):
        self.type = tile_type
        self.x = pos_x
        self.y = pos_y
        self.piece = piece

    def __repr__(self):
        return f'{self.__class__}{self.__dict__}'

    def __str__(self):
        return self.to_string(add_values=False)

    def __int__(self):
        return int(self.to_string(add_values=False))

    def __eq__(self, other):
        return other is not None and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(f'{self.type.value} {self.x} {self.y} {repr(self.piece)}')

    def to_string(self, add_values=True, piece_priority=True):
        string = self.type.value
        if self.piece is not None and piece_priority:
            if add_values:
                string += self.piece.type.value
            else:
                string = self.piece.type.value

        return str(string)

    def update(self):
        if self.piece is not None:
            self.piece.x = self.x
            self.piece.y = self.y


class BaseBoard(ABC):
    def __init__(self, board: Union['BaseBoard', str, list] = None, load_file: str = None, save_file: str = None,
                 max_repeats: int = None, store_initial_state=False, custom=False, _store_past_states=True, _max_past_states: int = None):
        self.__kwargs = locals()
        del self.__kwargs['self']
        self.save_file = save_file
        self.is_custom = custom
        self.pieces = []
        self.killed_pieces = []
        self.max_repeats = max_repeats
        self._store_past_states = _store_past_states

        if isinstance(board, list):
            self._board = board
            self.height = len(board)
            self.width = len(board[0])
        else:
            self.width = self.height = None
            self._board = None
        if board and not self._board:
            if isinstance(board, BaseBoard):
                board = board.to_string()
            self.load(board)
        elif isinstance(board, str):
            self.load_file(load_file)

        if _store_past_states:
            self._past_states = []
            self._max_past_states = _max_past_states
            if _max_past_states:
                self._turn_count = 0
            if store_initial_state: self._update_state(move=None)
        else:
            self._turn_count = 0
            self.__last_move = {}
            self.__repeats = {}

    @abstractmethod
    def load(self, data: str) -> List[str]:
        rows = data.split('\n')
        self.height = len(rows)
        self.width = len(rows[0])
        self._board = []
        self.pieces = []
        return rows
        # ...

    def load_file(self, path: str) -> None:
        with open(path, 'r') as f:
            data = f.read()
        self.load(data)

    def save(self):
        with open(self.save_file, 'w') as f:
            f.write(self.to_string(add_values=True))

    def to_string(self, add_values=True, add_spaces=False, piece_priority=True):
        return '\n'.join(
            [(' ' if add_spaces else '').join(
                [tile.to_string(add_values, piece_priority=piece_priority) for tile in row]
            ) for row in self]
        )

    def __len__(self):
        return len(self._board)

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    def __delitem__(self, key):
        del self._board[key]

    def __iter__(self):
        return iter(self._board)

    def __contains__(self, item):
        return item in self._board

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_string(add_spaces=True, add_values=False)

    def __int__(self):
        return [[int(tile) for tile in row] for row in self]

    def __eq__(self, other: 'BaseBoard') -> bool:
        return (
            self.width == other.width
            and self.height == other.height
            and self.num_turns == other.num_turns
            and self.get_winner() == other.get_winner()
            and [tile == other_tile for row, other_row in zip(self, other) for tile, other_tile in zip(row, other_row)]
        )

    def __copy__(
            self,
            store_past_states=True,
            state: Union[str, list] = None,
            past_states: Tuple['BaseBoard', 'Move'] = None,
            _copy_state=True
    ):
        store = self._store_past_states and store_past_states
        kwargs = self.__kwargs.copy()
        kwargs['board'] = self.to_string() if not state else state.copy() if _copy_state else state
        kwargs['_store_past_states'] = store
        new_board = self.__class__(**kwargs)

        if store:
            new_board._past_states = past_states or [
                (state[0], state[1].copy(new_board) if state[1] else None) for state in self._past_states
            ]
            if self._max_past_states: new_board._turn_count = self._turn_count
        else:
            new_board._turn_count = len(self._past_states) \
                if self._store_past_states and not self._max_past_states else self._turn_count

        return new_board

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)

    @property
    def num_turns(self):
        return len(self._past_states) if self._store_past_states and not self._max_past_states else self._turn_count

    def _repeats_from(self, start_index: int = 0, piece_type=None, stop_at_max=False) -> int:
        if self._store_past_states:
            count = 0
            last_move = self.get_past_move(start_index)
            for i in range(start_index + _TWO_MOVES_BACK, self.num_turns, _TWO_MOVES_BACK):
                move = self.get_past_move(i)
                if move != last_move:
                    break
                count += 1
                if stop_at_max and self.max_repeats and count >= self.max_repeats: break

            return count

        return max(self.__repeats[piece_type].values()) if self.__repeats.get(piece_type) else 0

    @staticmethod
    def get_piece(tile_or_piece):
        p = tile_or_piece
        if isinstance(tile_or_piece, BaseTile):
            p = tile_or_piece.piece
        return p

    def get_tile(self, piece):
        if piece: return self[piece.y][piece.x]

    def undo(self) -> bool:
        if not self._store_past_states or not len(self._past_states): return False
        self.__init__(self._past_states.pop(0)[0])
        return True

    def in_bounds(self, x, y):
        """
        Check if coordinates are in bounds
        :param x:
        :param y:
        :return: :type bool: True if in bounds, False if not
        """
        return self.width - 1 >= x >= 0 and self.height - 1 >= y >= 0

    @abstractmethod
    def valid_moves(self, tile_or_piece: Union[BaseTile, BasePiece]) -> Set['Move']:
        """
        Get the valid moves of a piece
        :param tile_or_piece: piece or tile to get valid moves of
        :return: set of tiles that the piece can move to
        """
        pass

    @abstractmethod
    def all_valid_moves(self, piece_type) -> Set['Move']:
        """
        Get the valid moves of all pieces of specified type on the board
        :param piece_type: :type boardgame.BasePieceType: type of piece to get valid moves of
        :return: list of all the possible moves of all pieces of specified type
        """
        pass

    def relative_tile(self, tile_or_piece: Union[BaseTile, BasePiece], x_amount: int, y_amount: int) -> BaseTile:
        """
        Get the tile relative to a given tile
        :param tile_or_piece: piece or tile to get relative tile from
        :param x_amount: :type int: relative amount in the x direction
        :param y_amount: :type int: relative amount in the y direction
        :return: relative tile
        """
        new_x = tile_or_piece.x + x_amount
        new_y = tile_or_piece.y + y_amount

        if self.in_bounds(new_x, new_y):
            return self[new_y][new_x]

    def right_tile(self, tile_or_piece):
        return self.relative_tile(tile_or_piece, x_amount=1, y_amount=0)

    def left_tile(self, tile_or_piece):
        return self.relative_tile(tile_or_piece, x_amount=-1, y_amount=0)

    def up_tile(self, tile_or_piece):
        return self.relative_tile(tile_or_piece, x_amount=0, y_amount=-1)

    def down_tile(self, tile_or_piece):
        return self.relative_tile(tile_or_piece, x_amount=0, y_amount=1)

    def _get_surrounding_funcs(self) -> List[Callable]:
        return [self.right_tile, self.left_tile, self.up_tile, self.down_tile]

    def get_surrounding_tiles(self, tile_or_piece: Union[BaseTile, BasePiece]) -> List[BaseTile]:
        return [func(tile_or_piece) for func in self._get_surrounding_funcs()]

    def _kill(self, tile_or_piece: Union[BaseTile, BasePiece]) -> None:
        tile = self.get_tile(tile_or_piece)
        if tile.piece:
            self.killed_pieces.append(tile.piece)
            self.pieces.remove(tile.piece)
            self[tile.y][tile.x].piece = None

    @abstractmethod
    def _check_kill(self, piece: BasePiece) -> None:
        """
        Check whether a piece should be killed or not after a move. Kills piece if yes.
        :param piece: piece that just moved to check if it kills something
        """
        pass

    @abstractmethod
    def move(self, move: 'Move') -> None:
        """
        Move a piece from one tile to another, updates board state. Call _update_state before if valid move.
        :param move: Move object defining the move
        """
        pass

    @abstractmethod
    def get_team_colour(self, piece_type):
        """Given piece_type, should always return the same constant piece type for the team that piece_type is on."""
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Boolean indicating whether or not the game has ended."""
        pass

    @abstractmethod
    def get_winner(self):
        """Returns the player indicated by a piece type that has won the game. None if there is no winner yet."""
        pass

    def _repeat_exceeded(self, piece_type=None):
        """To be called after a move is done to check if a player has exceed their maximum allowed repeats."""
        return self.max_repeats and self._repeats_from(piece_type=piece_type, stop_at_max=True) >= self.max_repeats

    def _update_state(self, move: 'Move'):
        if self._store_past_states:
            self._past_states.insert(0, (self.copy(store_past_states=False), move))
            if self._max_past_states:
                self._past_states = self._past_states[:self._max_past_states]
                self._turn_count += 1
        else:
            self._turn_count += 1
            if not self.max_repeats: return

            piece_type = self.get_team_colour(move.get_piece().type)
            remove_repeat = False
            if self.__last_move.get(piece_type):
                self.__last_move[piece_type].insert(0, move)
                while len(self.__last_move[piece_type]) > 3:
                    del self.__last_move[piece_type][-1]
                    remove_repeat = True
            else:
                self.__last_move[piece_type] = [move]
                return
            
            repeat = int(
                len(self.__last_move[piece_type]) > 2
                and move == self.__last_move[piece_type][-1]
            )
            if self.__repeats.get(piece_type):
                if self.__repeats[piece_type].get(move):
                    if repeat:
                        self.__repeats[piece_type][move] += repeat
                    else:
                        if remove_repeat:
                            del self.__repeats[piece_type][move]
                        else:
                            self.__repeats[piece_type][move] = repeat

                else: self.__repeats[piece_type].update({move: repeat})
            else: self.__repeats[piece_type] = {move: repeat}

    def get_past_move(self, index) -> Optional['Move']:
        """Get the a past move that occurred on the board from the given index. Only works if past states are stored."""
        if not self._store_past_states or index >= len(self._past_states): return
        return self._past_states[index][1]

    def get_last_move(self) -> Optional['Move']:
        return self.get_past_move(0)

    def reset(self):
        self.__init__(**self.__kwargs)


class Move(object):
    def __init__(self, board: BaseBoard, *args, _check_in_bounds=True):
        self.__args = args

        while (
            len(args) == 1
            and isinstance(args, (list, tuple))
        ):
            args = args[0]

        self.board = board
        self.tile = None
        self.new_tile = None

        if len(args) == 2:
            self.tile = self.board.get_tile(self.__item_from_args(args[0]))
            self.new_tile = self.board.get_tile(self.__item_from_args(args[1]))

        elif len(args) == 4:
            xy = args[:2]
            new_xy = args[2:]
            if _check_in_bounds:
                if any(map(lambda x: not self.board.in_bounds(*x), (xy, new_xy))):
                    raise InvalidMoveError('The coordinates are out of bounds.')

            self.tile = self.board.get_tile(self.__item_from_args(*xy))
            self.new_tile = self.board.get_tile(self.__item_from_args(*new_xy))

        if self.tile is None or self.new_tile is None:
            print('empty move, args: ', args)

    def __repr__(self):
        return f'({self.tile.x},{self.tile.y})->({self.new_tile.x},{self.new_tile.y})'

    def __str__(self):
        return f'{self.__class__.__name__}(tile={repr(self.tile)}, new_tile={repr(self.new_tile)})'

    def __eq__(self, other):
        return other is not None and self.tile == other.tile and self.new_tile == other.new_tile

    def __hash__(self):
        return hash(f'{hash(self.tile)} {hash(self.new_tile)}')

    def __reversed__(self):
        move = self.copy()
        move.reverse()
        return move

    def __item_from_args(self, *args) -> Union[BaseTile, BasePiece]:
        tile_or_piece = None

        if len(args) == 1:
            if isinstance(args[0], BaseTile) or isinstance(args[0], BasePiece):
                tile_or_piece = args[0]
        elif len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int):
                tile_or_piece = self.board[args[1]][args[0]]

        return tile_or_piece
    
    def get_piece(self):
        if self.tile.piece: return self.tile.piece
        if self.new_tile.piece: return self.new_tile.piece

    def copy(self, new_board: BaseBoard = None):
        new_board = new_board or self.board
        return self.__class__(new_board, self.__args)

    @property
    def is_vertical(self):
        return (self.tile.x - self.new_tile.x) == 0

    @property
    def is_horizontal(self):
        return not self.is_vertical

    @property
    def is_diagonal(self):
        return abs(self.tile.x - self.new_tile.x) == abs(self.tile.y - self.new_tile.y) and self.tile != self.new_tile

    def reverse(self):
        self.tile, self.new_tile = self.new_tile, self.tile

    def serialize(self) -> bytes:
        def byte(i: int) -> bytes: return bytes([i])

        def get_cords(t: BaseTile) -> bytes: return byte(t.x) + byte(t.y)

        return get_cords(self.tile) + get_cords(self.new_tile)

    @classmethod
    def from_serial(cls, board: BaseBoard, serial: bytes) -> 'Move':
        return cls(board, list(serial))


class BaseGame(object):
    def __init__(self, board: BaseBoard, white_player: BasePlayer, black_player: BasePlayer, raise_win=False, id=None):
        self.id = id
        self.board = board
        self.white = white_player
        self.white.raise_win = raise_win
        self.black = black_player
        self.black.raise_win = raise_win
        self.started = False
        self.game_over = False
        if not self.id: self.id = uuid.uuid4().hex

    def __repr__(self):
        return f'{self.__class__}(id={self.id}, started={self.started}, game_over={self.game_over})'

    def start(self) -> None:
        self.started = True

    @property
    def is_custom(self) -> bool:
        return self.board.is_custom

    def is_turn(self, tile_or_piece):
        piece = self.board.get_piece(tile_or_piece)
        if not piece:
            return False

        return (piece.is_white and self.white.is_turn) or (piece.is_black and self.black.is_turn)

    def _update_turn(self):
        self.white.is_turn = not self.white.is_turn
        self.black.is_turn = not self.black.is_turn

    def _white_won(self):
        self.game_over = True
        self.white.win()

    def _black_won(self):
        self.game_over = True
        self.black.win()

    def _check_moves(self, piece_type):
        return len(self.board.all_valid_moves(piece_type)) == 0

    def move(self, *args) -> None:
        """Method for handling turns for moving on the board."""
        pass

    def undo(self):
        if self.board.undo():
            self._update_turn()
            if self.game_over:
                self.game_over = False
                if self.white.won:
                    self.white.won = False
                elif self.black.won:
                    self.black.won = False

    def reset(self):
        self.black.reset()
        self.white.reset()
        self.board.reset()


class BaseBot(object):
    def __init__(self, player: BasePlayer):
        self.player = player

    def get_move(self, board: BaseBoard) -> Optional[Move]:
        """Method for getting a move from the bot's logic based on the given :param board:"""
        pass
