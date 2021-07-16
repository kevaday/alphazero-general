# cython: language_level=3

from hnefatafl.engine import variants
from boardgame import errors, BaseTile, BaseBoard, BasePiece, Move

from typing import Set, List, Union
from functools import partial
from enum import IntEnum


class PieceType(IntEnum):
    white = 1
    black = 2
    king = 3


class Piece(BasePiece):
    def __init__(self, piece_type, piece_id, pos_x, pos_y):
        """
        Class for the piece object. Can be black or white, king or not king
        :param piece_type: :type hnefetafl.piece.PieceType: Type of piece, white, black, or king
        :param piece_id: :type int: unique integer to identify piece
        :param pos_x: :type int: x position of piece in the board
        :param pos_y: :type int: y position of piece in the board
        """
        self.__white = piece_type == PieceType.white or piece_type == PieceType.king
        self._king = piece_type == PieceType.king
        super().__init__(self.__white, piece_type, piece_id, pos_x, pos_y)

    def copy(self) -> 'Piece':
        p = super().copy()
        p._king = self._king
        return p

    @property
    def is_king(self):
        return self._king


class TileType(IntEnum):
    normal = 0
    special = 4
    exit = 5


class Tile(BaseTile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_exit = False
        if self.type == TileType.exit:
            self.is_exit = True
            self.type = TileType.special

    def __eq__(self, other):
        return super().__eq__(other) and self.is_exit == other.is_exit

    def __hash__(self):
        return super().__hash__() + self.is_exit

    def __str__(self):
        return self.to_string(add_values=False)

    def copy(self) -> 'Tile':
        tile = super().copy()
        tile.is_exit = self.is_exit
        return tile

    def to_string(self, add_values=True, piece_priority=True):
        string = self.type.value if not self.is_exit else TileType.exit.value
        if self.piece is not None and piece_priority:
            if add_values:
                string += self.piece.type.value
            else:
                string = self.piece.type.value

        return str(string)

    @property
    def is_special(self):
        return self.type == TileType.special


class Board(BaseBoard):
    def __init__(self, board: Union['Board', str, list] = variants.hnefatafl, first_repeat_winner=True, *args,
                 num_start_white=None, num_start_black=None, **kwargs):
        self.first_repeat_winner = first_repeat_winner
        self.king = None
        self.king_captured = False
        self.king_escaped = False
        self.__moved_piece = None
        self.__recurse_exit = False
        self.__recurse_checked = []
        self.__recurse_kill = False
        self.__first_repeat = None

        super().__init__(board=board, *args, **kwargs)
        self.num_start_black = num_start_black or self.num_black
        self.num_start_white = num_start_white or self.num_white
    
    def copy(self, *args, **kwargs):
        board = super().copy(*args, **kwargs)
        board.king = board.get_king()
        board.king_escaped = self.king_escaped
        board.king_captured = self.king_captured
        board.num_start_black = self.num_start_black
        board.num_start_white = self.num_start_white
        board.__first_repeat = self.__first_repeat
        return board

    def load(self, data: str) -> None:
        rows = super().load(data)
        piece_id = 0

        for y, line in enumerate(rows):
            row = []
            for x, n in enumerate(line):
                try:
                    n = int(n)
                except ValueError:
                    raise errors.LoadError(
                        f'Failed to load board data: invalid input character `{n}` at x={x}, y={y}.\n'
                        f'The following data was received (for debug):\n{data}'
                    )

                tile_type = None
                piece = None

                if n == TileType.normal.value or n == TileType.special.value or n == TileType.exit.value:
                    tile_type = TileType(n)
                elif n == PieceType.black.value or n == PieceType.white.value or n == PieceType.king.value:
                    piece = PieceType(n)
                    tile_type = TileType.normal
                elif n == TileType.special.value + PieceType.king.value:
                    tile_type = TileType.special
                    piece = PieceType.king
                elif n == TileType.exit.value + PieceType.king.value:
                    tile_type = TileType.exit
                    piece = PieceType.king
                    self.king_escaped = True
                else:
                    raise errors.LoadError(
                        f'Failed to load board data: invalid tile type `{n}` at x={x}, y={y}.\n'
                        f'The following data was received (for debug):\n{data}'
                    )

                if piece is not None:
                    piece = Piece(piece, piece_id, x, y)
                    if piece.is_king:
                        self.king = piece
                    self.pieces.append(piece)
                    piece_id += 1

                row.append(Tile(tile_type, x, y, piece=piece))
            self._board.append(row)

    @property
    def num_white(self):
        return len([piece for piece in self.pieces if piece.is_white])

    @property
    def num_black(self):
        return len([piece for piece in self.pieces if piece.is_black])

    def num_repeats(self, piece_type: PieceType) -> int:
        return self._repeats_from(int(piece_type != self.to_play()))

    def get_team_colour(self, piece_type: PieceType) -> PieceType:
        return PieceType.white if piece_type == PieceType.king else piece_type

    def get_king(self):
        for piece in self.pieces:
            if piece.is_king:
                return piece

    def get_winner(self) -> PieceType:
        self._update_game_over()
        if (
            self.king_captured
            or ((self.__first_repeat == PieceType.black) if self.first_repeat_winner else (self.__first_repeat == PieceType.white))
            or not self.has_valid_moves(PieceType.white)
        ):
            return PieceType.black

        if (
            self.king_escaped
            or ((self.__first_repeat == PieceType.white) if self.first_repeat_winner else (self.__first_repeat == PieceType.black))
            or not self.has_valid_moves(PieceType.black)
        ):
            return PieceType.white

    def is_game_over(self) -> bool:
        return self.get_winner() is not None

    def to_play(self) -> PieceType:
        return PieceType(2 - self.num_turns % 2)

    def valid_moves(self, tile_or_piece: Union[Tile, Piece], ret_moves=True) -> Union[List[Move], List[Tile]]:
        """
        Get the valid moves of a piece
        :param tile_or_piece: :type hnefatafl.piece.Piece or hnefatafl.board.Tile or int,int: piece or tile or
        coordinates to get valid moves of
        :param ret_moves: whether to return the move objects or the tiles instead
        :return: :type set(hnefatafl.board.Tile): set of tiles that the piece can move to
        """
        moves = []
        from_tile = self.get_tile(tile_or_piece)
        piece = self.get_piece(tile_or_piece)
        assert piece is not None

        def do_check(tile: Tile) -> bool:
            def add_move(t: Tile):
                if ret_moves: moves.append(Move(self, from_tile, t, _check_in_bounds=False))
                else: moves.append(t)

            allowed = tile.piece is None
            if piece.is_king:
                allowed = allowed or tile.is_exit
                if allowed:
                    if tile.is_special and tile.is_exit: add_move(tile)
                    elif not tile.is_special: add_move(tile)
            else:
                if allowed and not tile.is_special: add_move(tile)
            return allowed

        for x in range(from_tile.x + 1, self.width):
            if not do_check(self[from_tile.y][x]):
                break

        for x in range(from_tile.x - 1, -1, -1):
            if not do_check(self[from_tile.y][x]):
                break

        for y in range(from_tile.y + 1, self.height):
            if not do_check(self[y][from_tile.x]):
                break

        for y in range(from_tile.y - 1, -1, -1):
            if not do_check(self[y][from_tile.x]):
                break

        return moves

    def all_valid_moves(self, piece_type: PieceType = None, include_king=True) -> Set[Move]:
        """
        Get the valid moves of all pieces of specified type on the board
        :param piece_type: :type PieceType: type of piece to get valid moves of
        :param include_king: whether or not to include the king when checking white pieces
        :return: :type set(hnefatafl.board.Move): list of all the possible moves of all pieces of specified type
        """
        moves = set()
        for piece in self.pieces:
            if not piece_type or piece.type == piece_type:
                moves.update(self.valid_moves(piece))

        if piece_type == PieceType.white and include_king:
            moves.update(self.valid_moves(self.king))

        return moves

    def has_valid_moves(self, piece_type: PieceType) -> bool:
        for piece in self.pieces:
            if piece.type == piece_type and self.valid_moves(piece):
                return True
        return False

    @staticmethod
    def __blocked(t: Tile, king=False) -> bool:
        if not t:
            return True
        elif t.is_special:
            return True
        else:
            if king:
                return t.piece.type == PieceType.black if t.piece else False
            else:
                return t.piece is not None

    def check_king_captured(self) -> bool:
        return all(map(partial(self.__blocked, king=True), self.get_surrounding_tiles(self.get_tile(self.get_king())))) if self.get_king() else False

    def check_king_escaped(self) -> bool:
        king = self.get_king()
        return self.get_tile(king).is_exit if king else False

    def __check_surround(self):
        """Recursive algorithm to kill surrounded groups"""

        def next_tiles(tiles: List[Tile]) -> List[Tile]:
            tiles = list(filter(lambda t: t is not None, tiles))
            return list(filter(lambda t: t.piece.is_black != self.__moved_piece.is_black if t.piece else False, tiles))

        start_tiles = next_tiles(self.get_surrounding_tiles(self.__moved_piece))
        if len(start_tiles) == 0:
            return

        self.__recurse_exit = False
        self.__recurse_checked = []
        self.__recurse_surround(start_tiles[0], next_tiles)
        if self.__recurse_kill:
            for tile in filter(lambda x: x is not None, self.__recurse_checked):
                if tile.piece is not None and tile.piece.is_king:
                    self.king_captured = True
                    return

            [self._kill(piece) for piece in self.__recurse_checked]

    def __recurse_surround(self, tile, next_tiles_func):
        if self.__recurse_exit:
            return

        self.__recurse_checked.append(tile)
        tiles = self.get_surrounding_tiles(tile)
        if all(list(map(self.__blocked, tiles))):
            self.__recurse_kill = True
            next_tiles = list(filter(lambda t: t not in self.__recurse_checked, next_tiles_func(tiles)))
            if len(next_tiles) == 0:
                return

            [self.__recurse_surround(t, next_tiles_func) for t in next_tiles]
        else:
            self.__recurse_kill = False
            self.__recurse_exit = True
            return

    def _check_kill(self, piece: Piece) -> None:
        """
        Check whether a piece should be killed or not after a move. Kills a piece surrounding piece if yes.
        :param piece: :type hnefatafl.piece.Piece: piece that just moved to check if it kills something
        :return: None
        """
        piece = self.get_piece(piece)

        if piece.is_white or piece.is_king:
            for func in self._get_surrounding_funcs():
                p = func(piece)
                if p is not None:
                    p = p.piece
                    if p is not None:
                        if p.is_black:
                            tile = func(p)
                            if tile is not None:
                                if tile.is_special and not self.get_piece(p).is_king:
                                    self._kill(p)
                                elif tile.piece is not None and not self.get_piece(p).is_king:
                                    if tile.piece.is_white:
                                        self._kill(p)
        elif piece.is_black:
            for func in self._get_surrounding_funcs():
                p = func(piece)
                if p is not None:
                    p = p.piece
                    if p is not None:
                        if p.is_white and not p.is_king:
                            tile = func(p)
                            if tile is not None:
                                if tile.is_special and not tile.piece:
                                    self._kill(p)
                                elif tile.piece is not None:
                                    if tile.piece.is_black:
                                        self._kill(p)

    def _update_game_over(self):
        if self.check_king_captured():
            self.king_captured = True
        elif self.check_king_escaped():
            self.king_escaped = True

    def move(self, move: Move, _check_game_end=True, _check_valid=True) -> None:
        """
        Move a piece from one tile to another, updates board state
        :param move: Move object defining the move
        :param _check_game_end: check if the game has ended after a move. Should only
        be used if in need of speed optimization.
        :param _check_valid: Check if the move is in the current valid moves of the board.
        Should also only be used for optimization.
        """
        tile, new_tile = move.tile, move.new_tile
        piece = self.get_piece(tile)

        if not piece:
            raise errors.InvalidMoveError('The tile has no piece on it to move.')
        if self.get_piece(new_tile) is not None:
            raise errors.InvalidMoveError('The designated tile cannot be moved to because it has a piece on it.')

        if _check_valid:
            if new_tile not in self.valid_moves(piece, ret_moves=False):
                raise errors.InvalidMoveError('Move is invalid.')

        self.__moved_piece = piece
        self._update_state(move)
        new_tile.piece = piece
        new_tile.update()
        self.get_tile(tile).piece = None

        self.__check_surround()
        self._check_kill(self.get_piece(new_tile))

        if self.max_repeats:
            piece_type = self.get_team_colour(piece.type)
            if not self.__first_repeat and self._repeat_exceeded():
                self.__first_repeat = piece_type

        if _check_game_end:
            self._update_game_over()


if __name__ == '__main__':
    while True:
        try:
            exec(input('>>> '))
        except Exception as e:
            print(e)
