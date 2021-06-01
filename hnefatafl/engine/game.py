from hnefatafl import GAME_HELP_MSG
from hnefatafl.engine.board import Board, PieceType
from boardgame import BasePlayer, BaseGame, BaseBot, BaseBoard, Win, Move
from boardgame.errors import TurnError, InvalidMoveError


class Player(BasePlayer):
    def __init__(self, white, bot=False):
        super().__init__(not white, white, bot)


class Game(BaseGame):
    def __init__(self, board: Board = None, white_player: Player = None, black_player: Player = None, **kwargs):
        if not board: board = Board()
        if not white_player: white_player = Player(True, False)
        if not black_player: black_player = Player(False, False)
        super().__init__(board, white_player, black_player, **kwargs)

    def serialize(self) -> bytes:
        b = self.id.encode()
        b += bytes([int(self.started)]) + bytes([int(self.black.is_turn)]) + \
             bytes([int(self.game_over)]) + bytes([int(self.black.won)]) + bytes([int(self.board.is_custom)])
        b += bytes([self.board.num_start_black]) + bytes([self.board.num_start_white])
        return b + self.board.to_string(add_values=True, add_spaces=False).encode()

    @classmethod
    def from_serial(cls, serial: bytes, _board: Board = None) -> 'Game':
        uuid = serial[:32].decode()
        started, black_turn, game_over, black_won, is_custom = [bool(b) for b in serial[32:37]]
        num_black, num_white = serial[37:39]
        game = cls(
            _board or Board(custom_board=serial[39:].decode(), num_start_black=num_black,
                            num_start_white=num_white, custom=is_custom), id=uuid
        )
        game.started = started
        game.black.is_turn = black_turn
        game.white.is_turn = not black_turn
        game.game_over = game_over
        game.black.won = black_won
        game.white.won = game_over and not black_won
        return game

    def move(self, move: Move):
        tile = move.tile
        piece = self.board.get_piece(tile)
        if not piece:
            raise InvalidMoveError(f'There is no piece on the tile {repr(tile)}.')

        if piece.is_white and self.white.is_turn or piece.is_black and self.black.is_turn:
            self.board.move(move)
            if self.board.king_captured:
                self._black_won()
                return
            elif self.board.king_escaped:
                self._white_won()
                return
            self._update_turn()

            # Check if moves available for each team. If not, opposing team wins
            if self.white.is_turn:
                if self._check_moves(PieceType.white):
                    self._black_won()
            else:
                if self._check_moves(PieceType.black):
                    self._white_won()

        else:  # It's not that player's turn
            raise TurnError(f"The piece with coordinates {piece.x},{piece.y} cannot be moved because"
                            f"it is not {bool_to_colour(self.black.is_turn)}'s turn.")

    def __copy__(self, *args, **kwargs):
        serial = self.serialize()
        return self.from_serial(serial, _board=self.board.copy(*args, **kwargs))

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)


def is_turn(is_white: bool, game: Game):
    return (is_white and game.white.is_turn) or (not is_white and game.black.is_turn)


def bool_to_colour(is_white: bool) -> str:
    return 'white' if is_white else 'black'


def string_to_move(move: str):
    COMMA = ','
    SPACE = ' '

    move = move.split(COMMA) if COMMA in move else move.split(SPACE)
    if move[-1] == SPACE:
        del move[-1]
    if len(move) != 4:
        return
    try:
        return [int(c) for c in move]
    except ValueError:
        return


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        board = Board(load_file=sys.argv[1])
    else:
        board = Board()
    game = Game(board=board)
    print(GAME_HELP_MSG + '\n')


    def print_invalid_move():
        print('Invalid move entered.\n' + GAME_HELP_MSG + '\n')


    def play_again() -> bool:
        return input('Play again? (y/n) ').lower() in 'y yes yea yep yeah'.split()


    def print_board():
        print(game.board.to_string(add_values=False, add_spaces=True))


    while True:
        print_board()
        print(f"{bool_to_colour(game.white.is_turn).capitalize()}'s turn.")
        print(f'{game.board.num_white}/{game.board.num_start_white} white pieces')
        print(f'{game.board.num_black}/{game.board.num_start_black} black pieces')
        move = string_to_move(input('Move: '))
        if not move:
            print_invalid_move()
            continue

        try:
            # import pdb;pdb.set_trace()
            game.move(*move)
        except TurnError:
            print_invalid_move()
        except Win:
            print_board()
            print(f"{bool_to_colour(game.white.won).capitalize()} has won the game!")
            if not play_again():
                break
            game.reset()
        print('\n\n')
