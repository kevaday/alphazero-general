from hnefatafl.engine import Move, PieceType, Board, BoardGameException
from alphazero.tafl.tafl import get_action, TaflGame


class HumanTaflPlayer:
    def __init__(self, game: TaflGame):
        self.game = game

    def play(self, board: Board, turn):
        valid_moves = self.game.getValidMoves(board, 1)

        def string_to_action(player_inp: str) -> int:
            try:
                move_lst = [int(x) for x in player_inp.split()]
                move = Move(board, move_lst)
                return get_action(board, move)
            except (ValueError, AttributeError, BoardGameException):
                return -1

        action = string_to_action(input(f"Enter the move to play for the player {board.to_play()}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(f"Illegal move (action={action}, "
                                            f"in valids: {bool(valid_moves[action])}). Enter a valid move: "))

        return action
