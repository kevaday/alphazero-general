from hnefatafl.engine import Move, PieceType, Board, BoardGameException
from alphazero.tafl.tafl import get_action, TaflGame
from alphazero.GenericPlayers import BasePlayer


class HumanTaflPlayer(BasePlayer):
    def __init__(self, game: TaflGame):
        super().__init__(game)

    def play(self, board: Board, turn):
        valid_moves = self.game.getValidMoves(board, self.game.getPlayers()[0])

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


class GreedyTaflPlayer(BasePlayer):
    def __init__(self, game: TaflGame):
        super().__init__(game)

    def play(self, board: Board, turn):
        player = self.game.getPlayers()[0]
        valids = self.game.getValidMoves(board, player)
        candidates = []

        for a in range(self.game.getActionSize()):
            if valids[a] == 0: continue

            next_board, _ = self.game.getNextState(board, player, a)
            score = self.game.getScore(next_board, player)
            candidates.append((-score, a))

        candidates.sort()
        return candidates[0][1]
