import numpy as np


class HumanConnect4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board, turn):
        valid_moves = self.game.getValidMoves(board, 1)
        print('\nMoves:', [i for (i, valid)
                           in enumerate(valid_moves) if valid])

        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                print('Invalid move')
        return move


class OneStepLookaheadConnect4Player():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""

    def __init__(self, game, verbose=False):
        self.game = game
        self.verbose = verbose

    def play(self, board, turn):
        valid_moves = self.game.getValidMoves(board, 0)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid:
                continue
            state, _ = self.game.getNextState(board, 0, move)
            result = self.game.getGameEnded(state, 0)
            if result == -1:
                win_move_set.add(move)
            elif result == 1:
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose:
                print('Playing winning action %s from %s' %
                      (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose:
                print('Playing loss stopping action %s from %s' %
                      (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose:
                print('Playing random action %s from %s' %
                      (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' %
                            self.game.stringRepresentation(board))

        return ret_move
