from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState


class HumanGobangPlayer(BasePlayer):
    def play(self, state: GameState) -> int:
        valid = state.valid_moves()
        """
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / state._board.n), int(i % state._board.n))
        """

        while True:
            a = input('Enter a move: ')

            x, y = [int(x) for x in a.split(' ')]
            a = state._board.n * x + y if x != -1 else state._board.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid move entered.')

        return a


class GreedyGobangPlayer(BasePlayer):
    def play(self, state: GameState) -> int:
        valids = state.valid_moves()
        candidates = []

        for a in range(state.action_size()):
            if not valids[a]: continue

            next_state = state.clone()
            next_state.play_action(a)
            candidates += [(int(next_state.win_state()[next_state.player]), a)]

        candidates.sort()
        return candidates[0][1]
