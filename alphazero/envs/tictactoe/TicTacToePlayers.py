from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState


class HumanTicTacToePlayer(BasePlayer):
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
