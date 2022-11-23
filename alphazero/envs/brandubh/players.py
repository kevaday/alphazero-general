import pyximport, numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})

#from hnefatafl.engine import Move, BoardGameException
#from alphazero.envs.brandubh.tafl import get_action
from alphazero.envs.brandubh.fastafl import get_action as ft_get_action
from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState
from alphazero.Evaluator import MCTSEvaluator

from boardgame.board import Square
from boardgame.errors import InvalidMoveError


"""
class HumanTaflPlayer(BasePlayer):
    def play(self, state: GameState):
        valid_moves = state.valid_moves()

        def string_to_action(player_inp: str) -> int:
            try:
                move_lst = [int(x) for x in player_inp.split()]
                move = Move(state._board, move_lst)
                return get_action(state._board, move)
            except (ValueError, AttributeError, BoardGameException):
                return -1
        
        action = string_to_action(input(f"Enter the move to play for the player {state.player}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(f"Illegal move (action={action}, "
                                            f"in valids: {bool(valid_moves[action])}). Enter a valid move: "))

        return action
"""


class HumanFastaflPlayer(BasePlayer):
    @staticmethod
    def is_human() -> bool:
        return True

    def play(self, state: GameState):
        valid_moves = state.valid_moves()

        def string_to_action(player_inp: str) -> int:
            try:
                move_lst = [int(x) for x in player_inp.split()]
                return ft_get_action(state._board, (Square(*move_lst[:2]), Square(*move_lst[2:])))
            except (ValueError, AttributeError, InvalidMoveError):
                return -1
        
        action = string_to_action(input(f"Enter the move to play for the player {state.player}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(f"Illegal move (action={action}, "
                                            f"in valids: {bool(valid_moves[action])}). Enter a valid move: "))

        return action



class GreedyTaflPlayer(BasePlayer):
    def play(self, state: GameState):
        valids = state.valid_moves()
        candidates = []

        for a in range(state.action_size()):
            if not valids[a]: continue
            new_state = state.clone()
            new_state.play_action(a)
            candidates.append((-new_state.crude_value(), a))

        candidates.sort()
        return candidates[0][1]


class GreedyMCTSTaflPlayer(BasePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = MCTSEvaluator(
            args=self.args,
            model=self._crude_model,
            #num_sims=self.args.numMCTSSims
            max_search_time=20
        )

    def _crude_model(self, state: GameState):
        value = state.crude_value()
        return (
            np.full(state.action_size(), 1, dtype=np.float32),
            np.array([value, 1 - value, 0], dtype=np.float32)
        )

    def play(self, state: GameState):
        self.evaluator.run(state, block=True)
        print('[DEBUG] GreedyMCTS value:', self.evaluator.get_value())
        return self.evaluator.get_best_actions()[0]

    def update(self, state: GameState, action: int) -> None:
        self.evaluator.update(state, action)
