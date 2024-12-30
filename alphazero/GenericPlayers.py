from alphazero.MCTS import MCTS
from alphazero.Game import GameState
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict, plot_mcts_tree

from abc import ABC, abstractmethod

import numpy as np
import torch


class BasePlayer(ABC):
    def __init__(self, game_cls: GameState = None, args: dotdict = None, verbose: bool = False):
        self.game_cls = game_cls
        self.args = args
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        return self.play(*args, **kwargs)

    @staticmethod
    def supports_process() -> bool:
        return False

    @staticmethod
    def requires_model() -> bool:
        return False

    @staticmethod
    def is_human() -> bool:
        return False

    def update(self, state: GameState, action: int) -> None:
        pass

    def reset(self):
        pass

    @abstractmethod
    def play(self, state: GameState) -> int:
        pass

    def process(self, batch):
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    def play(self, state):
        valids = state.valid_moves()
        valids = valids / np.sum(valids)
        a = np.random.choice(state.action_size(), p=valids)
        return a


class NNPlayer(BasePlayer):
    def __init__(self, nn: NNetWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.temp = self.args.startTemp

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def play(self, state) -> int:
        policy, _ = self.nn.predict(state.observation())
        valids = state.valid_moves()
        options = policy * valids
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        if self.temp == 0:
            bestA = np.argmax(options)
            probs = [0] * len(options)
            probs[bestA] = 1
        else:
            probs = [x ** (1. / self.temp) for x in options]
            probs /= np.sum(probs)

        choice = np.random.choice(
            np.arange(state.action_size()), p=probs
        )

        if valids[choice] == 0:
            print()
            print(self.temp)
            print(valids)
            print(policy)
            print(probs)
            assert valids[choice] > 0

        return choice

    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)


# class AlphaBetaPlayer(BasePlayer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(None, *args, **kwargs)
#         self.eval_func = self.args.eval_func
#         self.depth = self.args.depth
#     def play(self, state) -> int:
#         vals = []
#         for i, a in enumerate(state.valid_moves()):
#             if a == 1:
#                 s = state.clone()
                
#     def alphaBeta(self, state, depth, alpha, beta, maximizingPlayer):
#         if depth == 0 or np.any(state.win_state()):
#             return self.eval_func(state)

#         if maximizingPlayer:
#             maxEval = -np.inf
#             for i, a in enumerate(state.valid_moves()):
#                 if a == 1:
#                     s = state.clone()
#                     s.play_action(i)
#                     EVAL = self.alphaBeta(s, depth-1, alpha, beta, False)
#                     maxEval = max(maxEval, EVAL)
#                     alpha = max(alpha, EVAl)
#                     if beta <= alpha:
#                         break
#             return maxEval
#         else:
#             maxEval = np.inf
#             for i, a in enumerate(state.valid_moves()):
#                 if a == 1:
#                     s = state.clone()
#                     s.play_action(i)
#                     EVAL = self.alphaBeta(s, depth-1, alpha, beta, True)
#                     maxEval = min(maxEval, EVAL)
#                     beta = min(beta, EVAl)
#                     if beta <= alpha:
#                         break
#             return maxEval
        


class MCTSPlayer(BasePlayer):
    def __init__(self, nn: NNetWrapper, *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.reset()
        if self.verbose:
            self.mcts.search(
                self.game_cls(), self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp
            )
            value = self.mcts.value(self.average_value)
            self.__rel_val_split = value if value > 0.5 else 1 - value
            print('initial value:', self.__rel_val_split)

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def update(self, state: GameState, action: int) -> None:
        self.mcts.update_root(state, action)

    def reset(self):
        self.mcts = MCTS(self.args)

    def play(self, state) -> int:
        self.mcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        policy = self.mcts.probs(state, self.temp)

        if self.print_policy:
            print(f'policy: {policy}')

        if self.verbose:
            _, value = self.nn.predict(state.observation())
            print('max tree depth:', self.mcts.max_depth)
            print(f'raw network value: {value}')

            value = self.mcts.value(self.average_value)
            rel_val = 0.5 * (value - self.__rel_val_split) / (1 - self.__rel_val_split) + 0.5 \
                if value >= self.__rel_val_split else (value / self.__rel_val_split) * 0.5

            print(f'value for player {state.player}: {value}')
            print('relative value:', rel_val)

        if self.draw_mcts:
            plot_mcts_tree(self.mcts, max_depth=self.draw_depth)

        action = np.random.choice(len(policy), p=policy)
        if self.verbose:
            print('confidence of action:', policy[action])

        return action

    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)


class RawMCTSPlayer(MCTSPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self._POLICY_SIZE = self.game_cls.action_size()
        self._POLICY_FILL_VALUE = 1 / self._POLICY_SIZE
        self._VALUE_SIZE = self.game_cls.num_players() + 1

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return False

    def play(self, state) -> int:
        self.mcts.raw_search(state, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        policy = self.mcts.probs(state, self.temp)
        action = np.random.choice(len(policy), p=policy)
        if self.verbose:
            print('max tree depth:', self.mcts.max_depth)
            print(f'value for player {state.player}: {self.mcts.value(self.average_value)}')
            print(f'policy: {policy}')
            print('confidence of action:', policy[action])

        if self.draw_mcts:
            plot_mcts_tree(self.mcts, max_depth=self.draw_depth)

        return action

    def process(self, batch: torch.Tensor):
        return torch.full((batch.shape[0], self._POLICY_SIZE), self._POLICY_FILL_VALUE).to(batch.device), \
               torch.zeros(batch.shape[0], self._VALUE_SIZE).to(batch.device)
