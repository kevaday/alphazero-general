from alphazero.MCTS import MCTS
from alphazero.Game import GameState
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict

from abc import ABC, abstractmethod

import numpy as np


class BasePlayer(ABC):
    def __call__(self, *args, **kwargs):
        return self.play(*args, **kwargs)

    def update(self, state: GameState, action: int) -> None:
        pass

    def reset(self):
        pass

    @abstractmethod
    def play(self, state: GameState, turn: int) -> int:
        pass


class RandomPlayer(BasePlayer):
    def play(self, state, turn):
        valids = state.valid_moves()
        valids = valids / np.sum(valids)
        a = np.random.choice(state.action_size(), p=valids)
        return a


class NNPlayer(BasePlayer):
    def __init__(self, nn, temp=0, temp_threshold=10, args: dotdict = None):
        self.nn = nn
        self.temp = temp
        self.temp_threshold = temp_threshold
        if args: self.update_args(args)

    def update_args(self, args: dotdict):
        self.temp = args.arenaTemp
        self.temp_threshold = args.tempThreshold

    def play(self, state, turn: int) -> int:
        policy, _ = self.nn.predict(state)
        valids = state.valid_moves()
        options = policy * valids
        temp = 1 if turn <= self.temp_threshold else self.temp
        if temp == 0:
            bestA = np.argmax(options)
            probs = [0] * len(options)
            probs[bestA] = 1
        else:
            probs = [x ** (1. / temp) for x in options]
            probs /= np.sum(probs)

        choice = np.random.choice(
            np.arange(state.action_size()), p=probs
        )

        if valids[choice] == 0:
            print()
            print(temp)
            print(valids)
            print(policy)
            print(probs)
            assert valids[choice] > 0

        return choice


class MCTSPlayer(BasePlayer):
    def __init__(self, nn: NNetWrapper, temp=0, temp_threshold=10, num_sims=50, cpuct=2,
                 verbose=False, args: dotdict = None):
        self.nn = nn
        self.temp = temp
        self.temp_threshold = temp_threshold
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.verbose = verbose
        self.args = args
        self.mcts = None
        self.update_args(args)

    def update_args(self, args: dotdict = None):
        if args:
            self.args = args
            self.temp = args.temp
            self.temp_threshold = args.tempThreshold
            self.num_sims = args.numMCTSSims
            self.cpuct = args.cpuct
        else:
            self.args = dotdict({
                'temp': self.temp,
                'tempThreshold': self.temp_threshold,
                'numMCTSSims': self.num_sims,
                'cpuct': self.cpuct
            })

        self.reset()

    def update(self, state: GameState, action: int) -> None:
        self.mcts.update_root(state, action)

    def reset(self):
        self.mcts = MCTS(self.cpuct)

    def play(self, state, turn) -> int:
        self.mcts.search(state, self.nn, self.num_sims)
        temp = 1 if turn <= self.temp_threshold else self.temp
        policy = self.mcts.probs(state, temp)
        
        if self.verbose: print('max tree depth:', len(self.mcts.path))

        return np.random.choice(len(policy), p=policy)
