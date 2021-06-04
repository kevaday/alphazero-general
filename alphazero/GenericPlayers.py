from alphazero.MCTS import MCTS
from alphazero.Game import Game
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict

from abc import ABC, abstractmethod

import numpy as np


class BasePlayer(ABC):
    def __init__(self, game: Game):
        self.game = game

    @abstractmethod
    def play(self, board, turn: int) -> int:
        pass


class RandomPlayer(BasePlayer):
    def __init__(self, game):
        super().__init__(game)

    def play(self, board, turn):
        valids = self.game.getValidMoves(board, self.game.getPlayers()[0])
        valids = valids / np.sum(valids)
        a = np.random.choice(self.game.getActionSize(), p=valids)
        return a


class NNPlayer(BasePlayer):
    def __init__(self, game, nn, temp=0, temp_threshold=10, args: dotdict = None):
        super().__init__(game)
        self.nn = nn
        self.temp = temp
        self.temp_threshold = temp_threshold
        if args: self.update_args(args)

    def update_args(self, args: dotdict):
        self.temp = args.arenaTemp
        self.temp_threshold = args.tempThreshold

    def play(self, board, turn: int) -> int:
        policy, _ = self.nn.predict(board)
        valids = self.game.getValidMoves(board, self.game.getPlayers()[0])
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
            np.arange(self.game.getActionSize()), p=probs
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
    def __init__(self, game, nn: NNetWrapper, temp=0, temp_threshold=10, num_sims=50, cpuct=2,
                 reset_mcts=True, verbose=False, args: dotdict = None):
        super().__init__(game)
        self.nn = nn
        self.temp = temp
        self.temp_threshold = temp_threshold
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.reset_mcts = reset_mcts
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

        self.mcts = MCTS(self.game, self.nn, self.args)

    def play(self, board, turn) -> int:
        if self.reset_mcts and turn <= 2:
            self.mcts.reset()

        temp = 1 if turn <= self.temp_threshold else self.temp
        policy = self.mcts.getActionProb(board, temp=temp)
        
        if self.verbose: print('max tree depth:', self.mcts.depth)

        return np.random.choice(len(policy), p=policy)
