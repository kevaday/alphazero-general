from abc import ABC, abstractmethod
from typing import Tuple
from alphazero.Game import GameState

import numpy as np


class NeuralNet(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game_cls: GameState):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def train(self, examples, num_steps: int) -> Tuple[float, float]:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
            num_steps: the number of training steps to perform. Each step, a batch
                       is fed through the network and backpropogated.
        Returns:
            pi_loss: the average loss of the policy head during the training as a float
            val_loss: the average loss of the value head during the training as a float
        """
        pass

    @abstractmethod
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Input:
            board: current board as a numpy array

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.action_size()
            v: a float in float range [-1,1] that gives the value of the current board
        """
        pass

    @abstractmethod
    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    @abstractmethod
    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
