from boardgame import BaseBot, BaseBoard, BasePlayer, Move
from hnefatafl.engine import PieceType, variants
from hnefatafl import MODEL_PATH, MODEL_CONFIG_PATH

from alphazero.GenericPlayers import MCTSPlayer, NNPlayer

from threading import Lock

import os
import importlib
import random


class AlphaZeroBot(BaseBot):
    def __init__(self, player: BasePlayer, game_variant=variants.hnefatafl, use_mcts=True, use_default_args=True,
                 load=True, args=None, *a, **k):
        super().__init__(player)
        self.use_mcts = use_mcts
        self.use_default_args = use_default_args
        self.game_variant = game_variant
        self._game = None
        self._args = args
        self._model_player = None
        self.__a = a
        self.__k = k
        self.result = None
        self._result_lock = Lock()
        if load: self.load_model(MODEL_PATH)

    def reset(self):
        from alphazero.envs.tafl.train import args
        from alphazero.envs.tafl.tafl import TaflGame
        self._args = self._args if self._args is not None else args
        self._game = TaflGame()
        if self._model_player and self.use_mcts:
            self._model_player.mcts.reset()
    
    def update(self, board: BaseBoard, move: Move):
        if self.use_mcts:
            from alphazero.envs.tafl.tafl import get_action

            self._game._board = board
            self._game._player = 2 - board.to_play().value
            self._game._turns = board.num_turns
            self._model_player.update(self._game, get_action(board, move))

    def load_model(self, model_path: str):
        from alphazero.NNetWrapper import NNetWrapper

        self.reset()
        nn = NNetWrapper(type(self._game), self._args)
        nn.load_checkpoint('.', model_path)

        self.__k['args'] = self._args if not self.use_default_args else None
        if self.__k.get('verbose'): print('Loading model with args:', self.__k['args'])
        cls = MCTSPlayer if self.use_mcts else NNPlayer
        self._model_player = cls(type(self._game), nn, *self.__a, **self.__k)

    def get_move(self, board: BaseBoard) -> Move or None:
        self.result = None

        from alphazero.envs.tafl.tafl import get_move

        self._game._board = board
        self._game._player = 2 - board.to_play().value
        self._game._turns = board.num_turns
        action = self._model_player(self._game)
        move = get_move(board, action)

        self._result_lock.acquire()
        self.result = move
        self._result_lock.release()

        return move


class MuZeroBot(BaseBot):
    def __init__(self, player: BasePlayer, load=True, use_new_config=True):
        super().__init__(player)
        self.use_new_config = use_new_config
        self._model = None
        self._config_module = None
        self._config_game = None
        self._config_params = None
        
        self.history = None
        self.result = None
        self._result_lock = Lock()
        if load: self.load_model(MODEL_PATH, MODEL_CONFIG_PATH)

    def load_model(self, model_path, config_path=None):
        from muzero_general.models import MuZeroNetwork
        import torch

        if not os.path.exists(model_path):
            raise IOError(f'The file {model_path} does not exist, could not load model.')

        if config_path and not self.use_new_config:
            try:
                self._config_module = importlib.import_module(
                    config_path.replace('.py', '') if '.py' in config_path else config_path
                )
            except ModuleNotFoundError:
                print(f'WARNING: The default model config file {config_path} could not be loaded, using config from model file instead.')
                self.use_new_config = True
            else:
                self._config_game = self._config_module.Game()
                self._config_params = self._config_module.MuZeroConfig()
        
        checkpoint = torch.load(model_path)
        if self.use_new_config and checkpoint.get('config'): self._config_params = checkpoint['config']
        
        self._model = MuZeroNetwork(self._config_params)
        self._model.set_weights(checkpoint['weights'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running model on ' + device)
        self._model.to(torch.device(device))
        self._model.eval()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        import numpy as np

        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def get_move(self, board: BaseBoard) -> Move or None:
        from muzero_general.self_play import MCTS, GameHistory
        import torch

        self.result = None

        observation = self._config_module.get_observation(
            board,
            1 if self.player.white else 0
        )
        
        if not self.history:
            self.history = GameHistory()
            self.history.action_history.append(0)
            self.history.observation_history.append(observation)

        stacked_observations = self.history.get_stacked_observations(
            -1,
            self._config_params.stacked_observations,
        )

        with torch.no_grad():
            root, mcts_info = MCTS(self._config_params).run(
                self._model,
                stacked_observations,
                self._config_game.legal_actions(board),
                self._config_game.to_play(board),
                True,
            )
            action = self.select_action(root, 0)

        move = self._config_module.get_move(board, action)
        self._result_lock.acquire()
        self.result = move
        self._result_lock.release()

        [print(f'{k}: {v} ', end='') for k, v in mcts_info.items()]
        print()

        self.history.store_search_statistics(root, self._config_params.action_space)
        self.history.action_history.append(action)
        self.history.observation_history.append(observation)

        return move


class RandomBot(BaseBot):
    def get_move(self, board: BaseBoard) -> Move or None:
        if not board.all_valid_moves(PieceType.white if self.player.white else PieceType.black): return

        if self.player.white:
            pieces = [p for p in board.pieces if p.is_white]
        else:
            pieces = [p for p in board.pieces if p.is_black]

        while True:
            piece = random.choice(pieces)
            moves = list(board.valid_moves(piece))
            if moves: break

        return random.choice(moves)
