from pyximport import install as pyxinstall
from numpy import get_include

pyxinstall(setup_args={'include_dirs': get_include()})

from alphazero.Coach import DEFAULT_ARGS
from alphazero.Game import GameState
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Tuple, List, Union
from torch import from_numpy

# Options for args eval
from torch.optim import *
from torch.optim.lr_scheduler import *
from alphazero.GenericPlayers import *
from alphazero.utils import default_temp_scaling, const_temp_scaling

import multiprocessing as mp
import threading
import numpy as np
import time
import json
import os

CALLABLE_PREFIX = '__CALLABLE__'


# TODO: Make evaluator work with process instead of thread
# class BaseEvaluator(ABC):
#     _NULL_VALUE = -1
#
#     def __init__(self):
#         self._manager = mp.Manager()
#         self._best_actions = self._manager.list()
#         self._value = mp.Value('d', self._NULL_VALUE)
#         self.__last_state = None
#         self.__current_state = None
#         self._updated = True
#         self._run_process = None
#         self._stop_event = mp.Event()
#
#     @property
#     def last_state(self) -> Optional[GameState]:
#         return self.__last_state
#
#     @property
#     def current_state(self) -> Optional[GameState]:
#         return self.__current_state
#
#     @property
#     def is_running(self) -> bool:
#         return not self._stop_event.is_set()
#
#     @abstractmethod
#     def _run(self, state: GameState) -> None:
#         """Must be implemented by subclass, runs the evaluator
#         on the given state. super()._run(state) must be called
#         at the end of the method.
#         """
#         ...
#         if self.is_running:
#             self._stop_event.set()
#         self.__last_state = state.clone()
#         self._updated = False
#
#     def update(self, state: GameState, action: int) -> None:
#         """Must be called when a player performs an action on the state.
#         The state must be before the action is performed. This is done
#         automatically and doesn't need to be called if run() is called
#         after every new state in the game.
#         """
#         ...
#         self._updated = True
#
#     def run(self, state: GameState, block=False) -> None:
#         if self._run_process is not None and self._run_process.is_alive():
#             raise RuntimeError('Evaluator is already running')
#
#         self.__current_state = state.clone()
#         self._stop_event.clear()
#         if not self._updated and self.last_state is not None and state.last_action is not None:
#             self.update(self.last_state, state.last_action)
#
#         self._run_process = mp.Process(target=self._run, args=(state,), daemon=True)
#         self._run_process.start()
#         if block:
#             self._run_process.join()
#
#     def stop(self, block=True) -> None:
#         """Stop the evaluator. Keeps the last calculated value."""
#         if not self.is_running:
#             return
#         self._stop_event.set()
#         if block and self._run_process is not None and self._run_process.is_alive():
#             self._run_process.join()
#
#     def get_value(self, player: int = 0) -> Optional[float]:
#         """Get the value of the current state for the given player."""
#         with self._value.get_lock():
#             value = self._value.value
#             if value != self._NULL_VALUE:
#                 return value if player == 0 else 1 - value
#
#     def _set_value(self, value: float) -> None:
#         """This method must be called by the parent class
#         every time a new value is calculated.
#         """
#         with self._value.get_lock():
#             if value != self._value.value:
#                 self._value.value = value
#
#     def get_best_actions(self) -> List[int]:
#         """Returns a list of all actions sorted by their value
#         based on the current state.
#         """
#         return list(self._best_actions)
#
#     def _set_best_actions(self, actions: List[int]) -> None:
#         """Must be called by the parent class every time a new
#         set of best actions is calculated.
#         """
#         self._best_actions.clear()
#         self._best_actions.extend(actions)
#
#
# class MCTSEvaluator(BaseEvaluator):
#     def __init__(self, args=DEFAULT_ARGS,
#                  model: Union[Callable[[GameState], Tuple[np.ndarray, np.ndarray]], NNetWrapper] = None,
#                  num_sims: int = None, max_search_depth: int = None, max_search_time: float = None,
#                  best_actions_temp: float = 1, average_children=False):
#         super().__init__()
#         self.model = model
#         if isinstance(self.model, NNetWrapper):
#             self.nnet = self.model.nnet
#             self.args = self.model.args
#             self.model = self._nnet_model
#
#         self.average_children = average_children
#         self.best_actions_temp = best_actions_temp
#
#         self.num_sims = num_sims
#         self.max_search_depth = max_search_depth
#         self.max_search_time = max_search_time
#         self._mcts = MCTS(args)
#
#     def _nnet_model(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
#         return NNetWrapper.predict(self, state.observation())
#
#     def _raw_model(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
#         # always use uniform value and policy if no model is given
#         # v = np.zeros(state.num_players() + 1, dtype=np.float32)
#         v = np.full(state.num_players() + 1, 0.5, dtype=np.float32)
#         v[-1] = 0  # assume one player is always the winner
#         p = np.full(state.action_size(), 1, dtype=np.float32)
#         return p, v
#
#     def _search(self, state: GameState, model: Callable[[GameState], Tuple[np.ndarray, np.ndarray]],
#                 sims: int = None, add_root_noise: bool = False, add_root_temp: bool = False):
#         self._mcts.max_depth = 0
#         num_sims = 0
#         start_time = time.time()
#
#         while (num_sims < sims) if sims else True:
#             if self._stop_event.is_set():
#                 break
#
#             leaf = self._mcts.find_leaf(state)
#             p, v = model(leaf)
#             self._mcts.process_results(leaf, v, p, add_root_noise, add_root_temp)
#             self._set_value(self._mcts.value(average=self.average_children))
#
#             # TODO: TypeError: cannot pickle 'weakref' object
#             try:
#                 probs = self._mcts.probs(state, temp=self.best_actions_temp)
#             except FloatingPointError:
#                 probs = np.full(state.action_size(), 1 / state.action_size())
#             valids = state.valid_moves()
#             # create a list of actions sorted by their probability
#             self._set_best_actions([
#                 action for action, prob in sorted(enumerate(probs), key=lambda x: x[1], reverse=True) if valids[action]
#             ])
#
#             if (
#                 self.max_search_depth is not None and self._mcts.max_depth > self.max_search_depth
#                 or self.max_search_time is not None and time.time() - start_time > self.max_search_time
#             ):
#                 self._stop_event.set()
#                 break
#             num_sims += 1
#
#     def _run(self, state: GameState, *args, **kwargs) -> None:
#         if self.model is None:
#             self.model = self._raw_model
#
#         self._search(state, self.model, self.num_sims, *args, **kwargs)
#         super()._run(state)
#
#     def update(self, state: GameState, action: int):
#         self._mcts.update_root(state, action)
#         super().update(state, action)


class BaseEvaluator(ABC):
    def __init__(self):
        self._value = None
        self.__last_state = None
        self.__current_state = None
        self._updated = True
        self._run_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def last_state(self) -> Optional[GameState]:
        return self.__last_state

    @property
    def current_state(self) -> Optional[GameState]:
        return self.__current_state

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    @abstractmethod
    def _run(self, state: GameState) -> None:
        """Must be implemented by subclass, runs the evaluator
        on the given state. super()._run(state) must be called
        at the end of the method.
        """
        ...
        if self.is_running:
            self._stop_event.set()
        self.__last_state = state.clone()
        self._updated = False

    def update(self, state: GameState, action: int) -> None:
        """Must be called when a player performs an action on the state.
        The state must be before the action is performed. This is done
        automatically and doesn't need to be called if run() is called
        after every new state in the game.
        """
        ...
        self._updated = True

    def run(self, state: GameState, block=False) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            raise RuntimeError('Evaluator is already running')

        self.__current_state = state.clone()
        self._stop_event.clear()
        if not self._updated and self.last_state is not None and state.last_action is not None:
            self.update(self.last_state, state.last_action)

        self._run_thread = threading.Thread(target=self._run, args=(state,), daemon=True)
        self._run_thread.start()
        if block:
            self._run_thread.join()

    def stop(self, block=True) -> None:
        """Stop the evaluator. Keeps the last calculated value."""
        if not self.is_running:
            return
        self._stop_event.set()
        if block and self._run_thread is not None and self._run_thread.is_alive():
            self._run_thread.join()

    def get_value(self, player: int = 0) -> float:
        """Get the value of the current state for the given player."""
        with self._lock:
            return self._value if player == 0 else 1 - self._value

    @abstractmethod
    def get_best_actions(self) -> List[int]:
        """Returns a list of all actions sorted by their value
        based on the current state.
        """
        pass

    def _set_value(self, value: float) -> None:
        """This method must be called by the parent class
        every time a new value is calculated.
        """
        with self._lock:
            if value != self._value:
                self._value = value


class MCTSEvaluator(BaseEvaluator):
    def __init__(self, args=DEFAULT_ARGS,
                 model: Union[Callable[[GameState], Tuple[np.ndarray, np.ndarray]], NNetWrapper] = None,
                 num_sims: int = None, max_search_depth: int = None, max_search_time: float = None,
                 best_actions_temp: float = 1, average_children=False):
        super().__init__()
        self.model = model
        if isinstance(self.model, NNetWrapper):
            self.model = lambda state: model(state.observation())
        self.average_children = average_children
        self.best_actions_temp = best_actions_temp

        self.num_sims = num_sims
        self.max_search_depth = max_search_depth
        self.max_search_time = max_search_time
        self._mcts = MCTS(args)

    def _search(self, state: GameState, model: Callable[[GameState], Tuple[np.ndarray, np.ndarray]],
                sims: int = None, add_root_noise: bool = False, add_root_temp: bool = False):
        self._mcts.max_depth = 0
        num_sims = 0
        start_time = time.time()

        while (num_sims < sims) if sims else True:
            if self._stop_event.is_set():
                break

            leaf = self._mcts.find_leaf(state)
            p, v = model(leaf)
            self._mcts.process_results(leaf, v, p, add_root_noise, add_root_temp)
            self._set_value(self._mcts.value(average=self.average_children))

            if (
                self.max_search_depth is not None and self._mcts.max_depth > self.max_search_depth
                or self.max_search_time is not None and time.time() - start_time > self.max_search_time
            ):
                self._stop_event.set()
                break
            num_sims += 1

    def _run(self, state: GameState, *args, **kwargs) -> None:
        if self.model is None:
            # always use uniform value and policy if no model is given
            # v = np.zeros(state.num_players() + 1, dtype=np.float32)
            v = np.full(state.num_players() + 1, 0.5, dtype=np.float32)
            v[-1] = 0  # assume one player is always the winner
            p = np.full(state.action_size(), 1, dtype=np.float32)
            self.model = lambda x: (p, v)

        self._search(state, self.model, self.num_sims, *args, **kwargs)
        super()._run(state)

    def update(self, state: GameState, action: int):
        self._mcts.update_root(state, action)
        super().update(state, action)

    def get_best_actions(self) -> List[int]:
        if self.current_state:
            try:
                probs = self._mcts.probs(self.current_state, temp=self.best_actions_temp)
            except FloatingPointError:
                probs = np.full(self.current_state.action_size(), 1 / self.current_state.action_size())

            valids = self.current_state.valid_moves()
            # create a list of actions sorted by their probability
            return [
                action for action, prob in sorted(enumerate(probs), key=lambda x: x[1], reverse=True) if valids[action]
            ]
        return []


def load_args_file(filepath: str) -> dotdict:
    new_args = dotdict()
    raw_args = json.load(open(filepath, 'r'))

    for k, v in raw_args.items():
        if isinstance(v, str) and CALLABLE_PREFIX in v:
            try:
                v = eval(v.replace(CALLABLE_PREFIX, ''))
            except Exception as e:
                raise RuntimeError('Failed to parse argument file: ' + str(e))

        elif isinstance(v, dict):
            v = dotdict(v)

        new_args.update({k: v})

    return new_args


def save_args_file(args: dotdict or dict, filepath, replace=True):
    if not replace and os.path.exists(filepath): return

    save_args = dict()
    for k, v in args.items():
        if callable(v):
            v = CALLABLE_PREFIX + v.__name__
        save_args.update({k: v})

    with open(filepath, 'w') as f:
        json.dump(save_args, f)

    return save_args
