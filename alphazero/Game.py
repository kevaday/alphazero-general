from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class GameState(ABC):
    def __init__(self, board):
        self._board = board
        self._player = self.get_players()[0]
        self.turns = 0

    @abstractmethod
    def __eq__(self, other: 'GameState') -> bool:
        """Compare the current game state to an other"""
        pass

    @abstractmethod
    def clone(self) -> 'GameState':
        """Return a new clone of the game state, independent of the current one."""
        pass

    @staticmethod
    @abstractmethod
    def action_size() -> int:
        """The size of the action space for the game"""
        pass

    @staticmethod
    @abstractmethod
    def observation_size() -> Tuple[int, int, int]:
        """
        Returns:
            observation_size: the shape of observations of the current state,
                             must be in the form channels x width x height.
                             If only one plane is needed for observation, use 1 for channels.
        """
        pass

    @abstractmethod
    def valid_moves(self):
        """Returns a numpy binary array containing zeros for invalid moves and ones for valids."""
        pass

    @staticmethod
    @abstractmethod
    def get_players() -> List[int]:
        """
        Returns:
            players: a list of the players in the game.
                     Should be list(range(N)) where N is the
                     constant number of players in the game.
        """
        pass

    def current_player(self) -> int:
        return self._player

    def _next_player(self, player, turns=1):
        return (player + turns) % len(self.get_players())

    def _update_turn(self):
        """Should be called at the end of play_action"""
        self._player = self._next_player(self._player)
        self.turns += 1

    @abstractmethod
    def play_action(self, action: int) -> None:
        """Play the action in the current state given by argument action."""
        pass

    @abstractmethod
    def win_state(self) -> Tuple[bool, ...]:
        """
        Get the win state of the game, a tuple of boolean values
        for each player indicating if they have won, plus one more
        boolean at the end to indicate a draw.
        """
        pass

    @abstractmethod
    def observation(self):
        """Get an observation from the game state in the form of a numpy array with the size of self.observation_size"""
        pass

    @abstractmethod
    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        """
        Args:
            pi: the current policy for the given canonical state

        Returns:
            symmetries: list of state, pi pairs for symmetric samples of
                        the given state and pi (ex: mirror, rotation).
                        This is an optional method as symmetric samples
                        can be disabled for training.
        """
        pass

    def __str__(self):
        return f'Player:\t{self._player}\n{self._board}\n'
