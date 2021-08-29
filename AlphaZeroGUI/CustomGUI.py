from abc import ABC, abstractmethod
from alphazero.Game import GameState


class CustomGUI(ABC):
    @abstractmethod
    def update_state(self, state: GameState):
        """Update the state of the GUI based on the given game state."""
        pass

    @abstractmethod
    def get_state(self) -> GameState:
        """Convert the displayed board into a GameState and return it."""
        pass
