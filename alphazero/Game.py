from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    @abstractmethod
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    @abstractmethod
    def getBoardSize(self) -> Tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    @abstractmethod
    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    @abstractmethod
    def getPlayers(self) -> List[int]:
        """
        Returns:
            players: a list of the players in the game.
                     Should be list(range(N)) where N is the
                     constant number of players in the game.
        """
        pass

    def getNextPlayer(self, currentPlayer: int, turns: int = 1) -> int:
        """
        A useful method to get the next player in a game for linear turns.
        Inputs:
            currentPlayer: the current player playing in the game
            turns: the number of turns in the future for getting the
                   next player to play
        Returns:
            nextPlayer: the player number who's turn it will be from
                        the `currentPlayer` after `turns` number of turns
                        in the game.
        """
        players = self.getPlayers()

        nextPlayer = currentPlayer + turns
        while nextPlayer not in players:
            if nextPlayer < 0:
                nextPlayer += len(players)
            else:
                nextPlayer -= len(players)

        return nextPlayer

    def getPlayerToPlay(self, board) -> int:
        """
        This method is only used in batched Arena MCTS comparison,
        so it's not a big deal if this can't be implemented correctly.
        Input:
            board: the current state of the game
        Returns:
            playerToPlay: the next player to play based on the given
                          state of the game. Should return the next player
                          independent of the canonical form (ex. counting
                          the number of turns played in the game and using modulo).
        """
        pass

    @abstractmethod
    def getObservationSize(self) -> Tuple[int, int, int]:
        """
        Returns:
            observationSize: the shape of observations of the current state,
                             must be in the form channels x width x height.
                             If only one plane is needed for observation, use 1 for channels.
        """
        pass

    @abstractmethod
    def getNextState(self, board, player: int, action: int, copy=True) -> Tuple[Any, int]:
        """
        Input:
            board: current board
            player: current player (from the range given by getPlayers method)
            action: action taken by current player
            copy: whether or not to create a new copy of `board` for performance optimization.

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    @abstractmethod
    def getValidMoves(self, board, player: int):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    @abstractmethod
    def getGameEnded(self, board, player: int) -> int:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    @abstractmethod
    def getCanonicalForm(self, board, player: int, copy=True):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            copy: whether or not to create a new copy of `board` for performance optimization.

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    @abstractmethod
    def stringRepresentation(self, state) -> str:
        """
        Input:
            state: current board

        Returns:
            boardString: a quick and unique conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass

    def getSymmetries(self, state, pi) -> List[Tuple[Any, int]]:
        """
        Input:
            state: current canonical state
            pi: the current policy for the given canonical state

        Returns:
            symmetries: list of state, pi pairs for symmetric samples of
                        the given state and pi (ex: mirror, rotation).
                        This is an optional method as symmetric samples
                        can be disabled for training.
        """
        pass
