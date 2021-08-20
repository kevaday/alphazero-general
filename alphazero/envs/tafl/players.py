from hnefatafl.engine import Move, BoardGameException
from alphazero.envs.tafl.tafl import get_action
from alphazero.envs.tafl.fastafl import get_action as ft_get_action
from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState

import pyximport, numpy
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from fastafl.cengine import Square
from fastafl.errors import InvalidMoveError


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


class HumanFastaflPlayer(BasePlayer):
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
