from alphazero.GenericPlayers import BasePlayer

import pyximport, numpy
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.envs.stratego.stratego import get_action, Game, Square
from boardgame.errors import InvalidMoveError


class HumanStrategoPlayer(BasePlayer):
    def play(self, state: Game) -> int:
        valid_moves = state.valid_moves()

        def string_to_action(player_inp: str) -> int:
            try:
                move_lst = [int(x) for x in player_inp.split()]

                if state._board.play_phase:
                    return get_action(state._board, (Square(*move_lst[:2]), Square(*move_lst[2:])))
                else:
                    return get_action(state._board, (move_lst[0], Square(*move_lst[1:])))
            except (ValueError, AttributeError, InvalidMoveError):
                return -1

        action = string_to_action(input(f"Enter the move to play for the player {state.player}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(
                f"Illegal move (action={action}, in valids: {bool(valid_moves[action])}). Enter a valid move: "
            ))

        return action
