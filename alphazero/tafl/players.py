from hnefatafl.engine import Move, PieceType, Board, BoardGameException
from alphazero.tafl.tafl import get_action, TaflGame
from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState


class HumanTaflPlayer(BasePlayer):
    def play(self, state: GameState, turn: int):
        valid_moves = state.valid_moves()

        def string_to_action(player_inp: str) -> int:
            try:
                move_lst = [int(x) for x in player_inp.split()]
                move = Move(state._board, move_lst)
                return get_action(state._board, move)
            except (ValueError, AttributeError, BoardGameException):
                return -1

        action = string_to_action(input(f"Enter the move to play for the player {state.current_player()}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(f"Illegal move (action={action}, "
                                            f"in valids: {bool(valid_moves[action])}). Enter a valid move: "))

        return action


class GreedyTaflPlayer(BasePlayer):
    def play(self, state: GameState, turn: int):
        valids = state.valid_moves()
        candidates = []

        new_state = state.clone()
        for a in range(state.action_size()):
            if valids[a] == 0: continue

            new_state.play_action(a)
            candidates.append((-new_state.crude_value(), a))

        candidates.sort()
        return candidates[0][1]
