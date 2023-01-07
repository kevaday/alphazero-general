from typing import List, Tuple, Any

from alphazero.envs.reversi_othello.OthelloLogic import Board
from alphazero.Game import GameState

import numpy as np

MULTI_PLANE_OBSERVATION = True
NUM_PLAYERS = 2
NUM_CHANNELS = 4 if MULTI_PLANE_OBSERVATION else 1
BOARD_SIZE = 8
MAX_TURNS = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = BOARD_SIZE ** 2 + 1
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class Game(GameState):
	def __init__(self, _board=None):
		super().__init__(_board or self._get_board())

	def __hash__(self) -> int:
		return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

	def __eq__(self, other: 'Game') -> bool:
		return (
				np.asarray(self._board.pieces) == np.asarray(other._board.pieces)
				and self._player == other._player
				and self.turns == other.turns
		)

	def display(self):
		#display(self._board.pieces)
		print(np.asarray(self._board.pieces))

	@staticmethod
	def _get_board(*args, **kwargs):
		return Board(BOARD_SIZE, *args, **kwargs)

	def clone(self) -> 'Game':
		board = self._get_board(_pieces=np.copy(np.asarray(self._board.pieces)))
		game = Game(_board=board)
		game._player = self._player
		game._turns = self.turns
		return game

	@staticmethod
	def action_size() -> int:
		return ACTION_SIZE

	@staticmethod
	def observation_size() -> Tuple[int, int, int]:
		return OBSERVATION_SIZE

	@staticmethod
	def num_players() -> int:
		return NUM_PLAYERS

	@staticmethod
	def max_turns() -> int:
		return MAX_TURNS

	@staticmethod
	def has_draw() -> bool:
		return True

	def _player_range(self):
		return (-1, 1)[self.player]

	def valid_moves(self):
		valids = [0] * self.action_size()
		moves = self._board.get_legal(self._player_range())

		if len(moves) == 0:
			valids[-1] = 1
			return np.array(valids, dtype=np.intc)

		for x, y in moves:
			valids[self._board.n * x + y] = 1
		return np.array(valids, dtype=np.intc)

	def play_action(self, action: int):
		super().play_action(action)
		#Pass if no moves are possible
		if action == self._board.n * self._board.n:
			self._update_turn()
			return
		move = (action // self._board.n, action % self._board.n)
		self._board.execute_move(move, self._player_range())
		self._update_turn()

	def win_state(self) -> np.ndarray:
		result = [False] * (NUM_PLAYERS + 1)
		player = self._player_range()

		if len(self._board.get_legal(1)) == 0 and len(self._board.get_legal(-1)) == 0:
			pieces_turn = self._board.get_total(self._player_range())
			pieces_other = self._board.get_total(-self._player_range())
			if pieces_turn > pieces_other:
				result[0] = True
			elif pieces_turn < pieces_other:
				result[1] = True
			else:
				result[2] = True
		return np.array(result, dtype=np.uint8)

	def observation(self) -> np.ndarray:
		if MULTI_PLANE_OBSERVATION:
			pieces = np.asarray(self._board.pieces)

			player1 = np.where(pieces == 1, 1, 0)
			player2 = np.where(pieces == -1, 1, 0)
			colour = np.full_like(pieces, self._player_range())
			turn = np.full_like(pieces, self.turns / MAX_TURNS, dtype=np.float32)
			return np.array([player1, player2, colour, turn], dtype=np.float32)
		else:
			return np.copy(np.expand_dims(np.asarray(self._board.pieces), axis=0))

	def symmetries(self, pi) -> List[Tuple[Any, int]]:
		#TODO
		#pass
		# # mirror, rotational
		assert (len(pi) == self._board.n ** 2 + 1)

		pi_board = np.reshape(pi[:-1], (self._board.n, self._board.n))
		result = []

		for i in range(1, 5):
			for j in [True, False]:
				new_b = np.rot90(np.asarray(self._board.pieces), i)
				new_pi = np.rot90(pi_board, i)
				if j:
					new_b = np.fliplr(new_b)
					new_pi = np.fliplr(new_pi)

				gs = self.clone()
				gs._board.pieces = new_b
				result.append((gs, np.array(list(new_pi.ravel()) + [pi[-1]]) ))

		return result

#Cython hates this for some reason
# def display(board):
# 	n = BOARD_SIZE

# 	for y in range(n):
# 		print(y, "|", end="")

# 	print("")
# 	print(" -----------------------")
# 	for y in range(n):
# 		print(y, "|", end="")		# print the row #
# 		for x in range(n):
# 			piece = board[y][x]		# get the piece to print
# 			if piece == -1:
# 				print("b ", end="")
# 			elif piece == 1:
# 				print("W ", end="")
# 			else:
# 				if x == n:
# 					print("-", end="")
# 				else:
# 					print("- ", end="")
# 		print("|")

# 	print("	 -----------------------")



