from typing import List, Tuple, Any
from alphazero.envs.U3T.U3TLogic import Board
from alphazero.Game import GameState

import numpy as np

BOARD_SIZE = 9
MULTI_PLANE_OBSERVATION = True
NUM_PLAYERS = 2
NUM_CHANNELS = 24 if MULTI_PLANE_OBSERVATION else 1
MAX_TURNS = 3*3 * 3*3
ACTION_SIZE = 81
OBSERVATION_SIZE = (NUM_CHANNELS, 3, 3)

class Game(GameState):
	def __init__(self, _board=None):
		super().__init__(_board or self._get_board())

	def __hash__(self) -> int:
		return hash(np.reshape(np.asarray(self._board.pieces), (9, 9)).tobytes() + bytes([self.turns]) + bytes([self._player]))

	def __eq__(self, other: 'Game') -> bool:
		return (
				np.asarray(self._board.pieces) == np.asarray(other._board.pieces)
				and self._player == other._player
				and self.turns == other.turns
		)

	def display(self):
		#display(self._board.pieces)
		BOARD = self.convert_to_2d()

		print(BOARD)

	@staticmethod
	def _get_board(*args, **kwargs):
		return Board(*args, **kwargs)

	def clone(self) -> 'Game':
		board = self._get_board(_pieces=np.copy(np.asarray(self._board.pieces)))
		game = Game(_board=board)
		game._player = self._player
		game._turns = self.turns
		game.last_action = self.last_action
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
		return (1, -1)[self.player]

	def valid_moves(self):
		valids = [0] * self.action_size()
		minor = (-1, -1, -1, -1)

		if self.last_action != None:
			minor = self._board.num_to_point(self.last_action)

		if self.last_action == None or len(self._board.get_valid_moves(minor[2], minor[3])) == 0:
			for i in self._board.get_all_open():
				valids[self._board.point_to_num(*i)] = 1
		else:
			for i in self._board.get_valid_moves(minor[2], minor[3]):
				valids[self._board.point_to_num(*i)] = 1

		# if not np.any(valids):
		# 	print(self._board.get_valid_moves(minor[2], minor[3]), self._board.get_all_open(), minor)
		# 	self.display()
		# 	print(np.asarray(self._board.pieces))
		return np.array(valids, dtype=np.intc)

	def play_action(self, action: int):
		super().play_action(action)
		minor = self._board.num_to_point(action)
		self._board.place_piece(*minor, self._player_range())
		#print(minor)
		self._update_turn()

	def win_state(self) -> np.ndarray:
		result = [False] * (NUM_PLAYERS + 1)
		if self._board.check_win(1) == 1:
			result[0] = True
		elif self._board.check_win(-1) == 1:
			result[1] = True
		elif len(self._board.get_all_open()) == 0:
			result[2] = True
		return np.array(result, dtype=np.uint8)

	def observation(self) -> np.ndarray:
		pieces = np.asarray(self._board.pieces) 
		BOARD = self.convert_to_2d()
		if MULTI_PLANE_OBSERVATION:
			player1 = np.where(BOARD == 1, 1, 0)
			player2 = np.where(BOARD == -1, 1, 0)
			major_wins = np.zeros((3, 3), dtype=np.float32)
			playable = np.zeros((3, 3), dtype=np.float32)
			for y in range(3):
				for x in range(3):
					WIN = 0
					if self._board.check_minor_win(1, y, x) == 1:
						WIN = 1
					if self._board.check_minor_win(-1, y, x) == 1:
						WIN = -1
					major_wins[y][x] = WIN
					if WIN == 0 and len(self._board.get_valid_moves(y, x)) == 0:
						playable[y][x] = -1
					elif WIN != 0:
						playable[y][x] = -1
					else:
						playable[y][x] = 1
						

			colour = np.full_like(major_wins, self._player_range())
			turn = np.full_like(major_wins, self.turns / MAX_TURNS, dtype=np.float32)
			return np.array([np.where(pieces[0][0] == 1, 1, 0), np.where(pieces[0][0] == -1, 1, 0), 
											np.where(pieces[0][1] == 1, 1, 0), np.where(pieces[0][1] == -1, 1, 0),
											np.where(pieces[0][2] == 1, 1, 0), np.where(pieces[0][2] == -1, 1, 0), 
											np.where(pieces[1][0] == 1, 1, 0), np.where(pieces[1][0] == -1, 1, 0), 
											np.where(pieces[1][1] == 1, 1, 0), np.where(pieces[1][1] == -1, 1, 0),
											np.where(pieces[1][2] == 1, 1, 0), np.where(pieces[1][2] == -1, 1, 0),
											np.where(pieces[2][0] == 1, 1, 0), np.where(pieces[2][0] == -1, 1, 0), 
											np.where(pieces[2][1] == 1, 1, 0), np.where(pieces[2][1] == -1, 1, 0),
											np.where(pieces[2][2] == 1, 1, 0), np.where(pieces[2][2] == -1, 1, 0),
											np.where(major_wins == 1, 1, 0), np.where(major_wins == -1, 1, 0), 
											np.where(playable == 1, 1, 0), np.where(playable == -1, 1, 0), colour, turn], dtype=np.float32)
		else:
			return np.copy(np.expand_dims(np.asarray(BOARD), axis=0)) 

	def convert_to_2d(self, arr=None):
		BOARD = []
		pieces = np.asarray(self._board.pieces)
		if arr is not None:
			pieces = arr
		for y in range(3):
			for y_ in range(3):
				temp = []
				for x in range(3):
					for x_ in range(3):
						temp.append(pieces[y][x][y_][x_])
				BOARD.append(temp)

		return np.array(BOARD, dtype=np.float32)

	def convert_to_4d(self, arr):
		output = np.zeros((3, 3, 3, 3), dtype=np.intc)
		for y in range(9):
			for x in range(9):
				n = 9*y + x
				minor = self._board.num_to_point(n)
				output[minor[0]][minor[1]][minor[2]][minor[3]] = arr[y][x]
		return output

	def symmetries(self, pi) -> List[Tuple[Any, int]]:
		A = np.argmax(pi)
		TWO = self.convert_to_2d()
		minor = self._board.num_to_point(A)
		#Reflection over the y axis
		Y_ref = TWO[:, ::-1]
		Y_ref_board = self.convert_to_4d(Y_ref)
		Y_ref_game = self.clone()
		Y_ref_game._board.pieces = Y_ref_board
		Y_ref_policy = np.copy(pi)
		Y_ref_2d = np.reshape(Y_ref_policy, (9, 9))
		Y_ref_2d = Y_ref_2d[:, ::-1]
		Y_ref_policy = Y_ref_2d.flatten()
		#Reflection over x axis
		X_ref = TWO[::-1, :]
		X_ref_board = self.convert_to_4d(X_ref)
		X_ref_game = self.clone()
		X_ref_game._board.pieces = X_ref_board
		X_ref_policy = np.copy(pi)
		X_ref_2d = np.reshape(X_ref_policy, (9, 9))
		X_ref_2d = X_ref_2d[::-1, :]
		X_ref_policy = X_ref_2d.flatten()
		#Rotate 90
		Ninty = np.rot90(TWO)
		Ninty_board = self.convert_to_4d(Ninty)
		Ninty_game = self.clone()
		Ninty_game._board.pieces = Ninty_board
		Ninty_policy = np.copy(pi)
		TEMP = np.reshape(Ninty_policy, (9, 9))
		TEMP = np.rot90(TEMP)
		Ninty_policy = TEMP.flatten()
		#Rotate 270
		Ninty_2 = np.rot90(TWO, k=3)
		Ninty_2_board = self.convert_to_4d(Ninty_2)
		Ninty_2_game = self.clone()
		Ninty_2_game._board.pieces = Ninty_2_board
		Ninty_2_policy = np.copy(pi)
		TEMP = np.reshape(Ninty_2_policy, (9, 9))
		TEMP = np.rot90(TEMP, k=3)
		Ninty_2_policy = TEMP.flatten()
		return [(self.clone(), pi), 
		(Y_ref_game, Y_ref_policy), 
		(X_ref_game, X_ref_policy),
		(Ninty_game, Ninty_policy),
		(Ninty_2_game, Ninty_2_policy)]


