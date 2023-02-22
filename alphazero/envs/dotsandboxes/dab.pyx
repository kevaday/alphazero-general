from typing import List, Tuple, Any

from alphazero.envs.dotsandboxes.dab_logic import Board
from alphazero.Game import GameState

import numpy as np
import math
MULTI_PLANE_OBSERVATION = True
NUM_PLAYERS = 2
NUM_CHANNELS = 8 if MULTI_PLANE_OBSERVATION else 1
#Must be odd numbers
BOARD_HEIGHT = 7
BOARD_WIDTH = 7

ACTION_SIZE = math.floor(BOARD_WIDTH * BOARD_HEIGHT / 2) + 1
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
MAX_TURNS = ACTION_SIZE * 2

class Game(GameState):
	def __init__(self, _board=None):
		super().__init__(_board or self._get_board())
		self.recent_box = 0
		self.p1_boxes = 0
		self.p2_boxes = 0

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
		for y in range(BOARD_HEIGHT):
			for x in range(BOARD_WIDTH):
				s = " o " if self._board.pieces[y][x] == 1 else " x "
				if self._board.pieces[y][x] == 0:
					if y % 2 == 0:
						s = "---"
					else:
						s = " | "
				if self._board.pieces[y][x] == 3:
					s = " o "
				if self._board.pieces[y][x] == -3:
					s = " x "

				print " . " if self._board.pieces[y][x] == 2 else s,					

			print "\n"
			# for x in range(BOARD_WIDTH):
			# 	print " | ",
			# print "\n"
		print("\n")
	@staticmethod
	def _get_board(*args, **kwargs):
		return Board(BOARD_HEIGHT, BOARD_WIDTH, *args, **kwargs)

	def clone(self) -> 'Game':
		board = self._get_board(_pieces=np.copy(np.asarray(self._board.pieces)))
		game = Game(_board=board)
		game._player = self._player
		game._turns = self.turns
		game.recent_box = self.recent_box
		game.p1_boxes = self.p1_boxes
		game.p2_boxes = self.p2_boxes
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
		moves = self._board.get_valids()

		if self.recent_box != 0 and self.recent_box != self._player_range():
			valids[-1] = 1
			return np.array(valids, dtype=np.intc)

		for y, x in moves:
			valids[self._board.map_to_num((y, x))] = 1
		return np.array(valids, dtype=np.intc)

	def play_action(self, action: int):
		super().play_action(action)
		#Pass if no moves are possible
		if action == self.action_size()-1:
			self._update_turn()
			return
		move = self._board.reverse_map(action)
		old = self._board.check_boxes()
		self._board.add_piece(move[0], move[1], self._player_range())
		if self._board.check_boxes() > old:
			self.recent_box = self._player_range()
			if self._player_range() == 1:
				self.p1_boxes += 1
			else:
				self.p2_boxes += 1
		else:
			self.recent_box = 0
		self._update_turn()

	def win_state(self) -> np.ndarray:
		result = [False] * (NUM_PLAYERS + 1)
		player = self._player_range()

		if len(self._board.get_valids()) == 0:
			if self.p1_boxes > self.p2_boxes:
				result[0] = True
			elif self.p2_boxes > self.p1_boxes:
				result[1] = True
			else:
				result[2] = True
		return np.array(result, dtype=np.uint8)

	def observation(self) -> np.ndarray:
		if MULTI_PLANE_OBSERVATION:
			pieces = np.asarray(self._board.pieces)
			pieces = np.where(pieces == 2, 0, pieces)
			player1 = np.where(pieces == 1, 1, 0)
			player2 = np.where(pieces == -1, 1, 0)
			colour = np.full_like(pieces, self._player_range())
			p1_real_boxes = np.where(pieces == 3, 1, 0)
			p2_real_boxes = np.where(pieces == -3, 1, 0)
			p1_b = np.full_like(pieces, self.p1_boxes / self._board.possible_boxes, dtype=np.float32)
			p2_b = np.full_like(pieces, self.p2_boxes / self._board.possible_boxes, dtype=np.float32)
			turn = np.full_like(pieces, self.turns / MAX_TURNS, dtype=np.float32)
			return np.array([player1, player2, p1_real_boxes, p2_real_boxes, colour, p1_b, p2_b, turn], dtype=np.float32)
		else:
			return np.copy(np.expand_dims(np.asarray(self._board.pieces), axis=0))

	def symmetries(self, pi) -> List[Tuple[Any, int]]:
		#raise NotImplementedError
		assert len(pi) == self.action_size()

		pi_2d = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
		cdef Py_ssize_t x, y, i
		for i in range(len(pi)-1):
			p = self._board.reverse_map(i)
			pi_2d[p[0]][p[1]] = pi[i]

		pi_ref = np.zeros(self.action_size(), dtype=np.float32)
		pi_ref_2d = np.fliplr(pi_2d)
		for y in range(BOARD_HEIGHT):
			for x in range(BOARD_WIDTH):
				if (x+y) % 2 != 0:
					pi_ref[self._board.map_to_num((y, x))] = pi_ref_2d[y][x]
		b2_ref = np.copy(self._board.pieces)
		b2_ref = np.fliplr(b2_ref)
		ref_game = self.clone()
		ref_game._board.pieces = np.copy(b2_ref)

		pi_ref_2 = np.zeros(self.action_size(), dtype=np.float32)
		pi_ref_2_2d = np.flipud(pi_2d)
		for y in range(BOARD_HEIGHT):
			for x in range(BOARD_WIDTH):
				if (x+y) % 2 != 0:
					pi_ref_2[self._board.map_to_num((y, x))] = pi_ref_2_2d[y][x]
		b2_ref_2 = np.copy(self._board.pieces)
		b2_ref_2 = np.flipud(b2_ref_2)
		ref_game_2 = self.clone()
		ref_game_2._board.pieces = np.copy(b2_ref_2)

		return [(self.clone(), pi), (ref_game, pi_ref), (ref_game_2, pi_ref_2)]
