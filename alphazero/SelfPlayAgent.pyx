# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools

from alphazero.MCTS import MCTS


def get_game_results(result_queue, game, get_index=None):
    num_games = result_queue.qsize()
    wins = [0] * len(game.getPlayers())
    draws = 0
    for _ in range(num_games):
        player, result, board, agent_id = result_queue.get()
        if result == 1:
            index = get_index(player, agent_id) if get_index else player
            wins[index] += 1
        elif result == -1:
            for p in game.getPlayers():
                if game.getGameEnded(board, p) == 1:
                    index = get_index(p, agent_id) if get_index else p
                    wins[index] += 1
                    break
        else:
            draws += 1

    return wins, draws


def _translate(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


"""
def memory_usage(vars):
    def sizeof_fmt(num, suffix='B'):
        '''by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, asizeof.asizeof(value)) for name, value in vars.items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
"""


class SelfPlayAgent(mp.Process):
    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, args, _is_arena=False, _is_warmup=False, _player_order=None):
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        if _is_arena:
            self.batch_size = policy_tensor.shape[0]
        else:
            self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []
        self.canonical = []
        self.histories = []
        self.player = []
        self.turn = []
        self.next_reset = []
        self.mcts = []
        self.games_played = games_played
        self.complete_count = complete_count
        self.stop_event = stop_event
        self.args = args

        self._is_arena = _is_arena
        self._is_warmup = _is_warmup
        if _is_arena:
            self.player_to_index = _player_order
            self.batch_indices = None

        self.fast = False
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.player.append(self.game.getPlayers()[0])
            self.turn.append(1)
            self.next_reset.append(0)
            self.mcts.append(MCTS(self.game, None, self.args))
            self.canonical.append(None)

    def run(self):
        try:
            np.random.seed()
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration:
                self.generateCanonical()
                self.fast = np.random.random_sample() < self.args.probFastSim
                sims = self.args.numFastSims if self.fast else self.args.numMCTSSims \
                    if not self._is_warmup else self.args.numWarmupSims
                for i in range(sims):
                    self.generateBatch()
                    self.processBatch()
                self.playMoves()

            with self.complete_count.get_lock():
                self.complete_count.value += 1

            del self.games
            del self.canonical
            del self.histories
            del self.player
            del self.turn
            del self.next_reset
            del self.mcts

            if not self._is_arena:
                self.output_queue.close()
                self.output_queue.join_thread()
        except Exception:
            print(traceback.format_exc())

    def generateBatch(self):
        if self._is_arena:
            batch_tensor = [[] for _ in self.game.getPlayers()]
            self.batch_indices = [[] for _ in self.game.getPlayers()]

        for i in range(self.batch_size):
            board = self.mcts[i].findLeafToProcess(self.canonical[i], True)
            if self._is_warmup:
                if board is not None:
                    policy = self.game.getValidMoves(board, self.game.getPlayers()[0])
                    policy = policy / np.sum(policy)
                    self.policy_tensor[i] = torch.from_numpy(policy)
                    self.value_tensor[i] = np.random.uniform(-1, 1)
                else:
                    self.policy_tensor[i] = torch.zeros(self.game.getActionSize())
                    self.value_tensor[i] = 0.
                continue

            if board is not None:
                data = torch.from_numpy(board.astype(np.float32))
                if self._is_arena:
                    data = data.view(-1, *self.game.getObservationSize())
                    player = self.player_to_index[self.player[i]]
                    batch_tensor[player].append(data)
                    self.batch_indices[player].append(i)
                else:
                    self.batch_tensor[i].copy_(data)
            elif self._is_arena:
                batch_tensor[0].append(torch.zeros(1, *self.game.getObservationSize()))
                self.batch_indices[0].append(i)

        if self._is_arena:
            for player in self.game.getPlayers():
                player = self.player_to_index[player]
                data = batch_tensor[player]
                if data:
                    batch_tensor[player] = torch.cat(data)
            self.output_queue.put(batch_tensor)
            self.batch_indices = list(itertools.chain.from_iterable(self.batch_indices))

        if not self._is_warmup:
            self.ready_queue.put(self.id)

    def processBatch(self):
        if not self._is_warmup:
            self.batch_ready.wait()
            self.batch_ready.clear()
        for i in range(self.batch_size):
            index = self.batch_indices[i] if self._is_arena else i
            self.mcts[index].processResults(
                self.policy_tensor[i].data.numpy(), self.value_tensor[i][0]
            )

    def playMoves(self):
        for i in range(self.batch_size):
            temp = int(self.turn[i] < self.args.tempThreshold)
            policy = self.mcts[i].getExpertProb(self.canonical[i], temp, not self.fast)
            action = np.random.choice(self.game.getActionSize(), p=policy)
            if not self.fast and not self._is_arena:
                self.histories[i].append((
                    self.canonical[i],
                    self.mcts[i].getExpertProb(self.canonical[i], prune=True),
                    self.mcts[i].getExpertValue(self.canonical[i]),
                    self.player[i]
                ))

            self.games[i], self.player[i] = self.game.getNextState(self.games[i], self.player[i], action, copy=False)
            self.turn[i] += 1
            if self.args.mctsResetThreshold and self.turn[i] >= self.next_reset[i]:
                self.mcts[i].reset()
                self.next_reset[i] = self.turn[i] + self.args.mctsResetThreshold

            result = self.game.getGameEnded(self.games[i], self.player[i])
            if result != 0:
                self.result_queue.put((self.player[i], result, self.games[i], self.id))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            if self.args.symmetricSamples:
                                sym = self.game.getSymmetries(hist[0], hist[1])
                                for b, p in sym:
                                    self.output_queue.put((
                                        b, p,
                                        result * _translate(
                                            self.player[i], self.game.getPlayers()[0], self.game.getPlayers()[-1], 1, -1
                                        ) * (1 - self.args.expertValueWeight.current)
                                        + self.args.expertValueWeight.current * hist[2]
                                    ))
                            else:
                                self.output_queue.put((
                                    hist[0], hist[1],
                                    result * _translate(
                                        self.player[i], self.game.getPlayers()[0], self.game.getPlayers()[-1], 1, -1
                                    ) * (1 - self.args.expertValueWeight.current)
                                    + self.args.expertValueWeight.current * hist[2]
                                ))
                    self.games[i] = self.game.getInitBoard()
                    self.histories[i] = []
                    self.player[i] = self.game.getPlayers()[0]
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.game, None, self.args)
                else:
                    lock.release()

    def generateCanonical(self):
        for i in range(self.batch_size):
            self.canonical[i] = self.game.getCanonicalForm(
                self.games[i], self.player[i]
            )
