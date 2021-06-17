# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools

from alphazero.MCTS import MCTS


def get_game_results(result_queue, game_cls, _get_index=None):
    player_to_index = {p: i for i, p in enumerate(game_cls.get_players())}

    num_games = result_queue.qsize()
    wins = [0] * len(game_cls.get_players())
    draws = 0
    for _ in range(num_games):
        _, value, agent_id = result_queue.get()
        if value != 0:
            index = _get_index(value, agent_id) if _get_index else player_to_index[value]
            wins[index] += 1
        else:
            draws += 1

    return wins, draws


class SelfPlayAgent(mp.Process):
    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, args, _is_arena=False, _is_warmup=False, _player_order=None):
        super().__init__()
        self.id = id
        self.game_cls = type(game)
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
        self.histories = []
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
            self.games.append(self.game_cls())
            self.histories.append([])
            self.turn.append(1)
            self.next_reset.append(0)
            self.mcts.append(MCTS(self.args.cpuct))

    def run(self):
        try:
            np.random.seed()
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration:
                self.fast = np.random.random_sample() < self.args.probFastSim
                sims = self.args.numFastSims if self.fast else self.args.numMCTSSims \
                    if not self._is_warmup else self.args.numWarmupSims
                for i in range(sims):
                    self.generateBatch()
                    self.processBatch()
                self.playMoves()

            with self.complete_count.get_lock():
                self.complete_count.value += 1
            if not self._is_arena:
                self.output_queue.close()
                self.output_queue.join_thread()
        except Exception:
            print(traceback.format_exc())

    def generateBatch(self):
        if self._is_arena:
            batch_tensor = [[] for _ in self.game_cls.get_players()]
            self.batch_indices = [[] for _ in self.game_cls.get_players()]

        for i in range(self.batch_size):
            state = self.mcts[i].find_leaf(self.games[i])
            if self._is_warmup:
                policy = state.valid_moves()
                policy = policy / np.sum(policy)
                self.policy_tensor[i] = torch.from_numpy(policy)
                self.value_tensor[i] = np.random.uniform(-1, 1)
                continue

            data = torch.from_numpy(state.observation())
            if self._is_arena:
                data = data.view(-1, *state.observation_size())
                player = self.player_to_index[self.games[i].current_player()]
                batch_tensor[player].append(data)
                self.batch_indices[player].append(i)
            else:
                self.batch_tensor[i].copy_(data)

        if self._is_arena:
            for player in self.game_cls.get_players():
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
            self.mcts[index].process_results(
                self.games[i], self.value_tensor[i][0], self.policy_tensor[i].data.numpy()
            )

    def playMoves(self):
        for i in range(self.batch_size):
            temp = int(self.turn[i] < self.args.tempThreshold)
            policy = self.mcts[i].probs(self.games[i], temp)
            action = np.random.choice(self.games[i].action_size(), p=policy)
            if not self.fast and not self._is_arena:
                self.histories[i].append((
                    self.games[i].observation(),
                    self.mcts[i].probs(self.games[i]),
                    self.mcts[i].value()
                ))

            self.mcts[i].update_root(self.games[i], action)
            self.games[i].play_action(action)
            self.turn[i] += 1
            if self.args.mctsResetThreshold and self.turn[i] >= self.next_reset[i]:
                self.mcts[i] = MCTS(self.args.cpuct)
                self.next_reset[i] = self.turn[i] + self.args.mctsResetThreshold

            game_over, value = self.games[i].win_state()
            if game_over:
                self.result_queue.put((self.games[i], value, self.id))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            if self.args.symmetricSamples:
                                data = self.games[i].symmetries(hist[1])
                            else:
                                data = (hist[0], hist[1])
                            for obs, pi in data:
                                self.output_queue.put((
                                    obs, pi,
                                    value * (1 - self.args.expertValueWeight.current)
                                    + self.args.expertValueWeight.current * hist[2]
                                ))
                    self.games[i] = self.game_cls()
                    self.histories[i] = []
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.args.cpuct)
                else:
                    lock.release()
