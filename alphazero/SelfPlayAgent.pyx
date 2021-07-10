# cython: language_level=3
# cython: profile=True

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools

from alphazero.MCTS import MCTS


class SelfPlayAgent(mp.Process):
    def __init__(self, id, game_cls, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, args, _is_arena=False, _is_warmup=False, _player_order=None):
        super().__init__()
        self.id = id
        self.game_cls = game_cls
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
        self.temps = []
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
        if _is_warmup:
            self._WARMUP_POLICY = torch.full((game_cls.action_size(),), 1 / game_cls.action_size())
            self._WARMUP_VALUE = torch.zeros(len(game_cls.get_players()) + 1)

        self.fast = False
        for _ in range(self.batch_size):
            self.games.append(self.game_cls())
            self.histories.append([])
            self.temps.append(self.args.startTemp)
            self.next_reset.append(0)
            self.mcts.append(self._get_mcts())

    def _get_mcts(self):
        def mcts():
            return MCTS(
                len(self.game_cls.get_players()),
                self.args.cpuct,
                self.args.root_noise_frac,
                self.args.root_policy_temp
            )

        if self._is_arena:
            return tuple([mcts() for _ in range(len(self.game_cls.get_players()))])
        else:
            return mcts()

    def _mcts(self, index: int) -> MCTS:
        mcts = self.mcts[index]
        if self._is_arena:
            return mcts[self.games[index].current_player()]
        else:
            return mcts

    def run(self):
        try:
            np.random.seed()
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration:
                self.fast = np.random.random_sample() < self.args.probFastSim
                sims = self.args.numFastSims if self.fast else self.args.numMCTSSims \
                    if not self._is_warmup else self.args.numWarmupSims
                for _ in range(sims):
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
            state = self._mcts(i).find_leaf(self.games[i])
            if self._is_warmup:
                self.policy_tensor[i].copy_(self._WARMUP_POLICY)
                self.value_tensor[i].copy_(self._WARMUP_VALUE)
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
            self._mcts(i).process_results(
                self.games[i],
                self.value_tensor[index].data.numpy(),
                self.policy_tensor[index].data.numpy(),
                False if self._is_arena else self.args.add_root_noise
            )

    def playMoves(self):
        for i in range(self.batch_size):
            self.temps[i] = self.args.temp_scaling_fn(self.temps[i], self.games[i].turns, self.args.max_moves)
            policy = self._mcts(i).probs(self.games[i], self.temps[i])
            action = np.random.choice(self.games[i].action_size(), p=policy)
            if not self.fast and not self._is_arena:
                self.histories[i].append((
                    self.games[i].clone(),
                    self._mcts(i).probs(self.games[i])
                ))

            if self._is_arena:
                [mcts.update_root(self.games[i], action) for mcts in self.mcts[i]]
            else:
                self._mcts(i).update_root(self.games[i], action)
            self.games[i].play_action(action)
            if self.args.mctsResetThreshold and self.games[i].turns >= self.next_reset[i]:
                self.mcts[i] = self._get_mcts()
                self.next_reset[i] = self.games[i].turns + self.args.mctsResetThreshold

            winstate = self.games[i].win_state()
            if any(winstate):
                self.result_queue.put((self.games[i].clone(), winstate, self.id))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            if self.args.symmetricSamples:
                                data = hist[0].symmetries(hist[1])
                            else:
                                data = ((hist[0], hist[1]),)

                            for state, pi in data:
                                self.output_queue.put((
                                    state.observation(), pi, np.array(winstate, dtype=np.float32)
                                ))
                    self.games[i] = self.game_cls()
                    self.histories[i] = []
                    self.temps[i] = self.args.startTemp
                    self.mcts[i] = self._get_mcts()
                else:
                    lock.release()
