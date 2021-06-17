# cython: language_level=3

from alphazero.Game import Game
from alphazero.GenericPlayers import BasePlayer
from alphazero.SelfPlayAgent import SelfPlayAgent, get_game_results
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.utils import dotdict

from typing import Callable, List, Tuple, Any
from queue import Empty
from time import time

import torch.multiprocessing as mp
import torch
import random


class _PlayerWrapper:
    def __init__(self, player_func, index):
        self.player_func = player_func
        self.index = index
        self.wins = 0
        self.winrate = 0

    def __call__(self, *args, **kwargs):
        return self.player_func(*args, **kwargs)

    def reset(self):
        self.wins = 0
        self.winrate = 0

    def add_win(self):
        self.wins += 1

    def update_winrate(self, draws, num_games):
        if not num_games:
            self.winrate = 0
        else:
            self.winrate = (self.wins + 0.5 * draws) / num_games


class Arena:
    """
    An Arena class where any game's agents can be pit against each other.
    """

    def __init__(
            self,
            players: List[Callable],
            game: Game,
            use_batched_mcts=True,
            display: Callable = None,
            args: dotdict = None
    ):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        if len(players) != len(game.getPlayers()):
            raise ValueError('Argument `players` must have the same amount of players as the game supports. '
                             f'Got {len(players)} player agents, while the game requires {len(self.game.getPlayers())}')

        self.__players = None
        self.players = players
        self.game = game
        self.use_batched_mcts = use_batched_mcts
        self.display = display
        self.args = args.copy()
        self.draws = 0
        self.winrates = []

    @property
    def players(self):
        return self.__players

    @players.setter
    def players(self, value):
        self.__players = value
        self.__init_players()

    def __init_players(self):
        new_players = []
        for i, player in enumerate(self.__players):
            if not isinstance(player, _PlayerWrapper):
                player = _PlayerWrapper(player, i)
            new_players.append(player)
        self.__players = new_players

    def __reset_counts(self):
        self.draws = 0
        self.winrates = []
        [player.reset() for player in self.players]

    def __update_winrates(self, num_games):
        [player.update_winrate(self.draws, num_games) for player in self.players]
        self.winrates = [player.winrate for player in sorted(self.players, key=lambda p: p.index)]

    def play_game(self, verbose=False, player_to_index: list = None) -> Tuple[int, int, Any]:
        """
        Executes one episode of a game.

        Returns:
            result: the result of the game (based on last canonical board)
            cur_player: the last player to play in the game
            board: the last canonical state/board
        """
        start_player = self.game.getPlayers()[0]
        next_player = start_player
        to_play = start_player
        board = self.game.getInitBoard()
        turns = 0

        while True:
            index = to_play if not player_to_index else player_to_index[to_play]
            action = self.players[index](board, turns)
            valids = self.game.getValidMoves(board, start_player)

            if valids[action] == 0:
                print()
                print(action, index, next_player, to_play, turns)
                print(valids)
                print()
                assert valids[action] > 0

            board, next_player = self.game.getNextState(board, start_player, action)
            result = self.game.getGameEnded(board, start_player)

            if result != 0:
                if verbose:
                    assert self.display
                    print("Game over: Turn ", str(turns), "Result ", str(self.game.getGameEnded(board, start_player)))
                    self.display(board, to_play)
                return self.game.getGameEnded(board, start_player), to_play, board
            
            if verbose:
                assert self.display
                print("Turn ", str(turns), "Player ", str(to_play))
                self.display(board, to_play)

            board = self.game.getCanonicalForm(board, next_player)
            to_play = self.game.getNextPlayer(to_play)
            turns += 1

    def play_games(self, num, verbose=False) -> Tuple[List[int], int, List[int]]:
        """
        Plays num games in which the order of the players
        is randomized for each game. The order is simply switched
        if there are only two players.

        Returns:
            wins: number of wins for each player in self.players
            draws: number of draws that occurred in total
            winrates: the win rates for each player in self.players
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.play_games', max=num)
        end = time()
        self.__reset_counts()

        if self.use_batched_mcts:
            self.args.gamesPerIteration = num
            agents = []
            policy_tensors = []
            value_tensors = []
            batch_ready = []
            batch_queues = []
            stop_agents = mp.Event()
            ready_queue = mp.Queue()
            result_queue = mp.Queue()
            completed = mp.Value('i', 0)
            games_played = mp.Value('i', 0)
            player_to_index = self.game.getPlayers().copy()

            self.args.expertValueWeight.current = self.args.expertValueWeight.start
            # if self.args.workers >= mp.cpu_count():
            #    self.args.workers = mp.cpu_count() - 1

            def get_player_order():
                if len(player_to_index) == 2:
                    player_to_index.reverse()
                else:
                    random.shuffle(player_to_index)

            for i in range(self.args.workers):
                input_tensors = [[] for _ in player_to_index]
                batch_queues.append(mp.Queue())

                policy_tensors.append(torch.zeros(
                    [self.args.arena_batch_size, self.game.getActionSize()]
                ))
                policy_tensors[i].pin_memory()
                policy_tensors[i].share_memory_()

                value_tensors.append(torch.zeros([self.args.arena_batch_size, 1]))
                value_tensors[i].pin_memory()
                value_tensors[i].share_memory_()

                batch_ready.append(mp.Event())
                get_player_order()

                agents.append(
                    SelfPlayAgent(i, self.game, ready_queue, batch_ready[i],
                                  input_tensors, policy_tensors[i], value_tensors[i], batch_queues[i],
                                  result_queue, completed, games_played, stop_agents, self.args,
                                  _is_arena=True, _player_order=player_to_index.copy()))
                agents[i].daemon = True
                agents[i].start()

            sample_time = AverageMeter()
            end = time()

            n = 0
            while completed.value != self.args.workers:
                try:
                    id = ready_queue.get(timeout=1)

                    policy = []
                    value = []
                    data = batch_queues[id].get()
                    for player in self.game.getPlayers():
                        batch = data[player]
                        if isinstance(batch, torch.Tensor):
                            p, v = self.players[player](batch)
                            policy.append(p)
                            value.append(v)

                    policy_tensors[id].copy_(torch.cat(policy))
                    value_tensors[id].copy_(torch.cat(value))
                    batch_ready[id].set()
                except Empty:
                    pass

                size = games_played.value
                if size > n:
                    sample_time.update((time() - end) / (size - n), size - n)
                    n = size
                    end = time()

                wins, draws = get_game_results(
                    result_queue,
                    self.game,
                    get_index=lambda p, i: agents[i].player_to_index[p]
                )
                for i, w in enumerate(wins):
                    self.players[i].wins += w
                self.draws += draws
                self.__update_winrates(sum([player.wins for player in self.players]) + self.draws)

                bar.suffix = '({eps}/{maxeps}) Winrates: {wr} | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}' \
                    .format(
                        eps=size, maxeps=num, et=sample_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                        wr=[round(w, 2) for w in self.winrates]
                    )
                bar.goto(size)

            stop_agents.set()
            bar.update()
            bar.finish()

            for agent in agents:
                agent.join()
                del policy_tensors[0]
                del value_tensors[0]
                del batch_ready[0]

        else:
            player_to_index = self.game.getPlayers().copy()
            
            def update_players():
                # Change up the order of the players for even game
                if len(player_to_index) == 2:
                    player_to_index.reverse()
                else:
                    random.shuffle(player_to_index)

            for eps in range(1, num + 1):
                # Get a new lookup for self.players, randomized or reversed from original
                update_players()

                # Play a single game with the current player order
                result, player, board = self.play_game(verbose, player_to_index)

                # Bookkeeping + plot progress
                if result == 1:
                    self.players[player_to_index[player]].add_win()
                elif result == -1:
                    p = player
                    for _ in range(len(player_to_index)-1):
                        p = self.game.getNextPlayer(p)
                        b = self.game.getCanonicalForm(board, player - p)
                        if self.game.getGameEnded(b, 0) == 1:
                            self.players[player_to_index[p]].add_win()
                            break
                    else:
                        print(board.current_player)
                        raise RuntimeError(
                            f'Winner not found after game. Result={result}, player={player}, board:\n{board}'
                        )
                else:
                    self.draws += 1

                self.__update_winrates(eps)
                eps_time.update(time() - end)
                end = time()
                bar.suffix = '({eps}/{maxeps}) Winrates: {wr} | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}' \
                    .format(
                        eps=eps, maxeps=num, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                        wr=[round(w, 2) for w in self.winrates]
                    )
                bar.next()

            bar.update()
            bar.finish()

        wins = [player.wins for player in sorted(self.players, key=lambda p: p.index)]

        return wins, self.draws, self.winrates
