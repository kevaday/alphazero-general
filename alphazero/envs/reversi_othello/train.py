import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.reversi_othello.othello import Game
from alphazero.utils import dotdict

#from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


args = get_args(dotdict({
    'run_name': 'reversi_othello',
    'workers': 7,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 256,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 256 * 7,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 500,
    'numFastSims': 100,
    'probFastSim': 0.55,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 10,
    'arenaCompare': 32,
    'arena_batch_size': 128,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 3,
    'compareWithPast': True,
    'pastCompareFreq': 10,
    'cpuct': 1.75,
    'fpu_reduction': 0.4,
    'load_model': True,
    'root_noise_frac': 0.3,
    'min_next_model_winrate': 0.54,
    'root_policy_temp': 1.3,
    'train_on_past_data': False,
    'past_data_chunk_size': 25,
    'past_data_run_name': 'reversi_othello',
    "_num_players":2,
    'use_draws_for_winrate': False,

}),
    model_gating=False,
    max_gating_iters=100,
    max_moves=64 + 36, #Extra to account for pass

    lr=0.02,
    num_channels=128,
    depth=10,
    value_head_channels=64,
    policy_head_channels=64,
    value_dense_layers=[2048, 1024, 512],
    policy_dense_layers=[2048, 512]
    )
args.scheduler_args.milestones = [150, 225]



if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()


#Start 2:45
#End 6:32 7:07