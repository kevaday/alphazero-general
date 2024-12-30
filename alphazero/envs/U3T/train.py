import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.U3T.U3T import Game
from alphazero.utils import dotdict

#from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


args = get_args(dotdict({
    'run_name': 'U3T',
    'workers': 7,
    'startIter': 0,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 256,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 4,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 500,
    'numFastSims': 100,
    'probFastSim': 0.65,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 10,
    'arenaCompare': 32,
    'arena_batch_size': 512,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 3,
    'compareWithPast': False,
    'pastCompareFreq': 10,
    'cpuct': 1.25,
    'fpu_reduction': 0.4,
    'load_model': True,
    'root_noise_frac': 0.3,
    'min_next_model_winrate': 0.54,
    'root_policy_temp': 1.3,
    'train_on_past_data': False,
    'past_data_chunk_size': 25,
    'past_data_run_name': 'U3T',
    'use_draws_for_winrate': False,
    "_num_players":2

}),
    model_gating=False,
    max_gating_iters=100,
    max_moves=81,

    lr=0.02,
    num_channels=128,
    depth=10,
    value_head_channels=64,
    policy_head_channels=64,
    value_dense_layers=[2048, 1024, 512],
    policy_dense_layers=[2048, 1024, 512]
    )
args.scheduler_args.milestones = [150, 225]



if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    #c.learn()
    data = []
    with open("elo/U3T/ELOS.csv",'r') as data_file:
        for line in data_file:
            data = line.split(",")
    for i in range(len(data)):
        c.writer.add_scalar('elo/self_play_elo_4', float(data[i]), i)


#Start 2:45
#End 6:32 7:07