import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tafl.tafl import TaflGame as Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.envs.tafl.players import GreedyTaflPlayer

from hnefatafl.engine import variants


args = get_args(
    run_name='hnefatafl',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    tempThreshold=int(DRAW_MOVE_COUNT*0.8),
    cpuct=2,
    numWarmupIters=1,
    baselineCompareFreq=3,
    pastCompareFreq=3,
    baselineTester=GreedyTaflPlayer,
    process_batch_size=512,
    train_batch_size=4096,
    arena_batch_size=64,
    train_steps_per_iteration=256,
    gamesPerIteration=2048,
    lr=0.01,
    num_channels=128,
    depth=16,
    value_head_channels=1,
    policy_head_channels=2,
    value_dense_layers=[64, 32],
    policy_dense_layers=[1024]
)


if __name__ == "__main__":
    g = Game(variants.hnefatafl, max_moves=args.max_moves, num_stacked_obs=args.num_stacked_observations)
    nnet = nn(g, args)
    # nnet.save_checkpoint(args.checkpoint, f'iteration-0000.pkl')
    c = Coach(g, nnet, args)
    c.learn()
