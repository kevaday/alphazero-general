# AlphaZero General
This is an implementation of AlphaZero based on the following repositories:
The original repo: https://github.com/suragnair/alpha-zero-general
A fork of the original repo: https://github.com/bhansconnect/fast-alphazero-general

This repo is still work-in-progress, so expect frequent updates and much more detailed documentation soon.

### Current differences from the above repos:
1. Model Gating: after each iteration, the model is compared to the previous iteration. The model that performs better continues forward based on an adjustable parameter.
2. Batched MCTS: bhandsconnect's repo already includes this for self play, but it has been expanded upon to be included in Arena for faster comparison of models.
3. Multiplayer Support: Any number of players are supported! This allows for training on a greater variety of games such as many types of card games or something like Catan.
4. Warmup Iterations: A few self play iterations in the beginning of training are performed using random policy and value to speed up initial generation of training data.
5. Generally cleaner, more functional and universal code so far.

## Getting Started
### Install required packages
Make sure you have Python 3 installed. Then run:
```pip3 install -r requirements.txt```

### Try one of the existing examples:
1. Adjust the hyperparameters in one of the examples to your liking (path is ```alphazero/<game name>/train.py```). Take a look at Coach.py where the default arguments are stored to see the available choices you have. For example, edit ```alphazero/connect4/train.py```.

**Note: not all example games have the `train.py` file yet, I will add them soon. For now, you can copy the train file from another game and change it accordingly.**

2. After that, you can start training your AlphaZero on your chosen game by running the following: ```python3 -m alphazero.<game name>.train```. Make sure that your working directory is the root of the repo.
3. You can observe how training is progressing from the console output, or you can also run tensorboard for a visual representation. To start tensorboard, run ```tensorboard --logdir ./runs```, also from the project root. `runs` is the default directory for tensorboard data, but it can be changed in the hyperparameters `args`.
4. Once you have trained a model and want to test it, either against itself or you, change ```alphazero/pit.py``` to your needs and run it (once again, these things will be easier to do when I have the time to create proper documentation and make some tools more useable).

### Create your own game to train on:
Again, more detailed documentation is on the way, but in a nutshell, you must subclass `Game` from `alphazero/Game.py` and implement its abstract methods correctly. If this is done, just create a `train` file and choose hyperparameters accordingly and start training. You can also look at the original repos for more details about how it's done. Also, feel free to use and subclass the `boardgame` module I created to create a new game engine more easily as it implements some functions that may be useful to your case.

In order to increase performance, you can save your game engine files which may be a bottleneck with the extension `.pyx` to be compiled as C files using Cython.

### Description some of hyperparameters:
`workers`: Number of processes used for self play, Arena comparison, and training the model.

`process_batch_size`: The size of the batches used for batching MCTS during self play. Equivalent to the number of games that should be played at the same time.

`numItersForTrainExamplesHistory`: The number of past iterations to load self play training data from.

`max_moves`: Number of moves in the game before the game ends in a tie (should be implemented manually for now in getGameEnded of your Game class, automatic draw is planned).

`num_stacked_observations`: The number of past observations from the game to stack as a single observation. Should also be done manually for now, but take a look at how I did it in `alphazero/tafl/tafl.pyx`.

`numWarmupIters`: The number of warm up iterations to perform. Warm up is self play but with random policy and value to speed up initial generations of self play data. Should only be 1-3, otherwise the neural net is only getting random moves in games as training data. This can be done in the beginning because the model's actions are random anyways, so it's for performance.

`skipSelfPlayIters`: The number of self play data generation iterations to skip. This assumes that training data already exists for those iterations can be used for training. For example, useful when training is interrupted because data doesn't have to be generated from scratch because it's saved on disk.

`symmetricSamples`: Add symmetric samples to training data from self play based on the user-implemented method `getSymmetries` in their Game. Assumes that this is implemented. For example, in Viking Chess, the board can be rotated 90 degrees and mirror flipped any number of times while still being a valid game state, therefore this can be used for more training data.

`numMCTSSims`: Number of Monte Carlo Tree Search simulations to execute for each move in self play. A higher number is much slower, but also produces better value estimates.

`probFastSim`: The probability of a fast MCTS simulation to occur in self play, in which case `numFastSims` simulations are done instead of `numMCTSSims`. However, fast simulations are not saved to training history.

`tempThreshold`: The temperature is dropped to 0 in self play after this many steps in a game. In arena it is dropped to `arenaTemp`.

`max_gating_iters`: If a model doesn't beat it's own past iteration this many times, then gating is temporarily reset and the model is allowed to move on to the next iteration. Use `None` to disable this feature.

`min_next_model_winrate`: The miminum win rate required for the new iteration against the last model in order to move on. If it doesn't beat this number, the previous model is used again (model gating).

`cpuct`: A constant for controlling exploration vs exploitation in the MCTS aglorithm. A higher number promotes more exploration of new actopms whereas a lower one promotes exploitation of previously known good actions.

`num_channels`: The number of channels each ResNet Convolution block has.

`depth`: The number of stacked ResNet blocks to use in the network.

`value_head_channels/policy_head_channels`: The number of channels to use for the 1x1 value and policy convolution heads respectively. The value and policy heads pass data onto their respective dense layers.

`value_dense_layers/policy_dense_layers`: These arguments define the sizes and number of layers in the dense network of the value and policy head. This must be a list of integers where each element defines the number of neurons in the layer and the number of elements defines how many layers there should be.

## Results
Results also coming soon! As seen in the directory, I am currently training AlphaZero to play Viking Chess. I created the `boardgame` and `hnefatafl` modules to have a game engine for Viking Chess and to be able to visually play against the train model when ready using PyQt5. Help yourself to the code to see how I did this so maybe you can do something similar. The `boardgame` module is a base for building a game engine for a boardgame, and as you will see, `hnefatafl` builds off of it.
