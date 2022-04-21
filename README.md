# AlphaZero General
This is an implementation of AlphaZero based on the following repositories:

* The original repo: https://github.com/suragnair/alpha-zero-general
* A fork of the original repo: https://github.com/bhansconnect/fast-alphazero-general

This project is still work-in-progress, so expect frequent fixes, updates, and much more detailed documentation soon.

You may join the [Discord server](https://discord.gg/MVaHwGZpRC) if you wish to join the community and discuss this project, ask questions, or contribute to the framework's development.

### Current differences from the above repos
1. **Cython:** The most computationally intensive components are written in Cython to be compiled for a runtime speedup of [up to 30x](https://towardsdatascience.com/use-cython-to-get-more-than-30x-speedup-on-your-python-code-f6cb337919b6) compared to pure python.
2. **GUI:** Includes a graphical user interface for easier training and arena comparisons. It also allows for games to be played visually (agent-agent, agent-human, human-human) instead of through a command line interface (work-in-progress). Custom environments must implement their own GUI naturally.
3. **Node-based MCTS:** Uses a better implementation of MCTS that uses nodes instead of dictionary lookups. This allows for a huge increase in performance and much less RAM usage than what the previous implementation used, about 30-50% speed increase and 95% less RAM usage from experimental data. The base code for this was provided by [bhandsconnect](https://github.com/bhansconnect).
4. **Model Gating:** After each iteration, the model is compared to the previous iteration. The model that performs better continues forward based on an adjustable minimum winrate parameter.
5. **Batched MCTS:** [bhandsconnect's repo](https://github.com/bhansconnect/fast-alphazero-general) already includes this for self play, but it has been expanded upon to be included in Arena for faster comparison of models.
6. **N-Player Support:** Any number of players are supported! This allows for training on a greater variety of games such as many types of card games or something like Catan.
7. **Warmup Iterations:** A few self play iterations in the beginning of training can optionnally be done using random policy and value to speed up initial generation of training data instead of using a model that is initally random anyways. This makes these iterations purely CPU-bound.
8. **Root Dirichlet Noise & Root Temperature, Discount:** Allows for better exploration and MCTS doesn't get stuck in local minima as often. Discount allows AlphaZero to "understand" the concept of time and chooses actions which lead to a win more quickly/efficiently as opposed to choosing a win that would occur later on in the game.
9. **More Adjustable Parameters:** This implementation allows for the modification of numerous hyperparameters, allowing for substantial control over the training process. More on hyperparameters below where the usage of some are discussed.

## Getting Started
### Install required packages
Make sure you have Python 3 installed. Then run:
```pip3 install -r requirements.txt```

### Try one of the existing examples
1. Adjust the hyperparameters in one of the examples to your preference (path is ```alphazero/envs/<env name>/train.py```). Take a look at Coach.py where the default arguments are stored to see the available options. For example, edit ```alphazero/envs/connect4/train.py```.
2. After that, you can start training AlphaZero on your chosen environment by running the following:
```python3 -m alphazero.envs.<env name>.train```
Make sure that your working directory is the root of the repo.
3. You can observe how training is progressing from the console output, or you can also run tensorboard for a visual representation. To start tensorboard, run:
```tensorboard --logdir ./runs```
also from the project root. `runs` is the default directory for tensorboard data, but it can be changed in the hyperparameters.
4. Once you have trained a model and want to test it, either against itself or yourself, change ```alphazero/pit.py``` (or ```alphazero/envs/<env name>/pit.py```) to your needs and run it with:
```python3 -m alphazero.pit```
(once again, this will be easier to accomplish in future updates). You may also modify `roundrobin.py` to run a tournament with different iterations of models to rank them using a rating system.

### Create your own game to train on
More detailed documentation is on the way, but essentially you must subclass `GameState` from `alphazero/Game.py` and implement its abstract methods correctly. Your game engine subclass of `GameState` must be named `Game` and located in `alphazero/envs/<env name>/<env name>.py` in order for the GUI to recognize it. If this is done, just create a `train` file and choose hyperparameters accordingly and start training, or use the GUI to train and pit. Also, it may be helpful to use and subclass the `boardgame` module to create a new game engine more easily, as it implements some functions that can be useful.

As a general guideline, game engine files/other potential bottlenecks should be implemented in Cython, or at least stored as `.pyx` files to be compiled for runtime for increased performace.

### Description of some hyperparameters
**`workers`:** Number of processes used for self play, Arena comparison, and training the model. Should generally be set to the number of processors - 1.

**`process_batch_size`:** The size of the batches used for batching MCTS during self play. Equivalent to the number of games that should be played at the same time in each worker. For exmaple, a batch size of 128 with 4 workers would create 128\*4 = 512 total games to be played simultaneously.

**`minTrainHistoryWindow`, `maxTrainHistoryWindow`, `trainHistoryIncrementIters`:** The number of past iterations to load self play training data from. Starts at min and increments once every `trainHistoryIncrementIters` iterations until it reaches max.

**`max_moves`:** Number of moves in the game before the game ends in a draw (should be implemented manually for now in getGameEnded of your Game class, automatic draw is planned). Used for the calculation of `default_temp_scaling` function.

**`num_stacked_observations`:** The number of past observations from the game to stack as a single observation. Should also be done manually for now, but take a look at how it was implemented in `alphazero/envs/tafl/tafl.pyx`.

**`numWarmupIters`:** The number of warm up iterations to perform. Warm up is self play but with random policy and value to speed up initial generations of self play data. Should only be 1-3, otherwise the neural net is only getting random moves in games as training data. This can be done in the beginning because the model's actions are random anyways, so it's for performance.

**`skipSelfPlayIters`:** The number of self play data generation iterations to skip. This assumes that training data already exists for those iterations can be used for training. For example, useful when training is interrupted because data doesn't have to be generated from scratch because it's saved on disk.

**`symmetricSamples`:** Add symmetric samples to training data from self play based on the user-implemented method `symmetries` in their Game class. Assumes that this is implemented. For example, in Viking Chess, the board can be rotated 90 degrees and mirror flipped any number of times while still being a valid game state, therefore this can be used for more training data.

**`numMCTSSims`:** Number of Monte Carlo Tree Search simulations to execute for each move in self play. A higher number is much slower, but also produces better value and policy estimates.

**`probFastSim`:** The probability of a fast MCTS simulation to occur in self play, in which case `numFastSims` simulations are done instead of `numMCTSSims`. However, fast simulations are not saved to training history.

**`max_gating_iters`:** If a model doesn't beat its own past iteration this many times, then gating is temporarily reset and the model is allowed to move on to the next iteration. Use `None` to disable this feature.

**`min_next_model_winrate`:** The minimum win rate required for the new iteration against the last model in order to move on. If it doesn't beat this number, the previous model is used again (model gating).

**`cpuct`:** A constant for balancing exploration vs exploitation in the MCTS algorithm. A higher number promotes more exploration of new actions whereas a lower one promotes exploitation of previously known good actions. A normal range is between 1-4, depending on the environment; a game with less possible moves on each turn would need a lower CPUCT.

**`fpu_reduction`:** "First Play Urgency" reduction decreases the initialization Q value of an unvisited node by this factor, must be in the range `[-1, 1]`. The closer this value is to 1, it discourages MCTS to explore unvisited nodes further, which (hopefully) allows it to explore paths that are more familiar. If this is set to 0, no reduction is done and unvisited nodes inherit their parent's Q value. Closer to a value of -1 (not recommended to go below 0), unvisited nodes become more prefered which can lead to more exploration.

**`num_channels`:** The number of channels each ResNet convolution block has.

**`depth`:** The number of stacked ResNet blocks to use in the network.

**`value_head_channels/policy_head_channels`:** The number of channels to use for the 1x1 value and policy convolution heads respectively. The value and policy heads pass data onto their respective dense layers.

**`value_dense_layers/policy_dense_layers`:** These arguments define the sizes and number of layers in the dense network of the value and policy head. This must be a list of integers where each element defines the number of neurons in the layer and the number of elements defines how many layers there should be.

## Results
### Connect Four
`envs/connect4`

AlphaZero was trained on the `connect4` env for 208 iterations in the past, but unfortunately the specific args used to train it were lost. The args were quite close to the current default for the connect4 env (but with lower batch size and games/iteration, hence the large number of iterations), therefore the trained model can still be loaded with some trial and error.

This training instance was very successful, and was unbeatable by every human trial. Here are the Tensorboard logs:
![image](https://user-images.githubusercontent.com/28303167/164115107-61ccd431-0cd0-40c7-9814-10b8e277e1bf.png)
![image](https://user-images.githubusercontent.com/28303167/164115147-e79dc41b-4b68-4dc1-8146-198cf96e6647.png)
It can be seen that over time as total loss decreases, the model plays increasingly better against the baseline tester (which I believe was a raw MCTS player at the time). Note that the average game length and amount of draws also increase as the model understands the dynamics of the game better and struggles more to beat itself as it gets better.

Towards the end of training, the winrate against the past model suddenly decreases; I believe this is because the model has learnt to play a perfect game, and begins to overfit as it continues to generate very similar data via its self-play. This overfitting makes it less general and adaptable to dynamic situations, and therefore its past self can defeat it because it can adapt better.

The model file for the most successful iteration (193) can be downloaded [here](https://drive.google.com/file/d/111afRD0j9CD86nFyKueAAGdXNAmAxxDn/view?usp=sharing). As mentioned above, subsequent iterations underperformed most likely due to overfitting.

Another instance was trained later using the current default arguments. It was trained using more recent features such as FPU value, root temperature/dirichlet noise, etc. Only 35 iterations were trained, as it was intended just to test these new features.

I was surprised to see that even in only 35 iterations (which took approximately 8 hours on a GTX 1070 and i5-4690 CPU) it had reached superhuman capabilities. Take a look at the Tensorboard logs:
![image](https://user-images.githubusercontent.com/28303167/164116278-8eabfa9f-cd33-4f19-bf09-16906bfef0e6.png)
![image](https://user-images.githubusercontent.com/28303167/164116313-5afe1e73-b087-4827-a407-553b652b66de.png)
For unknown reasons, it does not perform as well against the baseline tester as the instance above, but this is probably due to the use of dirichlet noise and root temperature in Arena, which can cause AlphaZero to make a 'mistake' by random chance (which is intended for further exploration in self-play). However, if these are turned off, temperature is set to a low value (0-0.25), and more 'thinking' time is allowed (number of MCTS simulations are increased), then even this undertrained model can essentially play a perfect game.

The model for the lateset iteration (35) of this instance can be downloaded [here](https://drive.google.com/file/d/1goGnOWeQY2LmWounB-FPPoSH7ZdCg59X/view?usp=sharing).

### Viking Chess - Brandubh
`envs/brandubh`

The tensorboard logs have been corrupted for the best trained instance, therefore it cannot be included here. It was trained for 48 iterations with the default args included in the GUI (`AlphaZeroGUI/args/brandubh.json`), and achieved human-level results when testing.

However, the model does have a strange tendency to disregard obvious opportunities on occasion such as a victory in one move or blocking a defeat. Also, the game length seems to even out around 25 moves - despite the players' nearly even win rate - instead of increasing to the maximum as expected. This is being investigated, but it is either due to inappropriate hyperparameters, or a bug in the MCTS code regarding recent changes.

Iteration 48 of the model can be downloaded [here](https://drive.google.com/file/d/1rv9fiFQRUVBv-4PBkfmawtRm3wqAM67H/view?usp=sharing).
