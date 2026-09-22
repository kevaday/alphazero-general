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
7. **Warmup Iterations:** A few self play iterations in the beginning of training can optionally be done using random policy and value to speed up initial generation of training data instead of using a model that is initially random anyways. This makes these iterations purely CPU-bound.
8. **Root Dirichlet Noise & Root Temperature, Discount:** Allows for better exploration and MCTS doesn't get stuck in local minima as often. Discount allows AlphaZero to "understand" the concept of time and chooses actions which lead to a win more quickly/efficiently as opposed to choosing a win that would occur later on in the game.
9. **More Adjustable Parameters:** This implementation allows for the modification of numerous hyperparameters, allowing for substantial control over the training process. More on hyperparameters below where the usage of some are discussed.

## Getting Started
### Install required packages
Make sure you have Python 3 installed. Then run:

```pip3 install -r requirements.txt```

### GUI (work-in-progress)
![image](https://user-images.githubusercontent.com/28303167/164362451-01590045-5070-45a1-8989-ab70e364b19f.png)

AlphaZeroGUI, built using PyQt5, is intended to simplify the training, hyperparameter selection, and deployment/inference processes as opposed to modifying different files and running in the command line. It can be run with the following command:

`python -m AlphaZeroGUI`

After that, the controls are generally intuitive. Default/saved arguments can be loaded, the environment can be selected (see section ***Create your own game*** for implementing environment for GUI), arguments can be edited/created, and tensorboard can be opened. Simple training stats are shown on the left side, and the progress is shown at the bottom.

![image](https://user-images.githubusercontent.com/28303167/164365609-30e374a9-0b82-46fd-b3c1-ac8155f24d8c.png)

At the top left, the Arena tab can be toggled as seen above. Here, a separate set of args & env can be loaded and the type of players can be selected. For example, in the above image the brandubh environment was loaded and an MCTS Player with a model is pitted against a human player.

For now, Arena is still displayed in the console, but eventually there will be support for each environment to implement its own graphical interface to play games (agent-agent, agent-player, player-player).

### Try one of the existing examples
1. Adjust the hyperparameters in the GUI editor or pass them to the root training command. Take a look at `Coach.py` where the default arguments are stored to see the available options. For example:

```bash
python train.py --env connect4 --numMCTSSims 200 --save-args args/connect4.json
```

Saved argument files can be loaded with `--load-args`; command-line arguments override loaded values. The same options are available for `pit.py`, along with `--players`, and `--player-checkpoint`.

2. After that, you can start training AlphaZero on your chosen environment by pressing the 'play' button in the GUI, or running the command above from the project root. Existing environment-specific scripts remain available for compatibility.

3. You can observe how training is progressing in the GUI, from the console output, or you can also run tensorboard for a visual representation. To start tensorboard in the console, run:

```tensorboard --logdir ./runs```

also from the project root. `runs` is the default directory for tensorboard data, but it can be changed in the hyperparameters.

4. Once you have trained a model and want to test it, either against itself or yourself, use the Arena tab in the GUI as described above, or run the root arena command:

```bash
python pit.py --env connect4 --players mcts human \
    --player-checkpoint checkpoint/connect4 iteration-0035.pkl
```

(once again, this will be easier to accomplish in future updates). You may also modify `roundrobin.py` to run a tournament with different iterations of models to rank them using a rating system.

### Create your own game to train on
More detailed documentation is on the way, but essentially you must subclass `GameState` from `alphazero/Game.py` and implement its abstract methods correctly. Your game engine subclass of `GameState` must be named `Game` and located in `alphazero/envs/<env name>/<env name>.py` in order for the GUI to recognize it. If this is done, just create a `train` file and choose hyperparameters accordingly and start training, or use the GUI to train and pit. Also, it may be helpful to use and subclass the `boardgame` module to create a new game engine more easily, as it implements some functions that can be useful.

As a general guideline, game engine files/other potential bottlenecks should be implemented in Cython, or at least stored as `.pyx` files to be compiled for runtime for increased performance.

### Hyperparameters

The defaults are defined in `alphazero/Coach.py` and can be overridden in the GUI, with
the training scripts, or from the command line. The root `train.py` command converts
underscores in option names to hyphens (for example, `process_batch_size` becomes
`--process-batch-size`); `--arg NAME=VALUE` can be used for an additional custom
argument. Values loaded with `--load-args` are overridden by explicit command-line
options.

#### Run and data management

| Parameter | Default | Description |
| --- | ---: | --- |
| `run_name` | `boardgame` | Name used for checkpoints, training data, and TensorBoard logs. |
| `cuda` | Auto-detected | Use CUDA when it is available. |
| `workers` | CPU count | Number of worker processes used for self-play, arena games, and data loading. |
| `startIter` | `0` | Iteration at which to start or resume training. |
| `numIters` | `1000` | Final training iteration. |
| `load_model` | `True` | Load the latest checkpoint for `run_name` when one exists. |
| `checkpoint` | `checkpoint` | Directory containing model checkpoints. |
| `data` | `data` | Directory containing self-play training data. |
| `gamesPerIteration` | `256 * workers` | Number of self-play games generated per iteration. |
| `process_batch_size` | `256` | Number of games processed together by each self-play worker during batched MCTS. |
| `skipSelfPlayIters` | `None` | Skip self-play through this iteration and reuse data already saved on disk. |
| `selfPlayModelIter` | `None` | Checkpoint iteration used by self-play; `None` uses the current training iteration. |
| `train_on_past_data` | `False` | Also train on data from another run. |
| `past_data_run_name` | `boardgame` | Run from which past training data is read. |
| `past_data_chunk_size` | `25` | Number of past iterations loaded per training-data chunk. |

#### Training data and optimization

| Parameter | Default | Description |
| --- | ---: | --- |
| `minTrainHistoryWindow` | `2` | Minimum number of recent iterations retained in the training window. |
| `maxTrainHistoryWindow` | `10` | Maximum number of recent iterations retained in the training window. |
| `trainHistoryIncrementIters` | `2` | Number of iterations between increases to the history window. |
| `train_batch_size` | `1024` | Batch size used by the neural-network trainer. |
| `train_steps_per_iteration` | `64` | Fixed training steps per iteration when automatic step selection is disabled. |
| `train_sample_ratio` | `1` | Multiplier for the number of training steps selected from available samples. |
| `averageTrainSteps` | `False` | Base automatic training-step selection on the average history size instead of the latest size. |
| `autoTrainSteps` | `True` | Automatically select training steps from the available samples. |
| `symmetricSamples` | `True` | Add game-defined symmetric copies of self-play samples via `Game.symmetries`. |
| `selfPlayDataBalance` | `0` | Maximum ratio of retained decisive games for any player to the least represented player. Set to `1` to retain equal numbers of wins for each player; `0` disables balancing. Draws are always retained and excluded from this calculation. |
| `lr` | `0.01` | Initial learning rate. |
| `optimizer` | `SGD` | PyTorch optimizer used for training. |
| `optimizer_args` | `momentum=0.9`, `weight_decay=1e-4` | Keyword arguments passed to the optimizer. |
| `scheduler` | `MultiStepLR` | Learning-rate scheduler used during training. |
| `scheduler_args` | `milestones=[75, 125]`, `gamma=0.1` | Keyword arguments passed to the scheduler. |
| `value_loss_weight` | `1.5` | Weight applied to the value-head loss. |

#### Self-play and MCTS

| Parameter | Default | Description |
| --- | ---: | --- |
| `numWarmupIters` | `1` | Initial iterations using random policy and value estimates instead of the neural network. Use `0` to disable. |
| `numMCTSSims` | `100` | MCTS simulations performed for each move. More simulations improve estimates but increase runtime. |
| `numFastSims` | `20` | Simulations used by a fast self-play search. |
| `numWarmupSims` | `5` | Simulations used during warmup iterations. |
| `numArenaSims` | `100` | Legacy setting retained for compatibility with older arena configurations. |
| `probFastSim` | `0` | Probability of using fast self-play search; fast-search samples are not saved to training history. |
| `startTemp` | `1` | Initial action-selection temperature during self-play. |
| `temp_scaling_fn` | `const_temp_scaling` | Function that updates self-play temperature as the game progresses. |
| `min_discount` | `1` | Minimum reward discount applied by MCTS to account for the time to reach an outcome. |
| `fpu_reduction` | `0` | First-play urgency reduction for unvisited MCTS nodes. |
| `cpuct` | `1.25` | MCTS exploration constant: higher values favor exploration, lower values favor exploitation. |
| `num_stacked_observations` | `2` | Number of consecutive observations combined into one network input; the game implementation must provide compatible observations. |
| `root_policy_temp` | `1.1` | Temperature applied to the policy prior at the root of the search. |
| `root_noise_frac` | `0.1` | Fraction of Dirichlet noise mixed into the root policy. |
| `add_root_noise` | `True` | Add Dirichlet noise at the root of MCTS. |
| `add_root_temp` | `True` | Apply root policy temperature during MCTS. |
| `mctsResetThreshold` | `None` | Legacy setting retained for compatibility with older MCTS configurations. |

#### Arena and model gating

| Parameter | Default | Description |
| --- | ---: | --- |
| `arenaCompare` | `128` | Number of games used to compare the current model with the previous model. |
| `arenaCompareBaseline` | `128` | Number of games used for comparisons against the baseline player. |
| `arena_batch_size` | `64` | Number of arena games processed together when batched arena play is enabled. |
| `arenaTemp` | `0` | Action-selection temperature used in arena games. |
| `arena_temp_scaling_fn` | `const_temp_scaling` | Function that updates arena temperature as the game progresses. |
| `arenaMCTS` | `True` | Use neural-network-guided MCTS in arena games; `False` uses direct neural-network play. |
| `arenaBatched` | `True` | Run arena comparisons with batched processes where supported. |
| `compareWithBaseline` | `True` | Compare models against `baselineTester`. |
| `baselineTester` | `RawMCTSPlayer` | Player class used for baseline comparisons. |
| `baselineCompareFreq` | `1` | Compare against the baseline every N iterations. |
| `compareWithPast` | `True` | Compare each new model against the previous model. |
| `pastCompareFreq` | `1` | Compare against the previous model every N iterations. |
| `model_gating` | `True` | Keep the previous model when the new model fails the gating win-rate requirement. |
| `max_gating_iters` | `None` | Maximum consecutive failed gating iterations before allowing the new model; `None` means no limit. |
| `min_next_model_winrate` | `0.52` | Minimum win rate required for the new model to replace the previous model. |
| `use_draws_for_winrate` | `True` | Count a draw as half a win when calculating gating win rate. |

#### Network architecture

| Parameter | Default | Description |
| --- | ---: | --- |
| `nnet_type` | `resnet` | Network architecture: `resnet` or `fc`. |
| `num_channels` | `32` | Channels in each ResNet convolution block. |
| `depth` | `4` | Number of residual blocks in a ResNet. |
| `value_head_channels` | `16` | Channels in the value head's 1x1 convolution. |
| `policy_head_channels` | `16` | Channels in the policy head's 1x1 convolution. |
| `input_fc_layers` | `[1024, 1024, 1024, 1024]` | Dense-layer sizes for the `fc` network input. |
| `value_dense_layers` | `[512, 64]` | Dense-layer sizes in the value head. |
| `policy_dense_layers` | `[512, 256]` | Dense-layer sizes in the policy head. |

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

The model for the latest iteration (35) of this instance can be downloaded [here](https://drive.google.com/file/d/1goGnOWeQY2LmWounB-FPPoSH7ZdCg59X/view?usp=sharing).

### Viking Chess - Brandubh
`envs/brandubh`

The tensorboard logs have been corrupted for the best trained instance, therefore it cannot be included here. It was trained for 48 iterations with the default args included in the GUI (`AlphaZeroGUI/args/brandubh.json`), and achieved human-level results when testing.

However, the model does have a strange tendency to disregard obvious opportunities on occasion such as a victory in one move or blocking a defeat. Also, the game length seems to even out around 25 moves - despite the players' nearly even win rate - instead of increasing to the maximum as expected. This is being investigated, but it is either due to inappropriate hyperparameters, or a bug in the MCTS code regarding recent changes.

Iteration 48 of the model can be downloaded [here](https://drive.google.com/file/d/1rv9fiFQRUVBv-4PBkfmawtRm3wqAM67H/view?usp=sharing).
