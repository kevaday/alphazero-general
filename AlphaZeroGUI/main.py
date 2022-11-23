from PySide2.QtWidgets import QApplication, QMessageBox, QInputDialog, QTableWidgetItem, QLineEdit
from PySide2.QtGui import QFont
from PySide2.QtCore import Qt, QTimer
from AlphaZeroGUI import ARGS_DIR, ENVS_DIR, PLAYERS_MODULE, GENERIC_PLAYERS_MODULE, ALPHAZERO_ROOT
from AlphaZeroGUI._gui import Ui_FormMainMenu, Ui_DialogEditArgs, Ui_DialogCombo
from AlphaZeroGUI.CustomGUI import CustomGUI, MCTSEvaluator
from alphazero.Coach import DEFAULT_ARGS, Coach, TrainState
from alphazero.Arena import Arena, ArenaState
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict
from alphazero import CALLABLE_PREFIX
from tensorboard import program
from queue import Queue
from functools import partial
from pathlib import Path
from typing import Callable

# Options for args eval
from torch.optim import *
from torch.optim.lr_scheduler import *
from alphazero.GenericPlayers import *
from alphazero.utils import default_temp_scaling, const_temp_scaling

import os
import sys
import json
import errno
import inspect
import threading
import webbrowser

from pyximport import install as pyxinstall
from numpy import get_include

pyxinstall(setup_args={'include_dirs': get_include()})

ERROR_INVALID_NAME = 123
AUTO_END_TRAIN_TIME = 10
CUSTOM_GUI_UPDATE_INTERVAL = 33  # 33.33ms = 30fps
GUI_MODULE_NAME = 'gui'
EVAL_MAX_RUN_TIME = 10


def show_dialog(txt: str, parent, title: str = None, error=False, modal=True):
    dialog = QMessageBox(parent)
    dialog.setStandardButtons(QMessageBox.Ok)
    dialog.setText(txt)
    dialog.setModal(modal)

    if error:
        if not title:
            title = 'Error'
        dialog.setIcon(QMessageBox.Critical)
    elif not title:
        title = 'Dialog'

    dialog.setWindowTitle(title)
    dialog.show()
    return dialog


def is_pathname_valid(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.

    source: https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    """
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        _, pathname = os.path.splitdrive(pathname)
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError:
        return False
    else:
        return True


class _CustomGUIPlayerWrapper:
    SENTINEL = object()

    def __init__(self, player: BasePlayer, action_queue: Queue,
                 on_play_called: Callable[[BasePlayer, GameState], None]):
        self.player = player
        self.__action_q = action_queue
        self.on_play_called = on_play_called

        # Inherit methods from player_cls, except for the ones we override
        self.overrides = [method_name for method_name in dir(self) if
                          not method_name.startswith(f'_{self.__class__.__name__}__')
                          and callable(getattr(self, method_name))]
        """
        for method_name in dir(player_cls):
            if not method_name.startswith(f'_{player_cls.__name__}__') and method_name not in overrides:
                method = getattr(player_cls, method_name)
                if callable(method):
                    setattr(self, method_name, partial(method, self))
        """

    def __getattr__(self, item):
        if item in self.overrides:
            return getattr(self, item)
        else:
            return getattr(self.player, item)

    def __call__(self, *args, **kwargs):
        return self.play(*args, **kwargs)

    def play(self, state: GameState) -> int:
        self.on_play_called(self, state)
        if self.is_human():
            return self.__action_q.get()
        else:
            return self.player.play(state)


class _ThreadWithReturnValue(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self._return


class MainWindow(Ui_FormMainMenu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.btnTabTrain.clicked.connect(self.train_tab_clicked)
        self.btnTabPit.clicked.connect(self.pit_tab_clicked)
        self.btnLoadTrainArgs.clicked.connect(lambda: self._show_load_args_dialog(self.train_args))
        self.btnLoadPitArgs.clicked.connect(lambda: self._show_load_args_dialog(self.pit_args))
        self.btnEditTrainArgs.clicked.connect(lambda: self._show_args_editor(self.train_args))
        self.btnEditPitArgs.clicked.connect(lambda: self._show_args_editor(self.pit_args))
        self.btnLoadTrainEnv.clicked.connect(self._show_env_select_dialog)
        self.btnLoadPitEnv.clicked.connect(self._show_env_select_dialog)
        self.btnEditPlayers.clicked.connect(self._show_player_editor)
        self.btnOpenTensorboard.clicked.connect(self.open_tensorboard)
        self.btnRemoveArgs.clicked.connect(self.remove_saved_args)
        self.btnRemoveArgsPit.clicked.connect(self.remove_saved_args)
        self.btnTrainControlStart.clicked.connect(self.start_train_clicked)
        self.btnTrainControlPause.clicked.connect(self.pause_train)
        self.btnTrainControlStop.clicked.connect(self.stop_train)
        self.btnPitControlStart.clicked.connect(self.start_pit_clicked)
        self.btnPitControlPause.clicked.connect(self.pause_pit)
        self.btnPitControlStop.clicked.connect(self.stop_pit)

        self.btnTrainControlStart.setEnabled(False)
        self.btnTrainControlPause.setEnabled(False)
        self.btnTrainControlStop.setEnabled(False)
        self.btnPitControlStart.setEnabled(False)
        self.btnPitControlPause.setEnabled(False)
        self.btnPitControlStop.setEnabled(False)

        self.train_args = dotdict()
        self.train_timer = QTimer(self)
        self.train_timer.setInterval(500)
        self.train_timer.timeout.connect(self.train_update)
        self.train_args_name = None
        self.train_thread = None
        self.coach: Coach = None

        self.pit_args = dotdict()
        self.pit_timer = QTimer(self)
        self.pit_timer.setInterval(500)
        self.pit_timer.timeout.connect(self.pit_update)
        self.pit_args_name = None
        self.pit_player_classes = []
        self.pit_current_players = []
        self.pit_models = dict()
        self.pit_chosen_eval_model = None
        self.pit_eval_max_runtime = EVAL_MAX_RUN_TIME
        self.pit_thread = None
        self.arena: Arena = None

        self.custom_gui: CustomGUI = None
        self.custom_gui_action_queue = Queue()

        self.train_ended_counter = 0
        self.current_env_class: GameState = None
        self.current_gui_class: CustomGUI = None
        self.env_module = None
        self.env_name = None
        self.tb_url = None

        self.update_frames()
        self.update_env_and_args()

    @property
    def is_train(self):
        return self.btnTabTrain.isChecked()

    def update_frames(self):
        self.frameTrain.setHidden(not self.is_train)
        self.framePit.setHidden(self.is_train)

    def train_tab_clicked(self):
        if self.frameTrain.isHidden():
            self.btnTabPit.setChecked(False)
            self.update_frames()
        if not self.is_train:
            self.btnTabTrain.setChecked(True)

    def pit_tab_clicked(self):
        if self.framePit.isHidden():
            self.btnTabTrain.setChecked(False)
            self.update_frames()
        if not self.btnTabPit.isChecked():
            self.btnTabPit.setChecked(True)

    def get_args_name(self) -> str:
        return self.train_args_name if self.is_train else self.pit_args_name

    def set_args_name(self, value: str):
        if self.is_train:
            self.train_args_name = value
        else:
            self.pit_args_name = value

    def update_stats(self):
        train_status = pit_status = 'Status: '
        if self.coach:
            if self.coach.state == TrainState.STANDBY:
                train_status += 'Ready to train'
            elif self.coach.state == TrainState.INIT_AGENTS:
                train_status += 'Initializing self play agents'
            elif self.coach.state == TrainState.SELF_PLAY:
                train_status += 'Self playing'
            elif self.coach.state == TrainState.SAVE_SAMPLES:
                train_status += 'Saving self play samples'
            elif self.coach.state == TrainState.KILL_AGENTS:
                train_status += 'Killing self play agents'
            elif self.coach.state == TrainState.PROCESS_RESULTS:
                train_status += 'Processing self play games'
            elif self.coach.state == TrainState.TRAIN:
                train_status += 'Training net'
            elif self.coach.state == TrainState.COMPARE_BASELINE:
                train_status += 'Comparing to baseline'
            elif self.coach.state == TrainState.COMPARE_PAST:
                train_status += 'Comparing to past'
        else:
            if not self.train_args:
                train_status += 'Not ready, choose args'
            elif not self.current_env_class:
                train_status += 'Not ready, choose env'
            else:
                train_status += 'Ready to train'

        if self.arena:
            if self.arena.state == ArenaState.STANDBY:
                pit_status += 'Ready to start'
            elif self.arena.state == ArenaState.SINGLE_GAME:
                pit_status += 'Playing single game'
            elif self.arena.state == ArenaState.PLAY_GAMES:
                pit_status += 'Playing games'
        else:
            if not self.pit_args:
                pit_status += 'Not ready, choose args'
            elif not self.current_env_class:
                pit_status += 'Not ready, choose env'
            elif not self.pit_player_classes:
                pit_status += 'Not ready, choose players'
            else:
                pit_status += 'Ready to start'

        self.lblStatus.setText(train_status)
        self.lblPitStatus.setText(pit_status)
        if (not self.coach and self.is_train) or (not self.arena and not self.is_train): return

        if self.is_train:
            self.lblTrainIteration.setText('Iteration: ' + str(self.coach.model_iter))
            self.lblTrainSelfPlayIter.setText('Self Play Iter: ' + str(self.coach.self_play_iter))
            self.lblTrainLossPolicy.setText('Policy Loss: ' + str(round(self.coach.train_net.l_pi, 3)))
            self.lblTrainLossValue.setText('Value Loss: ' + str(round(self.coach.train_net.l_v, 3)))
            self.lblTrainLossTotal.setText('Total Loss: ' + str(round(self.coach.train_net.l_total, 3)))

            if self.coach.state == TrainState.SELF_PLAY:
                self.lblTrainNumGames.setText(
                    f'Games Played: {self.coach.games_played.value}/{self.train_args.gamesPerIteration}'
                )
                self.lblTrainEpsTime.setText('Episode Time: ' + str(round(self.coach.sample_time, 3)))
                self.lblTrainIterTime.setText('Iteration Time: ' + str(self.coach.iter_time))
                self.lblTrainTimeRemaining.setText('Est. Time Remaining: ' + str(self.coach.eta))

            elif self.coach.state == TrainState.TRAIN:
                self.lblTrainNumGames.setText(
                    f'Train Step: {self.coach.train_net.current_step}/{self.coach.train_net.total_steps}'
                )
                self.lblTrainEpsTime.setText('Train Step Time: ' + str(round(self.coach.train_net.step_time, 3)))
                self.lblTrainIterTime.setText('Train Time: ' + str(self.coach.train_net.elapsed_time))
                self.lblTrainTimeRemaining.setText('Est. Time Remaining: ' + str(self.coach.train_net.eta))

            elif (
                self.coach.state == TrainState.COMPARE_PAST or self.coach.state == TrainState.COMPARE_BASELINE
            ) and self.coach.arena:
                self.lblTrainNumGames.setText(
                    f'Games Played: {self.coach.arena.games_played}/{self.coach.arena.total_games}'
                )
                self.lblTrainEpsTime.setText('Episode Time: ' + str(round(self.coach.arena.eps_time, 3)))
                self.lblTrainIterTime.setText('Total Time: ' + str(self.coach.arena.total_time))
                self.lblTrainTimeRemaining.setText('Est. Time Remaining: ' + str(self.coach.arena.eta))

        else:
            if self.arena.total_games > 1:
                self.lblPitNumGames.setText(f'Games Played: {self.arena.games_played}/{self.arena.total_games}')
            else:
                self.lblPitNumGames.setText(
                    f'Num. Turns: {self.arena.game_state.turns}/{self.arena.game_state.max_turns()}'
                )

            winrates = [round(p.winrate, 3) for p in self.arena.players]
            if any(winrates):
                self.lblPitWinrates.setText(f'Win Rates: {winrates}')
            self.lblPitEpsTime.setText(f'Episode Time: {round(self.arena.eps_time, 3)}')
            self.lblPitIterTime.setText(f'Total Time: {self.arena.total_time}')
            self.lblPitTimeRemaining.setText(f'Est. Time Remaining: {self.arena.eta}')

    def start_train_clicked(self):
        if self.coach and self.coach.pause_train.is_set() and not self.coach.stop_train.is_set():
            self.coach.pause_train.clear()
            return
        try:
            self.coach = Coach(self.current_env_class, NNetWrapper(self.current_env_class, self.train_args), self.train_args)
        except (RuntimeError, IOError) as e:
            show_dialog('An error occurred loading the model: ' + str(e), self, error=True)
            return

        self.btnTrainControlStart.setEnabled(False)
        self.btnTrainControlPause.setEnabled(True)
        self.btnTrainControlStop.setEnabled(True)

        self.btnLoadTrainArgs.setEnabled(False)
        self.btnEditTrainArgs.setEnabled(False)
        self.btnLoadTrainEnv.setEnabled(False)

        self.train_thread = threading.Thread(target=self.coach.learn, daemon=True)
        self.train_thread.start()
        self.train_timer.start()

    def stop_train(self):
        self.coach.stop_train.set()
        if self.coach.arena:
            self.coach.arena.stop_event.set()

        self.train_timer.stop()
        self.train_thread.join()

        self.btnTrainControlStart.setEnabled(True)
        self.btnTrainControlPause.setEnabled(False)
        self.btnTrainControlStop.setEnabled(False)

        self.btnLoadTrainArgs.setEnabled(True)
        self.btnEditTrainArgs.setEnabled(True)
        self.btnLoadTrainEnv.setEnabled(True)

        self.progressIteration.setValue(0)
        self.progressTotal.setValue(0)

        self.train_args.selfPlayModelIter = self.coach.self_play_iter
        try:
            self._save_args_from(self.train_args, ARGS_DIR / (self.train_args_name + '.json'))
        except (IOError, OSError) as e:
            show_dialog('Unable to save args after training was stopped: ' + str(e), self, error=True)

        self.coach = None
        self.update_stats()

    def pause_train(self):
        self.coach.pause_train.set()
        if self.coach.arena:
            self.coach.arena.pause_event.set()

        self.btnTrainControlStart.setEnabled(True)
        self.btnTrainControlPause.setEnabled(False)
        self.btnTrainControlStop.setEnabled(True)

    def train_update(self):
        self.update_stats()
        if self.coach.state == TrainState.SELF_PLAY:
            self.progressIteration.setValue(self.coach.games_played.value / self.train_args.gamesPerIteration * 100)
        elif self.coach.state == TrainState.TRAIN and self.coach.train_net.total_steps:
            self.progressIteration.setValue(self.coach.train_net.current_step / self.coach.train_net.total_steps * 100)
        elif self.coach.state == TrainState.COMPARE_PAST or self.coach.state == TrainState.COMPARE_BASELINE:
            self.progressIteration.setValue(self.coach.arena.games_played / self.coach.arena.total_games * 100)
        else:
            self.progressIteration.setValue(0)
        self.progressTotal.setValue(self.coach.model_iter / self.train_args.numIters * 100)

        if self.coach.state == TrainState.STANDBY:
            self.train_ended_counter += 1
        else:
            self.train_ended_counter = 0

        if self.train_ended_counter * self.train_timer.interval() >= AUTO_END_TRAIN_TIME:
            # TODO: when a new iteration starts after saving samples, this is triggered while training is still running
            # (only happens when using model in self-play instead of warmup)
            self.stop_train()

    def set_pit_controls(self, enabled=True):
        self.btnPitControlStart.setEnabled(enabled)
        self.btnPitControlPause.setEnabled(not enabled)
        self.btnPitControlStop.setEnabled(not enabled)

        self.btnLoadPitArgs.setEnabled(enabled)
        self.btnLoadPitEnv.setEnabled(enabled)
        self.btnEditPitArgs.setEnabled(enabled)
        self.btnEditPlayers.setEnabled(enabled)

        self.checkShufflePlayers.setEnabled(enabled)
        self.checkBatchedArena.setEnabled(enabled)
        self.checkConsoleVerbose.setEnabled(enabled)
        self.checkShowVisual.setEnabled(enabled)

    def start_pit_clicked(self):
        self.set_pit_controls(False)

        if self.arena and self.arena.pause_event.is_set() and not self.arena.stop_event.is_set():
            self.arena.pause_event.clear()
            return

        if self.checkBatchedArena.isChecked() and not all([p.supports_process() for p in self.pit_player_classes]):
            show_dialog('Cannot start Arena with the current players and Batched Arena option because '
                        'not all selected players support batched Arena.', self, error=True)
            self.set_pit_controls()
            return

        text, ok = QInputDialog.getText(
            self, 'Number of Games', 'Enter the number of games to play:', QLineEdit.Normal,
            text=str(self.pit_args.arenaCompare), inputMethodHints=Qt.InputMethodHint.ImhDigitsOnly
        )
        if ok:
            try:
                num_games = int(text)
            except ValueError:
                show_dialog('Invalid value for number of games was provided.', self, error=True)
                self.set_pit_controls()
                return
        else:
            self.set_pit_controls()
            return

        self.pit_current_players = []
        for i, player in enumerate(self.pit_player_classes):
            args = [self.current_env_class, self.pit_args, self.checkConsoleVerbose.isChecked()]

            if player.requires_model():
                filename, model = self.pit_models[i]
                if not model.loaded:
                    try:
                        model.load_checkpoint(Path(self.pit_args.checkpoint) / self.pit_args.run_name, filename)
                    except (IOError, RuntimeError) as e:
                        show_dialog(f'Unable to load the model file {filename}: {e}', self, error=True)
                        self.set_pit_controls()
                        return

                args.insert(0, model)
            self.pit_current_players.append(self.pit_player_classes[i](*args))

        if hasattr(self.env_module, 'display'):
            display = self.env_module.display
            verbose = self.checkConsoleVerbose.isChecked()
        else:
            display = None
            verbose = False

        if self.checkShowVisual.isChecked() and not self.checkBatchedArena.isChecked():
            evaluator = None
            if self.pit_chosen_eval_model and self.pit_chosen_eval_model[0] != 'None':
                if (
                    not self.pit_chosen_eval_model[1].loaded
                    and self.pit_chosen_eval_model[0] != 'RawMCTS'
                ):
                    try:
                        self.pit_chosen_eval_model[1].load_checkpoint(
                            Path(self.pit_args.checkpoint) / self.pit_args.run_name, self.pit_chosen_eval_model[0]
                        )
                    except (IOError, RuntimeError) as e:
                        show_dialog(f'Unable to load the model file {self.pit_chosen_eval_model[0]}: {e}', self,
                                    error=True)
                        self.set_pit_controls()
                        return

                evaluator = MCTSEvaluator(
                    args=self.pit_args,
                    model=self.pit_chosen_eval_model[1] if self.pit_chosen_eval_model[0] != 'RawMCTS' else None,
                    max_search_time=self.pit_eval_max_runtime if self.pit_eval_max_runtime else None
                )

            # wrap all players with custom gui wrapper class
            # to receive signals, etc.
            for i in range(len(self.pit_current_players)):
                self.pit_current_players[i] = _CustomGUIPlayerWrapper(
                    self.pit_current_players[i],
                    self.custom_gui_action_queue,
                    self.custom_gui_arena_play_called
                )

            if evaluator:
                qm = QMessageBox
                show_hints = qm.question(self, 'Move Hints', 'Do you want to show move hints?',
                                         qm.Yes | qm.No) == qm.Yes
            else:
                show_hints = False

            self.custom_gui: CustomGUI = self.current_gui_class(
                self.custom_gui_player_moved,
                self.custom_gui_closed,
                user_input=False,
                show_hints=show_hints,
                title=f'{self.env_name} Game',
                evaluator=evaluator
            )
        elif self.checkShowVisual.isChecked():
            show_dialog('Cannot show visual output with Batched Arena option.', self, title='Warning', modal=True)

        try:
            self.arena = Arena(self.pit_current_players, self.current_env_class,
                               use_batched_mcts=self.checkBatchedArena.isChecked(),
                               display=display, args=self.pit_args)
        except ValueError as e:
            show_dialog(str(e), self, error=True)
            self.set_pit_controls()
            return

        self.pit_thread = _ThreadWithReturnValue(
            target=self.arena.play_games,
            args=(num_games, verbose, self.checkShufflePlayers.isChecked()),
            daemon=True
        )
        self.pit_thread.start()
        self.pit_timer.start()

        if self.custom_gui:
            self.custom_gui.show()

    def stop_pit(self):
        self.arena.stop_event.set()
        self.pit_timer.stop()

        self.set_pit_controls()
        self.progressPit.setValue(0)

        if self.custom_gui:
            self.custom_gui_action_queue.put(_CustomGUIPlayerWrapper.SENTINEL)
            self.custom_gui.user_input = False

        try:
            wins, draws, winrates = self.pit_thread.join()
        except TypeError:
            wins = draws = winrates = 'N/A'
        self.update_stats()
        if self.custom_gui:
            self.custom_gui.update_state(self.arena.game_state)
        self.arena = None

        msgbox = QMessageBox(self)
        msgbox.setText(
            f'Arena ended, the results are:\n\nWins of Players: {wins}\nDraws: {draws}\nWinrates: {winrates}'
        )
        msgbox.setFont(QFont('Arial', 14))
        msgbox.setWindowTitle('Arena Result')
        if self.custom_gui:
            msgbox.accepted.connect(self.custom_gui.close)
            msgbox.rejected.connect(self.custom_gui.close)
        msgbox.show()

    def pause_pit(self):
        self.arena.pause_event.set()
        self.btnPitControlStart.setEnabled(True)
        self.btnPitControlPause.setEnabled(False)
        self.btnPitControlStop.setEnabled(True)

    def pit_update(self):
        if self.arena and (self.arena.pause_event.is_set() or self.arena.stop_event.is_set()):
            return

        self.update_stats()
        if self.arena.total_games > 0:
            self.progressPit.setValue(self.arena.games_played / self.arena.total_games * 100)
        else:
            self.progressPit.setValue(self.arena.game_state.turns / self.arena.game_state.max_turns() * 100)

        if self.arena.state == ArenaState.STANDBY:
            self.stop_pit()

    def custom_gui_player_moved(self, action: int):
        self.custom_gui_action_queue.put(action)
        self.custom_gui.user_input = False

    def custom_gui_arena_play_called(self, player: BasePlayer, state: GameState):
        self.custom_gui.update_state(state)
        if player.is_human():
            self.custom_gui.user_input = True

    def custom_gui_closed(self):
        if self.arena:
            self.stop_pit()
        self.custom_gui = None

    def _show_env_select_dialog(self):
        def dialog_accepted():
            chosen_env = dialog.comboBox.currentText()
            if not chosen_env:
                show_dialog('No env was selected.', self, 'Info')
                return

            try:
                self.env_module = __import__(
                    str(ENVS_DIR / chosen_env / chosen_env).replace(os.sep, '.'), fromlist=['']
                )
                self.current_env_class = self.env_module.Game
            except Exception as e:
                show_dialog('Failed to load the selected training env: ' + str(e), self, error=True)
                return

            try:
                self.current_gui_class = __import__(
                    str(ENVS_DIR / chosen_env / GUI_MODULE_NAME).replace(os.sep, '.'), fromlist=['']
                ).GUI
            except Exception as e:
                show_dialog(
                    f'Could not find GUI module for the env `{chosen_env}`, custom GUI will not be available.', self
                )
                self.current_gui_class = None

            self.env_name = chosen_env
            self.pit_current_players = []
            self.update_env_and_args()
            dialog.close()

        envs = [x.name for x in ENVS_DIR.glob('*') if x.is_dir() and x.name != '__pycache__']
        if not envs:
            show_dialog(f'No training environments were found. Please make sure envs are located in the directory '
                        f'{ENVS_DIR} to be found.', self, 'No Envs Found')
            return

        dialog = Ui_DialogCombo(parent=self)
        dialog.comboBox.addItems(envs)
        if self.env_name and self.env_name in envs:
            dialog.comboBox.setCurrentText(self.env_name)
        dialog.lblTitle.setText('Select an environment from below:')
        dialog.setWindowTitle('Select Env')
        dialog.btnBox.accepted.connect(dialog_accepted)
        dialog.btnBox.rejected.connect(dialog.close)
        dialog.show()

    def _show_load_args_dialog(self, args):
        def dialog_accepted():
            nonlocal dialog
            chosen_args = dialog.comboBox.currentText()
            if not chosen_args:
                show_dialog('No args were selected.', self, 'Info')
                return

            chosen_file = ARGS_DIR / Path(chosen_args + '.json')
            if not chosen_file.exists():
                show_dialog('The selected arg could not be found.', self, error=True)
                return

            try:
                self._load_args_into(args, chosen_file)
            except (IOError, OSError) as e:
                show_dialog('Failed to load the selected args: ' + str(e), self, error=True)
                return

            self.set_args_name(chosen_args)
            self.update_env_and_args()
            dialog.close()

        arg_files = [x.name.replace('.json', '') for x in ARGS_DIR.glob('*.json')]
        if not arg_files:
            show_dialog('No saved args were found. Please create one by pressing "Edit Args".', self, 'Args Not Found')
            return

        dialog = Ui_DialogCombo(parent=self)
        dialog.comboBox.addItems(arg_files)
        dialog.lblTitle.setText('Select args from below:')
        dialog.setWindowTitle('Load Args')
        dialog.btnBox.accepted.connect(dialog_accepted)
        dialog.btnBox.rejected.connect(dialog.close)
        dialog.show()

    def _show_args_editor(self, args: dotdict):
        def init_table(arguments: dotdict):
            keys = []
            values = []

            for k, v in arguments.items():
                if k.startswith('_'):
                    # skip private fields to hide from user
                    continue

                keys.append(k)
                if isinstance(v, str):
                    if CALLABLE_PREFIX in v:
                        values.append(v.replace(CALLABLE_PREFIX, ''))
                    else:
                        values.append(f'"{v}"')

                elif callable(v):
                    values.append(v.__name__)
                else:
                    values.append(str(v))

            dialog.tableArgs.setRowCount(0)
            dialog.tableArgs.setRowCount(len(keys))
            dialog.tableArgs.setVerticalHeaderLabels(keys)
            for i, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                dialog.tableArgs.setItem(i, 0, item)

            dialog.tableArgs.resizeRowsToContents()
            dialog.tableArgs.resizeColumnsToContents()

        def default_table():
            if dialog.tableArgs.rowCount() > 0 and QMessageBox.question(
                    self,
                    'Are you sure?',
                    'Are you sure you want to load the default arguments into the editor?'
                    ' This will overwrite current changes.'
            ) == QMessageBox.No: return
            init_table(DEFAULT_ARGS)

        def dialog_accepted():
            new_args = dotdict()
            for i in range(dialog.tableArgs.rowCount()):
                key = dialog.tableArgs.verticalHeaderItem(i).text()
                str_value = dialog.tableArgs.item(i, 0).text()
                try:
                    value = eval(str_value)
                except Exception as e:
                    show_dialog(f'Invalid value encountered while saving args. '
                                f'The argument "{key}" has invalid value "{str_value}". Error: {e}', self, error=True)
                    return
                new_args.update({key: value})

            text, ok = QInputDialog.getText(
                self, 'Name Args', 'Choose a name for these args. Must be a valid path.',
                QLineEdit.Normal, text=self.get_args_name()
            )
            if ok:
                if not is_pathname_valid(text):
                    show_dialog('Invalid name was provided for args, must be a valid pathname.', self, error=True)
                    return

                self.set_args_name(text)

            try:
                if not ARGS_DIR.exists(): ARGS_DIR.mkdir()
                self._save_args_from(new_args, ARGS_DIR / (self.get_args_name() + '.json'))
            except (IOError, OSError) as e:
                show_dialog('Error while saving args to save file: ' + str(e), self, error=True)
                return

            if dialog.checkLoadArgs.isChecked():
                args.clear()
                args.update(self._parse_str_args(new_args))
            self.update_env_and_args()
            dialog.close()

        dialog = Ui_DialogEditArgs(parent=self)
        init_table(args)
        dialog.btnResetArgs.clicked.connect(default_table)
        dialog.btnBoxSave.accepted.connect(dialog_accepted)
        dialog.btnBoxSave.rejected.connect(dialog.close)
        dialog.show()

    def _show_player_editor(self):
        players = dict()
        player_modules = []

        def _add_module(mod_path):
            player_modules.append(__import__(str(mod_path).replace(os.sep, '.'), fromlist=['']))

        _add_module(ENVS_DIR / self.env_name / PLAYERS_MODULE)
        _add_module(ALPHAZERO_ROOT / GENERIC_PLAYERS_MODULE)

        for module in player_modules:
            for name, cls in inspect.getmembers(module):
                if inspect.isclass(cls) and issubclass(cls, BasePlayer) and cls != BasePlayer:
                    players.update({name: cls})

        chosen_players = []
        chosen_models = dict()
        chosen_eval_model = None
        max_eval_time = EVAL_MAX_RUN_TIME
        dialog = None
        model_dialog = None
        user_cancelled = False

        def _accept_model():
            chosen_model = model_dialog.comboBox.currentText()
            chosen_models[len(chosen_players) - 1] = (chosen_model, NNetWrapper(self.current_env_class, self.pit_args))
            model_dialog.close()
            dialog.close()

        def _accept_eval_model():
            nonlocal chosen_eval_model
            chosen_eval_model = (
                model_dialog.comboBox.currentText(),
                NNetWrapper(self.current_env_class, self.pit_args)
            )
            model_dialog.close()
            dialog.close()

        def _show_model_dialog(accepted=_accept_model, add_items=None, display_text=None):
            model_files = [x.name for x in (Path(self.pit_args.checkpoint) / self.pit_args.run_name).glob('*.pkl')]
            if add_items:
                model_files = add_items + model_files

            nonlocal model_dialog
            model_dialog = Ui_DialogCombo(parent=self)
            model_dialog.comboBox.addItems(model_files)

            last_model = self.pit_models.get(len(chosen_players) - 1)
            if last_model:
                model_dialog.comboBox.setCurrentText(last_model[0])

            model_dialog.lblTitle.setText(
                display_text if display_text else 'Select the model to use for this player:'
            )
            model_dialog.setWindowTitle('Edit Players')
            model_dialog.btnBox.accepted.connect(accepted)
            model_dialog.btnBox.rejected.connect(_cancel)
            dialog.hide()
            if model_dialog.exec_() != 0: _cancel()

        def _accept():
            chosen_players.append(players[dialog.comboBox.currentText()])
            if chosen_players[-1].requires_model():
                _show_model_dialog()
            else:
                dialog.close()

        def _cancel():
            nonlocal user_cancelled
            user_cancelled = True
            dialog.close()
            if model_dialog:
                model_dialog.close()

        for player_idx in range(self.current_env_class.num_players()):
            if user_cancelled: return
            dialog = Ui_DialogCombo(parent=self)
            dialog.comboBox.addItems(players.keys())

            if self.pit_player_classes and player_idx < len(self.pit_player_classes):
                used_player = self.pit_player_classes[player_idx].__name__
                if used_player in players.keys():
                    dialog.comboBox.setCurrentText(used_player)

            dialog.lblTitle.setText(f'Select a player type for player {player_idx + 1}:')
            dialog.setWindowTitle('Edit Players')
            dialog.btnBox.accepted.connect(_accept)
            dialog.btnBox.rejected.connect(_cancel)
            if dialog.exec_() != 0: return
        if user_cancelled: return

        _show_model_dialog(
            _accept_eval_model,
            add_items=['None', 'RawMCTS'],
            display_text='Select the model to use for evaluation:'
        )
        if user_cancelled: return

        if chosen_eval_model[0] != 'None':
            text, ok = QInputDialog.getText(
                self, 'Max Evaluator Runtime', 'Enter the max runtime for the evaluator in seconds (0 for infinite):',
                QLineEdit.Normal, inputMethodHints=Qt.InputMethodHint.ImhPreferNumbers,
                text=str(self.pit_eval_max_runtime if self.pit_eval_max_runtime is not None else EVAL_MAX_RUN_TIME)
            )
            if ok:
                try:
                    max_eval_time = float(text)
                    if max_eval_time < 0:
                        show_dialog('Invalid max eval time was provided, must be a positive number.', self, error=True)
                        return
                except ValueError:
                    show_dialog('Invalid value for number of games was provided.', self, error=True)
                    return
            else:
                return

        self.pit_player_classes = chosen_players
        self.pit_models = chosen_models
        self.pit_chosen_eval_model = chosen_eval_model
        self.pit_eval_max_runtime = max_eval_time
        self.update_env_and_args()

    def _parse_str_args(self, args: dotdict):
        new_args = dotdict()
        for k, v in args.items():
            if isinstance(v, str) and CALLABLE_PREFIX in v:
                try:
                    v = eval(v.replace(CALLABLE_PREFIX, ''))
                except Exception as e:
                    show_dialog('Failed to parse argument file: ' + str(e), self, error=True)
                    return

            elif isinstance(v, dict):
                v = dotdict(v)

            new_args.update({k: v})

        return new_args

    @staticmethod
    def _get_str_args(args: dotdict):
        save_args = dict()
        for k, v in args.items():
            if callable(v):
                v = CALLABLE_PREFIX + v.__name__
            save_args.update({k: v})

        return save_args

    def _load_args_into(self, args: dotdict or dict, filepath, clear_args=True):
        if clear_args: args.clear()

        load_args = dotdict()
        with open(filepath) as f:
            load_args.update(json.load(f))

        args.update(self._parse_str_args(load_args))  # TODO: TypeError: 'NoneType' object is not iterable
        return args

    def _save_args_from(self, args: dotdict or dict, filepath, replace=True):
        if not replace and os.path.exists(filepath): return

        save_args = self._get_str_args(args)
        with open(filepath, 'w') as f:
            json.dump(save_args, f)

        return save_args

    def update_env_and_args(self):
        self.lblCurrentRun.setText(f'Current Env: {self.env_name}\nCurrent Train Args: {self.train_args_name}')
        self.lblCurrentPitRun.setText(
            f'Env: {self.env_name}\nArena Args: {self.pit_args_name}\nPlayers: '
            f'{", ".join([p.__name__ for p in self.pit_player_classes]) if self.pit_player_classes else None}\n'
            f'Evaluator: {self.pit_chosen_eval_model[0] if self.pit_chosen_eval_model else None}'
        )
        self.btnTrainControlStart.setEnabled(
            (self.coach is None or (self.coach is not None and self.coach.pause_train.is_set())) and self.current_env_class is not None and bool(self.train_args)
        )
        self.btnPitControlStart.setEnabled(
            (self.arena is None or (self.arena is not None and self.arena.pause_event.is_set())) and self.current_env_class is not None and bool(self.pit_args) and bool(self.pit_player_classes)
        )
        self.btnEditPlayers.setEnabled(self.current_env_class is not None and bool(self.pit_args))
        self.update_stats()

    def open_tensorboard(self):
        if not self.tb_url:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', 'runs'])
            self.tb_url = tb.launch()
        webbrowser.open(self.tb_url)

    def remove_saved_args(self):
        def dialog_accepted():
            chosen_args = dialog.comboBox.currentText()
            if not chosen_args:
                show_dialog('No args were selected.', self, 'Info')
                return

            chosen_file = ARGS_DIR / Path(chosen_args + '.json')
            if not chosen_file.exists():
                show_dialog('The selected arg could not be found.', self, error=True)
                return

            if QMessageBox.question(
                    self,
                    'Are you sure?',
                    'Are you sure you want to delete the selected args file? This cannot be undone.'
            ) == QMessageBox.No:
                return

            try:
                chosen_file.unlink()
            except (IOError, OSError) as e:
                show_dialog('Failed to delete the selected args: ' + str(e), self, error=True)
                return

            dialog.close()

        arg_files = [x.name.replace('.json', '') for x in ARGS_DIR.glob('*.json')]
        if not arg_files:
            show_dialog('No saved args were found. Create args by pressing "Edit Args".', self, 'Args Not Found')
            return

        dialog = Ui_DialogCombo(parent=self)
        dialog.comboBox.addItems(arg_files)
        dialog.lblTitle.setText('Select args to delete from below:')
        dialog.setWindowTitle('Remove Args')
        dialog.btnBox.accepted.connect(dialog_accepted)
        dialog.btnBox.rejected.connect(dialog.close)
        dialog.show()


def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__": run()
