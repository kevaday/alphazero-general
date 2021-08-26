# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QMessageBox, QInputDialog, QTableWidgetItem, QLineEdit
from PySide2.QtCore import Qt, QTimer
from AlphaZeroGUI import ARGS_DIR, ENVS_DIR, CALLABLE_PREFIX
from AlphaZeroGUI._gui import Ui_FormMainMenu, Ui_DialogEditArgs, Ui_DialogCombo
from alphazero.Coach import DEFAULT_ARGS, Coach, TrainState
from alphazero.Arena import Arena
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict
from tensorboard import program
from pathlib import Path
from typing import Callable

# Options for args eval
from torch.optim import *
from torch.optim.lr_scheduler import *
from alphazero.GenericPlayers import *
from alphazero.utils import default_temp_scaling

import sys
import os
import json
import errno
import threading
import webbrowser

from pyximport import install as pyxinstall
from numpy import get_include

pyxinstall(setup_args={'include_dirs': get_include()})

ERROR_INVALID_NAME = 123


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
        self.btnOpenTensorboard.clicked.connect(self.open_tensorboard)
        self.btnRemoveArgs.clicked.connect(self.remove_saved_args)
        self.btnRemoveArgsPit.clicked.connect(self.remove_saved_args)
        self.btnTrainControlStart.clicked.connect(self.start_train_clicked)
        self.btnPitControlStart.clicked.connect(self.start_pit_clicked)

        self.btnTrainControlStart.setEnabled(False)
        self.btnTrainControlPause.setEnabled(False)
        self.btnTrainControlStop.setEnabled(False)
        self.btnPitControlStart.setEnabled(False)
        self.btnPitControlPause.setEnabled(False)
        self.btnPitControlStop.setEnabled(False)

        self.train_args = dotdict()
        self.pit_args = dotdict()
        self.train_timer = QTimer(self)
        self.train_timer.setInterval(500)
        self.train_timer.timeout.connect(self.train_update)
        self.pit_timer = QTimer(self)
        self.pit_timer.setInterval(500)
        self.pit_timer.timeout.connect(self.pit_update)
        self.train_args_name = None
        self.pit_args_name = None
        self.train_env_name = None
        self.pit_env_name = None
        self.current_env = None
        self.coach = None
        self.arena = None
        self.train_thread = None
        self.pit_thread = None
        self.tb_url = None

        self.update_frames()
        self.update_env_and_args()
        self.update_train_stats()

    def update_frames(self):
        self.frameTrain.setHidden(not self.btnTabTrain.isChecked())
        self.framePit.setHidden(self.btnTabTrain.isChecked())

    def train_tab_clicked(self):
        if self.frameTrain.isHidden():
            self.btnTabPit.setChecked(False)
            self.update_frames()
        if not self.btnTabTrain.isChecked():
            self.btnTabTrain.setChecked(True)

    def pit_tab_clicked(self):
        if self.framePit.isHidden():
            self.btnTabTrain.setChecked(False)
            self.update_frames()
        if not self.btnTabPit.isChecked():
            self.btnTabPit.setChecked(True)

    def get_args_name(self) -> str:
        return self.train_args_name if self.btnTabTrain.isChecked() else self.pit_args_name

    def get_env_name(self) -> str:
        return self.train_env_name if self.btnTabTrain.isChecked() else self.pit_env_name

    def set_args_name(self, value: str):
        if self.btnTabTrain.isChecked():
            self.train_args_name = value
        else:
            self.pit_args_name = value

    def set_env_name(self, value: str):
        if self.btnTabTrain.isChecked():
            self.train_env_name = value
        else:
            self.pit_env_name = value

    def update_train_stats(self):
        status = 'Status: '
        if not self.coach:
            if not self.train_args:
                status += 'Not ready, choose args'
            elif not self.current_env:
                status += 'Not ready, choose env'
            else:
                status += 'Ready to train'

        elif self.coach.state == TrainState.STANDBY:
            status += 'Ready to train'
        elif self.coach.state == TrainState.INIT_AGENTS:
            status += 'Initializing self play agents'
        elif self.coach.state == TrainState.SELF_PLAY:
            status += 'Self playing'
        elif self.coach.state == TrainState.SAVE_SAMPLES:
            status += 'Saving self play samples'
        elif self.coach.state == TrainState.KILL_AGENTS:
            status += 'Killing self play agents'
        elif self.coach.state == TrainState.PROCESS_RESULTS:
            status += 'Processing self play games'
        elif self.coach.state == TrainState.TRAIN:
            status += 'Training net'
        elif self.coach.state == TrainState.COMPARE_BASELINE:
            status += 'Comparing to baseline'
        elif self.coach.state == TrainState.COMPARE_PAST:
            status += 'Comparing to past'

        self.lblStatus.setText(status)
        if not self.coach: return
        self.lblTrainIteration.setText('Iteration: ' + str(self.coach.model_iter))
        self.lblTrainSelfPlayIter.setText('Self Play Iter: ' + str(self.coach.self_play_iter))
        self.lblTrainNumGames.setText(
            f'Games Played: {self.coach.games_played.value}/{self.train_args.gamesPerIteration}')
        self.lblTrainLossPolicy.setText('Policy Loss: ' + str(round(self.coach.train_net.l_pi, 3)))
        self.lblTrainLossValue.setText('Value Loss: ' + str(round(self.coach.train_net.l_v, 3)))
        self.lblTrainLossTotal.setText('Total Loss: ' + str(round(self.coach.train_net.l_total, 3)))
        if self.coach.state == TrainState.SELF_PLAY:
            self.lblTrainEpsTime.setText('Episode Time: ' + str(round(self.coach.sample_time, 3)
                                                                if self.coach.sample_time else ' '))
            self.lblTrainIterTime.setText('Iteration Time: ' + str(self.coach.iter_time or ' '))
            self.lblTrainTimeRemaining.setText('Est. Time Remaining: ' + str(self.coach.eta or ' '))
        elif self.coach.state == TrainState.TRAIN:
            self.lblTrainEpsTime.setText('Train Step Time: ' + str(round(self.coach.train_net.step_time, 3)))
            self.lblTrainIterTime.setText('Train Time: ' + str(self.coach.train_net.elapsed_time))
            self.lblTrainTimeRemaining.setText('Est. Time Remaining: ' + str(self.coach.train_net.eta))

    def start_train_clicked(self):
        self.btnTrainControlStart.setEnabled(False)
        self.btnTrainControlPause.setEnabled(True)
        self.btnTrainControlStop.setEnabled(True)

        if self.coach and self.coach.pause_train.is_set() and not self.coach.stop_train.is_set():
            self.coach.pause_train.clear()
            return
        try:
            self.coach = Coach(self.current_env, NNetWrapper(self.current_env, self.train_args), self.train_args)
        except (RuntimeError, IOError) as e:
            show_dialog('An error occurred loading the model: ' + str(e), self, error=True)
            return

        self.btnTrainControlPause.clicked.connect(
            lambda: (self.coach.pause_train.set(), self.coach.arena.pause_event.set() if self.coach.arena else None)
        )
        self.btnTrainControlStop.clicked.connect(
            lambda: (self.coach.stop_train.set(), self.coach.arena.stop_event.set() if self.coach.arena else None)
        )
        self.btnLoadTrainArgs.setEnabled(False)
        self.btnEditTrainArgs.setEnabled(False)
        self.btnLoadTrainEnv.setEnabled(False)

        self.train_thread = threading.Thread(target=self.coach.learn, daemon=True)
        self.train_thread.start()
        self.train_timer.start()

    def train_update(self):
        self.update_train_stats()
        if self.coach.state == TrainState.SELF_PLAY:
            self.progressIteration.setValue(self.coach.games_played.value / self.train_args.gamesPerIteration * 100)
        elif self.coach.state == TrainState.TRAIN:
            self.progressIteration.setValue(self.coach.train_net.current_step / self.coach.train_net.total_steps * 100)
        self.progressTotal.setValue(self.coach.model_iter / self.train_args.numIters * 100)

        if self.coach.stop_train.is_set():
            self.train_timer.stop()
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
            self.train_thread.join()

        elif self.coach.pause_train.is_set():
            self.btnTrainControlStart.setEnabled(True)
            self.btnTrainControlPause.setEnabled(False)
            self.btnTrainControlStop.setEnabled(True)

    def start_pit_clicked(self):
        self.btnPitControlStart.setEnabled(False)
        self.btnPitControlPause.setEnabled(True)
        self.btnPitControlStop.setEnabled(True)

        if self.arena and self.arena.pause_event.is_set() and not self.arena.stop_event.is_set():
            self.arena.pause_event.clear()
            return

        # self.arena = Arena()
        self.btnPitControlPause.clicked.connect(self.arena.pause_event.set)
        self.btnPitControlStop.clicked.connect(self.arena.stop_event.set)

    def pit_update(self):
        pass

    def _show_env_select_dialog(self):
        def dialog_accepted():
            chosen_env = dialog.comboBox.currentText()
            if not chosen_env:
                show_dialog('No env was selected.', self, 'Info')
                return

            try:
                env_module = __import__(str(ENVS_DIR / chosen_env / chosen_env).replace(os.sep, '.'), fromlist=[''])
                self.current_env = env_module.Game
            except Exception as e:
                show_dialog('Failed to load the selected training env: ' + str(e), self, error=True)
                return

            self.set_env_name(chosen_env)
            self.update_env_and_args()
            dialog.close()

        envs = [x.name for x in ENVS_DIR.glob('*') if x.is_dir() and x.name != '__pycache__']
        if not envs:
            show_dialog(f'No training environments were found. Please make sure envs are located in the directory '
                        f'{ENVS_DIR} to be found.', self, 'No Envs Found')
            return

        dialog = Ui_DialogCombo(parent=self)
        dialog.comboBox.addItems(envs)
        env_name = self.get_env_name()
        if env_name and env_name in envs:
            dialog.comboBox.setCurrentText(env_name)
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
                keys.append(k)
                if isinstance(v, str):
                    if CALLABLE_PREFIX in v:
                        values.append(v.replace(CALLABLE_PREFIX, ''))
                    else:
                        values.append(f'"{v}"')
                elif isinstance(v, Callable):
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
                value = dialog.tableArgs.item(i, 0).text().replace('"', '')
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
            if isinstance(v, Callable):
                v = CALLABLE_PREFIX + v.__name__
            save_args.update({k: v})

        return save_args

    def _load_args_into(self, args: dotdict or dict, filepath, clear_args=True):
        if clear_args: args.clear()

        load_args = dotdict()
        with open(filepath) as f:
            load_args.update(json.load(f))

        args.update(self._parse_str_args(load_args))
        return args

    def _save_args_from(self, args: dotdict or dict, filepath, replace=True):
        if not replace and os.path.exists(filepath): return

        save_args = self._get_str_args(args)
        with open(filepath, 'w') as f:
            json.dump(save_args, f)

        return save_args

    def update_env_and_args(self):
        self.lblCurrentRun.setText(f'Current Env: {self.train_env_name}\nCurrent Args: {self.train_args_name}')
        self.lblCurrentPitRun.setText(f'Current Env: {self.pit_env_name}\nCurrent Args: {self.pit_args_name}')
        self.btnTrainControlStart.setEnabled(self.current_env is not None and self.train_args is not None)
        self.btnPitControlStart.setEnabled(self.current_env is not None and self.pit_args is not None)
        self.update_train_stats()

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
