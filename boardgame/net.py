from boardgame.errors import GameFullError
from enum import Enum

import socketserver
import socket
import atexit
import threading
import time
import struct


def byte(n: int) -> bytes:
    return bytes([n])


class Message(Enum):
    DefaultGame = byte(0)
    CustomGame = byte(1)
    Welcome = byte(2)
    JoinGame = byte(3)
    AllGames = byte(4)

    Exit = byte(254)
    PreGame = byte(6)
    GameUpdate = byte(7)
    Game = byte(15)
    Chat = byte(8)
    Colour = byte(13)

    ErrorInvalidGame = byte(9)
    ErrorInvalidMessage = byte(10)
    ErrorGameFull = byte(11)
    ErrorPlayerLeft = byte(12)
    ErrorUserExists = byte(14)
    ErrorInternal = byte(15)
    Error = byte(255)


class _StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class _GameWrapper(list):
    def __init__(self, game, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = game


def _send(s, msg: bytes):
    msg = struct.pack('>I', len(msg)) + msg
    s.sendall(msg)


def _receive(s) -> bytes:
    def recvall(n):
        data = bytearray()
        while len(data) < n:
            packet = s.recv(n - len(data))
            if not packet:
                raise socket.error('Invalid packet received.')
            data.extend(packet)
        return data

    raw_msglen = recvall(4)
    if not raw_msglen:
        raise socket.error('Invalid message header received.')
    msglen = struct.unpack('>I', raw_msglen)[0]
    return bytes(recvall(msglen))


def print_time(text: str) -> None:
    print(f'{time.ctime(time.time())}: {text}')


def get_own_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    try:
        ip = s.getsockname()[0]
    except socket.error:
        ip = '127.0.0.1'
    s.close()

    return ip


class BaseRequestHandler(socketserver.StreamRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username = None

    def setup(self):
        super().setup()
        self.server.add_client(self)

    def send(self, msg: bytes):
        _send(self.request, msg)

    def send_error(self, message: Message):
        self.send(Message.Error.value + message.value)

    def receive(self) -> bytes:
        msg = _receive(self.request)
        return msg

    def finish(self) -> None:
        self.server.remove_client(self)
        super().finish()


class BaseGameServer(socketserver.ThreadingTCPServer):
    def __init__(self, thread_sleep_time, max_players, server_address, handler_class):
        super().__init__(server_address, handler_class, True)
        self.thread_sleep = thread_sleep_time
        self.max_players = max_players
        self.clients = set()
        self.games = dict()
        atexit.register(self.__del__)
        self._game_thread = _StoppableThread(target=self.__main_loop, daemon=True)
        self._game_thread.start()
        print_time(f'server started on port {server_address[1]}.')

    def __main_loop(self):
        while not self._game_thread.stopped():
            try:
                for wrapper in self.games.values():
                    game = wrapper.game
                    if len(wrapper) == self.max_players and not game.started:
                        self.start_game(wrapper)
                    elif len(wrapper) == 0:
                        print_time(f'removing game {game.id} with 0 players in it.')
                        self.remove_game(game)

                time.sleep(self.thread_sleep)
            except RuntimeError:
                pass

    def add_client(self, client: BaseRequestHandler) -> None:
        self.clients.add(client)

    def remove_client(self, client: BaseRequestHandler) -> None:
        self.clients.remove(client)
        for wrapper in self.games.values():
            if client in wrapper:
                wrapper.remove(client)
                self.broadcast(wrapper.game, Message.Error.value + Message.ErrorPlayerLeft.value)
                break

    def create_game(self, client: BaseRequestHandler, game) -> None:
        self.games[game.id] = _GameWrapper(game, [client])

    def join_game(self, client: BaseRequestHandler, game) -> None:
        wrapper = self.games[game.id]
        if len(wrapper) >= self.max_players:
            raise GameFullError(f'The game {game} is full therefore no more players can join.')
        wrapper.append(client)

    def search_games(self, custom: bool = None):
        games = [(wrapper[0].username, wrapper.game.serialize()) for wrapper in self.games.values()
                 if len(wrapper) < self.max_players]
        if custom is not None:
            return tuple(filter(lambda _, x: x.is_custom == custom, games))
        else:
            return tuple(games)

    def broadcast(self, game, msg: bytes, from_player: BaseRequestHandler = None) -> None:
        [client.send(msg) for client in self.games[game.id] if client is not from_player]

    def remove_game(self, game):
        del self.games[game.id]

    def start_game(self, wrapper: _GameWrapper) -> None:
        pass

    def update(self, game, from_player: BaseRequestHandler, move=None) -> None:
        pass

    def finish(self):
        [self.broadcast(wrapper.game, Message.Exit.value) for wrapper in self.games.values()]
        self._game_thread.stop()

    def __del__(self):
        self.finish()


class BaseGameClient(socket.socket):
    def __init__(self, username: str):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.username = username
        self.game = None

    @property
    def in_game(self):
        return self.game is not None

    def connect(self, *args, **kwargs):
        self.setblocking(True)
        super().connect(*args, **kwargs)

    def send_msg(self, msg: bytes) -> None:
        self.setblocking(True)
        _send(self, msg)

    def recv_msg(self, blocking=False) -> bytes:
        self.setblocking(blocking)
        return _receive(self)

    def join_game(self, game) -> None:
        pass

    def send_update(self, game) -> None:
        pass

    def exit(self):
        try:
            self.send_msg(Message.Exit.value)
            self.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
