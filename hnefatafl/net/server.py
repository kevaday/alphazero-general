from boardgame import errors, Move, Win
from boardgame.net import BaseRequestHandler, BaseGameServer, Message, print_time, get_own_address
from hnefatafl.net import PORT, TOKEN, MAX_PLAYERS, THREAD_SLEEP
from hnefatafl.engine.game import Game, is_turn, bool_to_colour

import pickle
import random
import sys


class _RequestHandler(BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        self.__is_white = None
        super().__init__(*args, **kwargs)

    @property
    def is_white(self):
        return self.__is_white

    @is_white.setter
    def is_white(self, value):
        self.__is_white = value
        self.send(Message.Colour.value + bytes([int(self.__is_white)]))

    def handle(self) -> None:
        addr_str = f'{self.client_address[0]}:{self.client_address[1]}'
        token = self.receive()
        if token != TOKEN:
            print_time(f'{addr_str} attempted to join with invalid token: {token}')
            return

        self.username = self.receive().decode()
        if self.username in map(lambda x: x.username if x != self else None, self.server.clients):
            print_time(f'{addr_str} attempted to join with existing username {self.username}')
            self.send_error(Message.ErrorUserExists)
            return
        else:
            self.send(Message.Welcome.value)

        def client_log(text: str) -> None:
            print_time(f'{self.username} ({self.client_address[0]}:{self.client_address[1]}): {text}')

        def is_equal(b: int, m: Message) -> bool:
            return bytes([b]) == m.value

        def get_server_game():
            return self.server.games[game.id].game

        client_log('connected.')
        game = None

        # Main Loop
        while True:
            msg = self.receive()
            # client_log(f'message received: {msg}.')
            if msg == Message.Exit.value:
                client_log('disconnected.')
                break

            elif msg.startswith(Message.PreGame.value):
                if is_equal(msg[1], Message.ErrorPlayerLeft):
                    if game:
                        client_log(f'canceled game with id {game.id}')
                        self.server.cancel_game(get_server_game())
                        return

                elif is_equal(msg[1], Message.JoinGame):
                    if is_equal(msg[2], Message.JoinGame):
                        client_log('receiving game to join.')
                        try:
                            game = Game.from_serial(msg[3:])
                        except ValueError:
                            client_log('invalid game sent.')
                            self.send_error(Message.ErrorInvalidGame)
                            continue
                        else:
                            client_log(f'joining game {game.id}')
                            try:
                                self.server.join_game(self, get_server_game())
                            except errors.GameFullError:
                                client_log("tried to join game that's full.")
                                self.send_error(Message.ErrorGameFul)
                            continue

                    elif is_equal(msg[2], Message.AllGames):
                        client_log('searching for all games.')
                        games = self.server.search_games()
                    elif is_equal(msg[2], Message.CustomGame):
                        client_log('searching for custom games.')
                        games = self.server.search_games(is_equal(msg[3], Message.CustomGame))
                    else:
                        client_log('sent invalid message in PreGame; disconnecting player.')
                        self.send_error(Message.ErrorInvalidMessage)
                        break

                    client_log(f'sending client {len(games)} games to choose from.')
                    self.send(pickle.dumps(games))
                    continue

                elif is_equal(msg[1], Message.DefaultGame):
                    client_log('creating default game...')
                    try:
                        game = Game()
                    except IOError:
                        self.send_error(Message.ErrorInternal)
                elif is_equal(msg[1], Message.CustomGame):
                    client_log('creating custom game...')
                    try:
                        game = Game.from_serial(msg[2:])
                    except Exception as e:
                        client_log(f'invalid game sent. {e}')
                        self.send_error(Message.ErrorInvalidGame)
                        continue
                else:
                    client_log('sent invalid message in PreGame; disconnecting player.')
                    self.send_error(Message.ErrorInvalidMessage)
                    break

                self.server.create_game(self, game)
                client_log(f'created game with id {game.id}. '
                           f'There are now {len(self.server.games)} games running.')
                continue

            elif msg.startswith(Message.GameUpdate.value):
                try:
                    move = Move.from_serial(get_server_game().board, msg[1:])
                except ValueError:
                    client_log('invalid game update sent.')
                    self.send_error(Message.ErrorInvalidGame)
                    continue

                if is_turn(self.is_white, get_server_game()):
                    self.server.update(get_server_game(), from_player=self, move=move)

            elif msg.startswith(Message.Chat.value):
                self.server.broadcast(get_server_game(), msg)


class Server(BaseGameServer):
    def __init__(self, *args, **kwargs):
        super().__init__(THREAD_SLEEP, MAX_PLAYERS, *args, **kwargs)

    def start_game(self, wrapper) -> None:
        game = wrapper.game
        player1 = wrapper[0]
        player2 = wrapper[1]
        print_time(f'starting game {game.id}')
        player1.is_white = bool(random.randint(0, 1))
        player2.is_white = not player1.is_white
        game.start()
        self.update(game)
        msg = 'Your opponent is {} and you are {}'
        player1.send(Message.Chat.value + msg.format(player2.username, bool_to_colour(player1.is_white)).encode())
        player2.send(Message.Chat.value + msg.format(player1.username, bool_to_colour(player2.is_white)).encode())

    def cancel_game(self, game) -> None:
        del self.games[game.id][0]

    def update(self, game: Game, from_player: _RequestHandler = None, move: Move = None) -> None:
        print_time(f'updating game {game.id}')
        wrapper = self.games[game.id]
        if move:
            try: wrapper.game.move(move)
            except Win: pass
            msg = Message.GameUpdate.value + move.serialize()
        else:
            wrapper.game = game
            msg = Message.Game.value + game.serialize()
        self.broadcast(game, msg, from_player=from_player)


def get_server(port: int = None):
    if not port: port = PORT
    addr = get_own_address(), port
    return Server(addr, _RequestHandler)


if __name__ == '__main__':
    port = None
    if len(sys.argv) > 1: port = sys.argv[1]
    server = get_server(port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.finish()
