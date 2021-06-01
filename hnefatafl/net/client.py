from boardgame import Move
from boardgame.errors import BoardGameException, TurnError
from boardgame.net import BaseGameClient, Message, get_own_address
from hnefatafl.net import TOKEN, SERVER_ADDR, PORT
from hnefatafl.engine.game import Game, string_to_move, bool_to_colour
from hnefatafl.engine.board import Board

import pickle
import time


class Client(BaseGameClient):
    def __init__(self, username: str):
        super().__init__(username)
        self.is_white = None

    def connect(self, *args, **kwargs):
        super().connect(*args, **kwargs)
        self.send_msg(TOKEN)
        self.send_msg(self.username.encode())

    def search_games(self, custom: bool = None):
        msg = Message.PreGame.value + Message.JoinGame.value
        if custom is not None:
            msg += Message.CustomGame.value
            if custom:
                msg += Message.CustomGame.value
            else:
                msg += Message.DefaultGame.value
        else:
            msg += Message.AllGames.value

        self.send_msg(msg)
        return pickle.loads(self.recv_msg(blocking=True))

    def join_game(self, game: Game) -> None:
        self.send_msg(Message.PreGame.value + Message.JoinGame.value*2 + game.serialize())
        self.game = game

    def create_game(self, game: Game = None):
        msg = Message.PreGame.value
        if not game:
            msg += Message.DefaultGame.value
        else:
            game.board.is_custom = True
            msg += Message.CustomGame.value + game.serialize()

        self.send_msg(msg)

    def cancel_game(self):
        self.send_msg(Message.PreGame.value + Message.ErrorPlayerLeft.value)
        self.game = None

    def move(self, *args):
        if not self.game: raise ValueError('Cannot move because the client is not in game.')
        if (not self.is_white and self.game.white.is_turn) or (self.is_white and self.game.black.is_turn):
            raise TurnError(f"Attempted move being {bool_to_colour(self.is_white)} "
                            f"when it is {bool_to_colour(not self.is_white)}'s turn.")
        if isinstance(args[0], Move): move = args[0]
        else: move = Move(self.game.board, args)
        self.game.move(move)
        self.send_update(move)

    def recv_msg(self, *args, **kwargs) -> bytes:
        msg = super().recv_msg(*args, **kwargs)
        self._server_msg(msg)
        return msg

    def _server_msg(self, msg: bytes):
        if msg.startswith(Message.Game.value):
            self.game = Game.from_serial(msg[1:])
        elif msg.startswith(Message.GameUpdate.value):
            if not self.game:
                raise ValueError('Cannot update game because client does not have a game yet.')
            self.game.move(Move.from_serial(self.game.board, msg[1:]))
        elif msg.startswith(Message.Colour.value):
            self.is_white = bool(msg[1])

    def send_chat(self, text: str):
        self.send_msg(Message.Chat.value + f'{self.username}: '.encode() + text.encode())

    def send_update(self, move: Move) -> None:
        self.send_msg(Message.GameUpdate.value + move.serialize())


def msg_to_item(msg):
    if msg.startswith(Message.Exit.value):
        return

    elif msg.startswith(Message.Colour.value):
        return bool(msg[1])  # is white

    elif msg.startswith(Message.Error.value):
        return msg[1]

    elif msg.startswith(Message.GameUpdate.value):
        return Game.from_serial(msg[1:])

    elif msg.startswith(Message.Chat.value):
        return msg[1:].decode()


if __name__ == '__main__':
    import sys


    def check_yes(prompt: str) -> bool:
        return input(prompt).lower() in 'y ye yea yeah yes yep'.split()


    client = Client(input('Username: '))
    try:
        while True:
            address = input('Server address (d for default, enter for self): ').lower()
            if address == 'd':
                address = SERVER_ADDR
            elif not address:
                address = get_own_address()
            port = input('Port (enter for default): ')
            if not port:
                port = PORT

            client.connect((address, port))
            if len(sys.argv) == 2:
                client.create_game(Game(Board(load_file=sys.argv[1])))
            else:
                if check_yes('Create game? (y/n) '):
                    client.create_game()
                    print('Waiting for second player to join...')
                else:
                    games = list(map(lambda x: (x[0], Game.from_serial(x[1])), client.search_games()))
                    print('Available games: ')
                    for i, temp in enumerate(games):
                        creator, game = temp
                        print(f'{i}: By {creator}')
                        custom = game.board.is_custom
                        print(f'Custom: {custom}')
                        if custom:
                            print(game.board.to_string(add_values=True, add_spaces=True))
                    try:
                        index = int(input('Game index: '))
                    except ValueError:
                        index = 0
                    game = games[index][1]
                    client.join_game(game)

            print_colour = False
            while True:
                time.sleep(.1)
                try:
                    item = msg_to_item(client.recv_msg())
                except BlockingIOError:
                    continue

                if print_colour:
                    print(f"You are {bool_to_colour(client.is_white)}.")
                    print_colour = False

                if item is None:
                    break

                elif isinstance(item, bool):
                    client.is_white = item
                    print_colour = True

                elif isinstance(item, int):
                    print(f'Error: {Message(bytes([item]))}')

                elif isinstance(item, Game):
                    if item.game_over:
                        print(f"Game over! {'Black' if item.black.won else 'White'} won!")
                        break
                    elif (client.is_white and item.white.is_turn) or (not client.is_white and item.black.is_turn):
                        print(item.board.to_string(add_values=True, add_spaces=True))
                        while True:
                            try:
                                client.move(item, string_to_move(input('Move: ')))
                            except BoardGameException as e:
                                print(f'Invalid move: {e}')
                            else:
                                break

                    else:
                        print(f"{'Black' if item.black.is_turn else 'White'} is moving.")

                elif isinstance(item, str):
                    print(f'Chat: {item}')

    except KeyboardInterrupt:
        client.exit()
        client.close()
