class BoardGameException(Exception):
    pass


class InvalidMoveError(BoardGameException):
    pass


class GameFullError(BoardGameException):
    pass


class TurnError(BoardGameException):
    pass


class LoadError(BoardGameException):
    pass
