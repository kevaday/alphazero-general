class BoardgameError(Exception):
    pass


class LoadError(BoardgameError):
    pass


class InvalidBoardState(LoadError):
    pass


class InvalidMoveError(BoardgameError):
    pass


class PositionError(BoardgameError):
    pass
