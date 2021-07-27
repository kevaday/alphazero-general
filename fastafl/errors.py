class LoadError(Exception):
    pass


class InvalidBoardState(LoadError):
    pass


class InvalidMoveError(Exception):
    pass
