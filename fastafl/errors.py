class TaflError(Exception):
    pass


class LoadError(TaflError):
    pass


class InvalidBoardState(LoadError):
    pass


class InvalidMoveError(TaflError):
    pass


class PositionError(TaflError):
    pass
