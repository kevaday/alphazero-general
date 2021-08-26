__name__ = "game2dboard"
__package__ = "game2dboard"
__version__ = "0.9.1"

try:
    from tkinter import Tk
except ImportError:
    from Tkinter import Tk

from .board import Board
from .cell import Cell
from .imagemap import ImageMap
from .outputbar import OutputBar

