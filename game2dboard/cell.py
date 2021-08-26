from tkinter import *
import game2dboard


# Metaclass for Cell
# Implement static class properties
class CellProperties(type):
    @property
    def width(cls):
        """
        Gets cell width
        """
        return cls.size[0]

    @property
    def height(cls):
        """
        Gets cell height
        """
        return cls.size[1]


class Cell(object, metaclass=CellProperties):
    # (w, h: px) Same size for all cells, so it's a static class member
    size = (50, 50)

    def __init__(self, parent, x, y, imgpath):
        self._image_id = None       # tkinter id from create_image
        self._value = None          # this value
        self._canvas = parent
        self._x = x
        self._y = y
        self._imgpath = imgpath
        self._bgcolor = None
        self._text_color = "darkorange"
        self._id = parent.create_rectangle(
            x,
            y,
            x+Cell.width,
            y+Cell.height,
            width=0
        )

    @property
    def id(self):
        """
        Gets the rectangle id
        """
        return self._id

    @property
    def value(self):
        """
        Gets or sets the cell value.
        """
        return self._value

    @value.setter
    def value(self, v):
        if self._value != v:        # Only update when value change
            self._value = v
            if self._image_id:
                self._canvas.delete(self._image_id)    # clear current image
            if not v is None:
                img = game2dboard.ImageMap.get_instance(self._imgpath)[v]
                hc = self._x + Cell.width // 2     # horizontal center
                vc = self._y + Cell.height // 2    # vertical center
                # Show image|text @ canvas center
                if img:
                    self._image_id = self._canvas.create_image(  # Draw a image
                        hc,
                        vc,
                        anchor=CENTER,
                        image=img)
                else:
                    self._image_id = self._canvas.create_text(  # or just draw the value as text
                        hc,
                        vc,
                        anchor=CENTER,
                        text=str(v),
                        fill=self._text_color,
                        font=(None, max(min(Cell.height//4, 12), 6)),
                        width=Cell.width-2)

    @property
    def bgcolor(self):
        """
        Gets or sets the background color.
        """
        return self._bgcolor

    @bgcolor.setter
    def bgcolor(self, value):
        self._bgcolor = value
        self._canvas.itemconfig(self._id, fill=value)   # Change bg color


    @property
    def x(self):
        """
        Gets x coordinate.
        """
        return self._x

    @property
    def y(self):
        """
        Gets y coordinate.
        """
        return self._y

