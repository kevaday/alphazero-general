#from .engine import *
import pyximport, numpy
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from .cengine import *

