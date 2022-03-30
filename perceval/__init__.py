
from pkg_resources import get_distribution

__version__ = get_distribution("perceval-quandela").version

from .components import *
from .backends import *
from .utils import *
from quandelibc import FockState
