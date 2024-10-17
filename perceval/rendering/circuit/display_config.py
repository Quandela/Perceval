# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .abstract_skin import ASkin
from .phys_skin import PhysSkin
from .symb_skin import SymbSkin
from .debug_skin import DebugSkin

from perceval.utils.persistent_data import PersistentData
from perceval.utils import get_logger

PDISPLAY_KEY = "pdisplay"
SKIN_KEY = "skin"

SKIN_MAPPING = {
    "PhysSkin": PhysSkin,
    "SymbSkin": SymbSkin,
    "DebugSkin": DebugSkin
}


def _get_default_skin():
    # Default skin is PhysSkin
    config = PersistentData().load_config()
    skin = "PhysSkin"
    if PDISPLAY_KEY in config:
        skin = config[PDISPLAY_KEY].get(SKIN_KEY, "PhysSkin")
        if skin not in SKIN_MAPPING:
            get_logger().error(f"Invalid skin in persistent configuration {skin}")
            skin = "PhysSkin"
    return SKIN_MAPPING[skin]


class DisplayConfig:
    """Handle the display configuration such as:

        - Skin use for pdisplay. Default skin is the one in the persistent data or, if no config is found, PhysSkin.
    """
    _selected_skin = _get_default_skin()

    @staticmethod
    def select_skin(skin: type[ASkin]) -> None:
        """Select the skin used by pdisplay

        :param skin: Skin to use for pdisplay
        """
        DisplayConfig._selected_skin = skin

    @staticmethod
    def get_selected_skin(**kwargs) -> ASkin:
        """Get the current selected skin

        :return: Current selected skin
        """
        return DisplayConfig._selected_skin(**kwargs)

    @staticmethod
    def save() -> None:
        """Save the current Display config in the persistent data
        """
        persistent_data = PersistentData()
        config = persistent_data.load_config()
        if PDISPLAY_KEY not in config:
            config[PDISPLAY_KEY] = {}
        config[PDISPLAY_KEY][SKIN_KEY] = DisplayConfig._selected_skin.__name__
        persistent_data.save_config(config)
