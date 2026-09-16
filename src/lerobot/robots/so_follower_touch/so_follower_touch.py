# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_follower.so_follower import SOFollower
from .config_so_follower_touch import SOFollowerTouchConfig
from .touch_client import COLS, ROWS, TouchClient

logger = logging.getLogger(__name__)

TOUCH_KEY = "touch"


class SO101FollowerTouch(SOFollower):
    """SO-101 follower whose observation also carries the CTS touch images.

    ``observation["touch"]`` is a float32 array of shape (sensors, 5, 12) of raw delta counts (a touch is a
    negative blob). It is declared as an ENV feature of that shape, so datasets store it flattened under
    ``observation.environment_state`` and ACT feeds it to the encoder as its own token.
    """

    config_class = SOFollowerTouchConfig
    name = "so_follower"  # share calibration files with the plain SO follower

    def __init__(self, config: SOFollowerTouchConfig):
        super().__init__(config)
        self.config = config
        self.touch = TouchClient(config.touch_port, config.touch_sensors)
        self._last_stale_warn = 0.0

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | PolicyFeature]:
        touch_ft = PolicyFeature(type=FeatureType.ENV, shape=(self.config.touch_sensors, ROWS, COLS))
        return {**self._motors_ft, TOUCH_KEY: touch_ft, **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        return super().is_connected and self.touch.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self.touch.connect()
        if self.config.touch_required and not all(self.touch.present):
            super().disconnect()
            self.touch.disconnect()
            raise ConnectionError(
                f"Touch sensors {[i for i, p in enumerate(self.touch.present) if not p]} not live on "
                f"{self.touch.source}. Check the Pico (lsusb 2e8a) and cts_web_pico.log, or set touch_required=false."
            )
        if self.config.touch_rezero_on_connect:
            self.touch.rezero()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()
        obs[TOUCH_KEY] = self.touch.read()
        stale = [i for i, a in enumerate(self.touch.age()) if a > 0.5]
        if stale and time.monotonic() - self._last_stale_warn > 2.0:
            logger.warning(
                f"TOUCH DATA STALE: sensors {stale} sent no new frame for >0.5 s "
                f"(source {self.touch.source}); recorded touch values are frozen"
            )
            self._last_stale_warn = time.monotonic()
        return obs

    @check_if_not_connected
    def disconnect(self):
        super().disconnect()  # first: the base checks is_connected, which includes the touch client
        self.touch.disconnect()
