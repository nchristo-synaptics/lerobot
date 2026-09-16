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

from dataclasses import dataclass

from ..config import RobotConfig
from ..so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("so101_follower_touch")
@dataclass
class SOFollowerTouchConfig(RobotConfig, SOFollowerConfig):
    """SO-101 follower plus Synaptics CTS touch modules read through a Raspberry Pi Pico 2 W."""

    # Serial port of the Pico (auto-detected from /dev/serial/by-id when None) or the URL of a running
    # cts_web_pico.py server, e.g. "http://localhost:8765", when the web heatmap should keep the port.
    touch_port: str | None = None
    # Number of modules on the Pico (bus order = observation order).
    touch_sensors: int = 2
    # Refuse to connect unless every sensor is live (recording frozen/zero touch data is worse than failing).
    touch_required: bool = True
    # Ask the modules for a fresh baseline (CMD_REZERO) at connect, with the gripper open. The firmware keeps
    # baseline relaxation off, so this is the reference every later delta is measured against.
    touch_rezero_on_connect: bool = True
