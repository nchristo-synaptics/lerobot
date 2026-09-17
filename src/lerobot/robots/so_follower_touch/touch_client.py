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

"""Client for Synaptics CTS touch modules read by a Raspberry Pi Pico 2 W.

The Pico firmware (``2026-09-01-cts-touch-bringup/scripts/pico/main.py``) streams one text line per
TouchComm report over USB CDC: ``D<idx> <hex>`` for a delta frame (60 int16 image cells, 5 rows x 12
columns, followed by 17 profile values), ``I<idx> <part>`` when a module is identified, ``X<idx>`` when it
drops out. This client keeps only the latest delta image per sensor.

Source is either a serial port (default, auto-detected) or the URL of a running ``cts_web_pico.py``
server (``http://host:8765``), whose ``/latest`` endpoint exposes the same frames when the port is busy.
"""

import glob
import json
import logging
import struct
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

ROWS, COLS = 5, 12
IMAGE_CELLS = ROWS * COLS
PICO_GLOB = "/dev/serial/by-id/usb-MicroPython_Board*"
WEB_URL = "http://localhost:8765"


def find_pico_port() -> str | None:
    ports = sorted(glob.glob(PICO_GLOB))
    return ports[0] if ports else None


def find_source() -> str | None:
    """Prefer a running cts_web_pico.py (it owns the serial port), else the Pico's port itself."""
    try:
        with urllib.request.urlopen(WEB_URL + "/latest", timeout=0.3) as r:
            json.loads(r.read())["sensors"]
            return WEB_URL
    except Exception:
        return find_pico_port()


class TouchClient:
    def __init__(self, source: str | None, num_sensors: int = 2):
        self.source = source
        self.num_sensors = num_sensors
        self._frames = np.zeros((num_sensors, ROWS, COLS), dtype=np.float32)
        self._stamps = [0.0] * num_sensors
        self.present = [False] * num_sensors
        self.parts = [None] * num_sensors
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser = None

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self, wait_s: float = 3.0) -> None:
        if self.source is None:
            self.source = find_source()
            if self.source is None:
                raise ConnectionError(f"No Pico found (looked for {PICO_GLOB}); set touch_port explicitly.")
        target = self._run_http if self.source.startswith("http") else self._run_serial
        self._stop.clear()
        self._thread = threading.Thread(target=target, daemon=True, name="cts-touch")
        self._thread.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s and not all(self.present):
            time.sleep(0.05)
        missing = [i for i, p in enumerate(self.present) if not p]
        if missing:
            logger.warning(f"Touch sensors {missing} not detected on {self.source}; their frames will be zeros.")
        else:
            logger.info(f"Touch sensors connected on {self.source}: {self.parts}")

    def send(self, line: str) -> None:
        """Send one firmware command line (e.g. "R0" = re-zero sensor 0) over whichever source is in use."""
        if self.source.startswith("http"):
            urllib.request.urlopen(self.source.rstrip("/") + "/cmd?c=" + urllib.parse.quote(line), timeout=1.0).close()
        elif self._ser is not None:
            self._ser.write(line.encode() + b"\n")

    def rezero(self, wait_s: float = 3.0) -> None:
        """Fresh firmware baseline on every sensor; the module re-identifies, so wait until all are back."""
        for i in range(self.num_sensors):
            self.send(f"R{i}")
        time.sleep(0.3)
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s and not (all(self.present) and max(self.age()) < 0.2):
            time.sleep(0.05)
        if not all(self.present):
            logger.warning(f"touch re-zero: sensors {[i for i, p in enumerate(self.present) if not p]} did not come back")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def read(self) -> np.ndarray:
        """Latest delta images, shape (num_sensors, 5, 12), float32, raw sensor counts."""
        with self._lock:
            return self._frames.copy()

    def age(self) -> list[float]:
        now = time.monotonic()
        return [now - t if t else float("inf") for t in self._stamps]

    def _store(self, idx: int, cells) -> None:
        img = np.asarray(cells[:IMAGE_CELLS], dtype=np.float32).reshape(ROWS, COLS)  # as sent, row-major
        with self._lock:
            self._frames[idx] = img
            self._stamps[idx] = time.monotonic()

    def _run_serial(self) -> None:
        import serial

        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.source, 115200, timeout=0.5, exclusive=True)
            except serial.SerialException as e:
                logger.warning(f"touch serial {self.source}: {e}; retrying")
                self._stop.wait(2.0)
                continue
            ser.write(b"\x03\x04")  # restart the Pico's main.py so identify/enable run fresh
            self._ser = ser
            silent = 0
            identified = False  # saw an I<idx> line since the restart above
            restart_at = time.monotonic()
            while not self._stop.is_set():
                try:
                    line = ser.readline()
                except serial.SerialException:
                    logger.warning("touch serial dropped; reconnecting")
                    break
                if not line:
                    silent += 1
                    if silent >= 6:
                        ser.write(b"\x03\x04")
                        silent = 0
                    continue
                silent = 0
                if not identified and time.monotonic() - restart_at > 1.0:
                    # Frames are flowing but the restart never took (bytes written right after a CDC open
                    # can be dropped): send it once more so identify/enable run fresh.
                    ser.write(b"\x03\x04")
                    identified = True
                kind, _, rest = line.strip().partition(b" ")
                if len(kind) != 2 or not (0x30 <= kind[1] < 0x30 + self.num_sensors):
                    continue
                idx = kind[1] - 0x30
                try:
                    if kind[0:1] == b"D":
                        p = bytes.fromhex(rest.decode())
                        self._store(idx, struct.unpack(f"<{len(p) // 2}h", p))
                        self.present[idx] = True  # a delta frame is proof of life even without an I line
                    elif kind[0:1] == b"I":
                        identified = True
                        self.present[idx], self.parts[idx] = True, rest.decode()
                    elif kind[0:1] == b"X":
                        self.present[idx] = False
                except ValueError:
                    continue
            self._ser = None
            ser.close()

    def _run_http(self) -> None:
        url = self.source.rstrip("/") + "/latest"
        last_seq = None
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    st = json.loads(r.read())
                seq = st.get("seq")
                for idx, s in enumerate(st["sensors"][: self.num_sensors]):
                    self.present[idx], self.parts[idx] = bool(s["present"]), s["part"]
                    # Only count a frame as fresh when the server's state actually advanced,
                    # so a stalled server shows up as stale data instead of a frozen "live" frame.
                    if s["delta"] and seq != last_seq:
                        self._store(idx, s["delta"])
                last_seq = seq
            except Exception as e:  # server down or mid-restart: keep last frame, retry
                logger.warning(f"touch http {url}: {e}")
                self._stop.wait(1.0)
                continue
            self._stop.wait(0.01)
