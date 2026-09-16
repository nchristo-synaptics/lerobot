# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Async-blend inference engine: free-running background inference with an exponential-blend
action buffer, in place of RTC's hard queue-replace.

Reproduces the board's async control loop (board_act_loop_2cam.py): a background thread infers
continuously on the latest observation with no backpressure gate; each newly completed chunk's
overlapping predictions are blended into a persistent per-absolute-step buffer (weighted toward
the newer chunk) rather than replacing a queue outright. This is a hand-rolled stand-in for ACT's
built-in temporal ensembling, which lives in ``select_action`` and needs a policy call per control
tick -- infeasible without per-tick inference latency below the control period.
"""

from __future__ import annotations

import logging
import time
import traceback
from threading import Event, Lock, Thread

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S: float = 0.01
_ERROR_RETRY_DELAY_S: float = 0.5
_MAX_CONSECUTIVE_ERRORS: int = 10
_JOIN_TIMEOUT_S: float = 3.0


class AsyncBlendInferenceEngine(InferenceEngine):
    """Continuous free-running inference with an exponential-blend action buffer.

    Unlike RTC (which replaces the queue wholesale on each new chunk), this engine keeps a
    persistent buffer keyed by absolute control step. Every newly completed chunk's predictions
    for steps that haven't executed yet are blended into that buffer -- so the action eventually
    commanded to the robot for a given step may be an average of several independently-computed
    chunks, smoothing chunk-to-chunk disagreement without needing per-tick inference.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        blend_weight: float = 0.7,
        hold_on_gap: bool = True,
        shutdown_event: Event | None = None,
    ) -> None:
        if not 0.0 < blend_weight <= 1.0:
            raise ValueError(f"blend_weight must be in (0, 1], got {blend_weight}")
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = torch.device(device or "cpu")
        self._blend_weight = blend_weight
        self._hold_on_gap = hold_on_gap
        self._global_shutdown_event = shutdown_event

        # Persistent per-absolute-step action buffer, shared between the inference thread
        # (writer, blends new chunks in) and get_action (reader/consumer, called from the main
        # control thread). step_no is "how many real actions get_action has handed out so far" --
        # advanced only by get_action, so it stays correct regardless of any interpolation layered
        # on top by the caller.
        self._buf: dict[int, torch.Tensor] = {}
        self._buf_lock = Lock()
        self._step_no = 0
        self._last_action: torch.Tensor | None = None

        self._obs_holder: dict = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._error = Event()
        self._thread: Thread | None = None

    @property
    def failed(self) -> bool:
        """True if the background thread exited due to an unrecoverable error."""
        return self._error.is_set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the free-running background inference thread."""
        self._obs_holder = {"obs": None, "step": 0}
        with self._buf_lock:
            self._buf.clear()
            self._step_no = 0
            self._last_action = None
        self._shutdown_event.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="AsyncBlendInference")
        self._thread.start()
        logger.info("Async-blend inference thread started (blend_weight=%.2f)", self._blend_weight)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        logger.info("Stopping async-blend inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("Async-blend thread did not join within %.1fs", _JOIN_TIMEOUT_S)
            else:
                logger.info("Async-blend inference thread stopped")
        self._thread = None

    def pause(self) -> None:
        """Pause the background inference thread."""
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the background inference thread."""
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action buffer."""
        logger.info("Resetting async-blend inference state (policy + processors + buffer)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._buf_lock:
            self._buf.clear()
            self._step_no = 0
            self._last_action = None

    # ------------------------------------------------------------------
    # Action production (called from the main control thread)
    # ------------------------------------------------------------------

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation, tagged with the step it corresponds to."""
        with self._obs_lock, self._buf_lock:
            self._obs_holder["obs"] = obs
            self._obs_holder["step"] = self._step_no

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the buffered action for the current step, or hold the last command on a gap."""
        with self._buf_lock:
            step = self._step_no
            action = self._buf.pop(step, None)
            self._buf.pop(step - 1, None)  # the just-passed step's entry is no longer needed
            self._step_no += 1
        if action is None:
            if self._hold_on_gap and self._last_action is not None:
                return self._last_action.clone()
            return None
        self._last_action = action
        return action

    # ------------------------------------------------------------------
    # Background inference thread
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            consecutive_errors = 0
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                    t_plan = self._obs_holder.get("step", 0)
                if obs is None:
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                try:
                    obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                    obs_batch = prepare_observation_for_inference(
                        obs_batch, self._device, self._task, self._robot.robot_type
                    )
                    obs_batch["task"] = [self._task]
                    preprocessed = self._preprocessor(obs_batch)
                    chunk = self._policy.predict_action_chunk(preprocessed)
                    processed = self._postprocessor(chunk).squeeze(0).cpu()

                    with self._buf_lock:
                        current_step = self._step_no
                        for i in range(processed.shape[0]):
                            ts = t_plan + i
                            if ts < current_step:
                                continue  # already executed by the time this chunk landed
                            new_a = processed[i]
                            if ts in self._buf:
                                self._buf[ts] = (
                                    self._blend_weight * new_a + (1 - self._blend_weight) * self._buf[ts]
                                )
                            else:
                                self._buf[ts] = new_a.clone()
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        "Async-blend inference error (%d/%d): %s",
                        consecutive_errors,
                        _MAX_CONSECUTIVE_ERRORS,
                        e,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_ERROR_RETRY_DELAY_S)
                # No throttle on success: loop straight back around and infer again on whatever
                # the latest observation is now -- "as often as possible" by construction.

        except Exception as e:
            logger.error("Fatal error in async-blend thread: %s", e)
            logger.error(traceback.format_exc())
            self._error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
