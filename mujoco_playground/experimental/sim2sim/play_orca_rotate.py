# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.orca_hand import orca_hand_constants
from mujoco_playground._src.manipulation.orca_hand.base import get_assets, uniform_quat
from mujoco_playground._src.mjx_env import get_qpos_ids
from mujoco_playground._src.mjx_env import get_qvel_ids

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


def uniform_quat() -> np.ndarray:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = np.random.uniform(0, 1, 3)
  return np.array([
      np.sqrt(1 - u) * np.sin(2 * np.pi * v),
      np.sqrt(1 - u) * np.cos(2 * np.pi * v),
      np.sqrt(u) * np.sin(2 * np.pi * w),
      np.sqrt(u) * np.cos(2 * np.pi * w),
  ])



class OnnxController:
  """ONNX controller for the Leap hand."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      hand_qids: np.ndarray,
      hand_dqids: np.ndarray,
      ctrl_init: np.ndarray,
      lowers: np.ndarray,
      uppers: np.ndarray,
      history_length: int,
      n_substeps: int,
      action_scale: float = 0.5,
      ema_alpha: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._default_angles = default_angles
    self._hand_qids = hand_qids
    self._hand_dqids = hand_dqids
    self._action_scale = action_scale
    self._last_action = np.zeros_like(hand_qids, dtype=np.float32)
    self._motor_targets = ctrl_init.copy()
    self._lowers = lowers
    self._uppers = uppers
    self._ema_alpha = ema_alpha
    self._counter = 0
    self._n_substeps = n_substeps

    self.obs_buffer = np.zeros((history_length * 2 * len(hand_qids)), dtype=np.float32)


  def get_obs(self, model, data) -> np.ndarray:  # pylint: disable=unused-argument
    joint_angles = data.qpos[self._hand_qids] + (2 * np.random.uniform(0, 1, len(self._hand_qids)) - 1) * 0.05

    obs = np.hstack([
        joint_angles,
        self._last_action,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)

      # Roll the buffer
      self.obs_buffer = np.roll(self.obs_buffer, 2 * len(self._hand_qids))
      self.obs_buffer[:2 * len(self._hand_qids)] = obs

      obs = self.obs_buffer


      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      curr_ctrl = onnx_pred * self._action_scale + self._default_angles
      motor_targets = (
        self._ema_alpha * curr_ctrl
        + (1 - self._ema_alpha) * self._motor_targets
      )
      data.ctrl[:] = motor_targets
    

      self._motor_targets = data.ctrl.copy()
      self._last_action = onnx_pred.copy()


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      orca_hand_constants.CUBE_XML_MESH.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.05
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  hand_qids = get_qpos_ids(model, orca_hand_constants.JOINT_NAMES)
  hand_dqids = get_qvel_ids(model, orca_hand_constants.JOINT_NAMES)

  data.mocap_pos = np.array([-100.0, -100.0, -100.0])

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "orca_single.onnx").as_posix(),
      default_angles=np.array(model.keyframe("home").qpos[hand_qids]),
      hand_qids=hand_qids,
      hand_dqids=hand_dqids,
      n_substeps=n_substeps,
      action_scale=0.6,
      ctrl_init=data.ctrl,
      lowers=model.actuator_ctrlrange[:, 0],
      uppers=model.actuator_ctrlrange[:, 1],
      history_length=1,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
