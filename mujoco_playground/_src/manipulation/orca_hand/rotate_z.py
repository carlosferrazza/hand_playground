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
"""Rotate-z with orca hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand import base as orca_hand_base
from mujoco_playground._src.manipulation.orca_hand import orca_hand_constants as consts

# NOTE: better real-world performance by fine-tuning with higher penalties and ema_alpha ~ 0.1-0.3
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.6,
      ema_alpha=1.0,
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=3,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              angvel=1.0,
              linvel=-0.1,
              pose=0.0,
              torques=-0.0,
              energy=-0.0,
              termination=-100.0,
              action_rate=-0.0,
              other_angvel=-0.1,
          ),
      ),
      hand_type="left"
  )


class CubeRotateZAxis(orca_hand_base.OrcaHandEnv):
  """Rotate a cube around the z-axis as fast as possible wihout dropping it."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.CUBE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id

    if self._config.hand_type == "right":
      home_key = self._mj_model.keyframe("home_right")
    else:
      home_key = self._mj_model.keyframe("home_left")
    self._init_q = jp.array(home_key.qpos)
    self._default_pose = self._init_q[self._hand_qids]
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize hand qpos and qvel.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize cube qpos and qvel.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    if self._config.hand_type == "right":
      start_pos = jp.array([0.0853076, -0.180837, 0.2187]) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    else:
      start_pos = jp.array([0.00236401, -0.180987, 0.215153]) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
      )
    start_quat = orca_hand_base.uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    q_wrist = jp.array([-0.14])
    v_wrist = jp.array([0.0])

    qpos = jp.concatenate([q_cube, q_wrist, q_hand])
    qvel = jp.concatenate([v_cube, v_wrist, v_hand])
    data = mjx_env.init(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=q_hand,
        mocap_pos=jp.array([-100, -100, -100]),  # Hide goal for this task.
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_cube_angvel": jp.zeros(3),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(self._config.history_len * consts.NQ * 2)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

    motor_targets = (
        self._config.ema_alpha * motor_targets
        + (1 - self._config.ema_alpha) * state.info["motor_targets"]
    )
    
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_cube_angvel"] = self.get_cube_angvel(data)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_cube_position(data)[2] < 0.1
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    state = jp.concatenate([
        noisy_joint_angles,  # 17
        info["last_act"],  # 17
    ])  # 34
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_quat = self.get_cube_orientation(data)
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    fingertip_positions = self.get_fingertip_positions(data)
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        cube_pos_error,
        cube_quat,
        cube_angvel,
        cube_linvel,
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    return {
        "angvel": self._reward_angvel(cube_angvel, cube_pos_error),
        "linvel": self._cost_linvel(cube_linvel),
        "termination": done,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "pose": self._cost_pose(data.qpos[self._hand_qids]),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(
            data.qvel[self._hand_dqids], data.actuator_force
        ),
        "other_angvel": self._cost_other_angvel(cube_angvel),
    }

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(cube_linvel, ord=1, axis=-1)
  
  def _cost_other_angvel(self, cube_angvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(cube_angvel[:2], ord=1, axis=-1)

  def _reward_angvel(
      self, cube_angvel: jax.Array, cube_pos_error: jax.Array
  ) -> jax.Array:
    # Unconditionally maximize angvel in the z-direction.
    del cube_pos_error  # Unused.
    return cube_angvel @ jp.array([0.0, 0.0, 1.0])

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_angles - self._default_pose))


def domain_randomize(model: mjx.Model, rng: jax.Array):
  hand_type = "left"
  mj_model = CubeRotateZAxis().mj_model
  cube_geom_id = mj_model.geom("cube").id
  cube_body_id = mj_model.body("cube").id
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
  if hand_type == "right":
    hand_body_names = [
        "right_tower",
        "right_palm",
        "right_thumb_mp",
        "right_thumb_pp",
        "right_thumb_ip",
        "right_thumb_dp",
        "right_index_mp",
        "right_index_pp",
        "right_index_ip",
        "right_middle_mp",
        "right_middle_pp",
        "right_middle_ip",
        "right_ring_mp",
        "right_ring_pp",
        "right_ring_ip",
        "right_pinky_mp",
        "right_pinky_pp",
        "right_pinky_ip",
    ]
  else:
    hand_body_names = [
        "left_tower",
        "left_palm",
        "left_thumb_mp",
        "left_thumb_pp",
        "left_thumb_ip",
        "left_thumb_dp",
        "left_index_mp",
        "left_index_pp",
        "left_index_ip",
        "left_middle_mp",
        "left_middle_pp",
        "left_middle_ip",
        "left_ring_mp",
        "left_ring_pp",
        "left_ring_ip",
        "left_pinky_mp",
        "left_pinky_pp",
        "left_pinky_ip",
    ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  if hand_type == "right":
      fingertip_geoms = ["right_thumb_tip_1", "right_thumb_tip_2", "right_index_tip_1", "right_index_tip_2", "right_index_tip_3", "right_middle_tip_1", "right_middle_tip_2", "right_middle_tip_3", "right_ring_tip_1", "right_ring_tip_2", "right_ring_tip_3", "right_pinky_tip_1", "right_pinky_tip_2", "right_pinky_tip_3", "right_palm_1", "right_palm_2", "right_palm_3", "right_thumb_ph_1", "right_index_ph_1", "right_middle_ph_1", "right_middle_ph_2", "right_ring_ph_1", "right_ring_ph_2", "right_pinky_ph_1"]
  else:
    fingertip_geoms = ["left_thumb_tip_1", "left_thumb_tip_2", "left_index_tip_1", "left_index_tip_2", "left_index_tip_3", "left_middle_tip_1", "left_middle_tip_2", "left_middle_tip_3", "left_ring_tip_1", "left_ring_tip_2", "left_ring_tip_3", "left_pinky_tip_1", "left_pinky_tip_2", "left_pinky_tip_3", "left_palm_1", "left_palm_2", "left_palm_3", "left_thumb_ph_1", "left_index_ph_1", "left_middle_ph_1", "left_middle_ph_2", "left_ring_ph_1", "left_ring_ph_2", "left_pinky_ph_1"]
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

  @jax.vmap
  def rand(rng):

    # Scale size of cube: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    dsize = jax.random.uniform(key, (1,), minval=0.8, maxval=1.2)
    cube_size = model.geom_size[cube_geom_id : cube_geom_id + 1, :] * dsize
    geom_size = model.geom_size.at[
      cube_geom_id : cube_geom_id + 1, :
    ].set(cube_size)

    # Cube friction: =U(0.5, 1.0).
    rng, key = jax.random.split(rng)
    cube_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
    geom_friction = model.geom_friction.at[
        cube_geom_id : cube_geom_id + 1, 0
    ].set(cube_friction)

    # Fingertip friction: =U(0.5, 1.5).
    fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.5)
    geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
        fingertip_friction
    )

    # Scale cube mass: *U(0.8, 1.2).
    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
    cube_mass = model.body_mass[cube_body_id]
    body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass
    )
    dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(key, shape=(consts.NQ,), minval=-0.05, maxval=0.05)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(consts.NQ,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(consts.NQ,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    # Joint stiffness: *U(0.2, 1.8).
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.2, maxval=1.8 
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Joint damping: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key, (consts.NQ,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_size,
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_size,
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_size": 0,
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_size": geom_size,
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes
