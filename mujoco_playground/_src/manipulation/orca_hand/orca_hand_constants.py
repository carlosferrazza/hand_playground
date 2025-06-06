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
"""Constants for leap hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "orca_hand"
CUBE_XML = ROOT_PATH / "scene_left_mjx.xml"
CUBE_XML_MESH = ROOT_PATH / "scene_left.xml"

NQ = 16
NV = 16
NU = 16

JOINT_NAMES = [
    # wrist
    # "right_wrist",
    # thumb
    "left_thumb_mcp",
    "left_thumb_abd",
    "left_thumb_pip",
    "left_thumb_dip",
    # index
    "left_index_abd",
    "left_index_mcp",
    "left_index_pip",
    # middle
    "left_middle_abd",
    "left_middle_mcp",
    "left_middle_pip",
    # ring
    "left_ring_abd",
    "left_ring_mcp",
    "left_ring_pip",
    # pinky
    "left_pinky_abd",
    "left_pinky_mcp",
    "left_pinky_pip",
]

ACTUATOR_NAMES = [
    # thumb
    "left_thumb_mcp_actuator",
    "left_thumb_abd_actuator",
    "left_thumb_pip_actuator",
    "left_thumb_dip_actuator",
    # index
    "left_index_abd_actuator",
    "left_index_mcp_actuator",
    "left_index_pip_actuator",
    # middle
    "left_middle_abd_actuator",
    "left_middle_mcp_actuator",
    "left_middle_pip_actuator",
    # ring
    "left_ring_abd_actuator",
    "left_ring_mcp_actuator",
    "left_ring_pip_actuator",
    # pinky
    "left_pinky_abd_actuator",
    "left_pinky_mcp_actuator",
    "left_pinky_pip_actuator",
]

FINGERTIP_NAMES = [
    "th_tip",
    "if_tip",
    "mf_tip",
    "rf_tip",
    "pf_tip",
]
