"""
Tests for opticalib.dmutils.dm_analysis module.
"""

import pytest
import numpy as np
from opticalib.dmutils import get_buffer_mean_values, make_modal_base


# class TestGetBufferMeanValues:
#     """Test get_buffer_mean_values function."""

#     def _make_position_buffers(
#         self,
#         n_acts: int,
#         cmds: np.ndarray,
#         cmd_len: int = 15,
#         k: int = 5,
#         trigger_len: int = 5,
#         min_cmd: float = 1e-9,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Create synthetic position and position_error buffers matching the
#         format expected by get_buffer_mean_values.

#         The data structure contains:
#         - A 5-sample preamble of zeros
#         - 3 trigger transitions at trigger_len-sample intervals
#         - nCmds command buffers each cmd_len samples long

#         After the function removes the first 3 indices ("trigger"), the
#         remaining indices correspond exactly to the nCmds commands.

#         Parameters
#         ----------
#         n_acts : int
#             Number of actuators.
#         cmds : np.ndarray, shape (n_acts, nCmds)
#             Command values per actuator per command step.
#         cmd_len : int
#             Number of samples per command buffer (must be > k).
#         k : int
#             Settling offset used inside get_buffer_mean_values.
#         trigger_len : int
#             Samples per trigger block.
#         min_cmd : float
#             Minimum command change used to detect transitions.

#         Returns
#         -------
#         position : np.ndarray, shape (n_acts, nSteps)
#         position_error : np.ndarray, shape (n_acts, nSteps), all zeros
#         """
#         nCmds = cmds.shape[1]
#         # preamble + 3 trigger blocks + nCmds command blocks
#         n_preamble = 5
#         nSteps = n_preamble + 3 * trigger_len + nCmds * cmd_len

#         position = np.zeros((n_acts, nSteps))
#         position_error = np.zeros((n_acts, nSteps))

#         # Trigger values – small but detectable transitions distinct from cmds
#         trigger_vals = [0.5e-7, 1.0e-7, 1.5e-7]

#         for act in range(n_acts):
#             # Fill trigger blocks
#             for t, tv in enumerate(trigger_vals):
#                 start = n_preamble + t * trigger_len
#                 position[act, start : start + trigger_len] = tv

#             # Fill command blocks
#             cmd_start = n_preamble + 3 * trigger_len
#             for c in range(nCmds):
#                 start = cmd_start + c * cmd_len
#                 position[act, start : start + cmd_len] = cmds[act, c]

#         return position, position_error

#     def test_basic_output_shape(self):
#         """Test that output arrays have the expected shapes."""
#         n_acts = 3
#         nCmds = 5
#         cmds = np.random.randn(n_acts, nCmds) * 1e-6
#         position, position_error = self._make_position_buffers(n_acts, cmds)

#         posMeans, cmdIds = get_buffer_mean_values(
#             position, position_error, k=5, min_cmd=1e-9
#         )

#         assert posMeans.shape == (n_acts, nCmds)
#         assert cmdIds.shape[0] == n_acts

#     def test_mean_values_close_to_commanded(self):
#         """Test that mean position values approximate the commanded values."""
#         n_acts = 2
#         nCmds = 3
#         cmds = np.array([
#             [1e-6, 2e-6, 3e-6],
#             [0.5e-6, 1.5e-6, 2.5e-6],
#         ])
#         position, position_error = self._make_position_buffers(
#             n_acts, cmds, cmd_len=20, k=5
#         )

#         posMeans, _ = get_buffer_mean_values(
#             position, position_error, k=5, min_cmd=1e-9
#         )

#         np.testing.assert_allclose(posMeans, cmds, atol=1e-10)

#     def test_different_k_values(self):
#         """Test with a smaller k (settling offset)."""
#         n_acts = 2
#         nCmds = 3
#         cmds = np.random.randn(n_acts, nCmds) * 1e-6
#         position, position_error = self._make_position_buffers(
#             n_acts, cmds, cmd_len=20, k=3
#         )

#         posMeans, _ = get_buffer_mean_values(
#             position, position_error, k=3, min_cmd=1e-9
#         )

#         assert posMeans.shape == (n_acts, nCmds)

#     def test_multiple_actuators_exercises_chunk_logic(self):
#         """Test with more actuators than the internal chunk size (10)."""
#         n_acts = 25
#         nCmds = 3
#         cmds = np.random.randn(n_acts, nCmds) * 1e-6
#         position, position_error = self._make_position_buffers(n_acts, cmds)

#         posMeans, cmdIds = get_buffer_mean_values(
#             position, position_error, k=5, min_cmd=1e-9
#         )

#         assert posMeans.shape == (n_acts, nCmds)

#     def test_output_types_are_ndarray(self):
#         """Test that both outputs are numpy arrays."""
#         n_acts = 2
#         nCmds = 3
#         cmds = np.random.randn(n_acts, nCmds) * 1e-6
#         position, position_error = self._make_position_buffers(n_acts, cmds)

#         posMeans, cmdIds = get_buffer_mean_values(
#             position, position_error, k=5, min_cmd=1e-9
#         )

#         assert isinstance(posMeans, np.ndarray)
#         assert isinstance(cmdIds, np.ndarray)

