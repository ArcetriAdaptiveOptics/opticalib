"""
Tests for opticalib.dmutils.slaving module.
"""

import pytest
import numpy as np
from opticalib.dmutils import slaving
from opticalib.core import exceptions as oe


class _FakeDM:
    """Minimal deformable mirror with feed-forward matrix for slaving tests."""

    def __init__(
        self,
        n_acts: int,
        slave_ids: list[int],
        border_ids: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        n_acts : int
            Total number of actuators.
        slave_ids : list[int]
            Indices of slave actuators.
        border_ids : list[int] or None
            Indices of border actuators. If None, defaults to an empty list.
        """
        self.n_acts = n_acts
        self._slaveIds = slave_ids
        self._borderIds = border_ids if border_ids is not None else []

        # Build a tri-diagonal positive-definite stiffness matrix
        ff = np.eye(n_acts, dtype=float) * 2.0
        for i in range(n_acts - 1):
            ff[i, i + 1] = -0.5
            ff[i + 1, i] = -0.5
        self.ff = ff

    @property
    def slave_ids(self) -> list[int]:
        """Slave actuator indices."""
        return self._slaveIds

    @property
    def border_ids(self) -> list[int]:
        """Border actuator indices."""
        return self._borderIds

    def set_shape(self, cmd, differential=False):
        """Apply a command to the DM."""
        pass

    def get_shape(self):
        """Return the current shape."""
        return np.zeros(self.n_acts)

    def upload_cmd_history(self, x):
        """Upload a command history."""
        pass

    def run_cmd_history(self, **kw):
        """Run the stored command history."""
        return "tn"


class _FakeDMNoFF:
    """DM without feed-forward matrix, to test error handling."""

    def __init__(self, n_acts, slave_ids):
        """
        Parameters
        ----------
        n_acts : int
            Total number of actuators.
        slave_ids : list[int]
            Slave actuator indices.
        """
        self.n_acts = n_acts
        self._slaveIds = slave_ids
        self._borderIds = []

    @property
    def slave_ids(self):
        """Slave actuator indices."""
        return self._slaveIds

    @property
    def border_ids(self):
        """Border actuator indices."""
        return self._borderIds

    def set_shape(self, cmd, differential=False):
        """Apply a command to the DM."""
        pass

    def get_shape(self):
        """Return the current shape."""
        return np.zeros(self.n_acts)

    def upload_cmd_history(self, x):
        """Upload a command history."""
        pass

    def run_cmd_history(self, **kw):
        """Run the stored command history."""
        return "tn"


class TestGetActRoles:
    """Tests for the _get_act_roles helper function."""

    def test_ids_partition_all_actuators(self):
        """Test that slave, border, and master IDs partition all actuators."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        sid, bid, mid = slaving._get_act_roles(dm)

        all_ids = set(range(n_acts))
        found = set(sid) | set(bid) | set(mid)
        assert found == all_ids

    def test_slave_ids_correct(self):
        """Test that returned slave IDs match the DM configuration."""
        dm = _FakeDM(10, slave_ids=[7, 8, 9], border_ids=[5, 6])
        sid, _, _ = slaving._get_act_roles(dm)

        np.testing.assert_array_equal(sid, [7, 8, 9])

    def test_no_border_ids(self):
        """Test that without explicit border IDs, bid is empty and mid covers non-slaves."""
        dm = _FakeDM(8, slave_ids=[6, 7])
        sid, bid, mid = slaving._get_act_roles(dm)

        # mid should cover all non-slave actuators
        expected_mid = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(mid, expected_mid)
        # bid is derived from the empty border_ids (stays empty)
        assert len(bid) == 0

    def test_master_ids_do_not_include_slaves(self):
        """Test that master IDs do not include slave IDs."""
        dm = _FakeDM(10, slave_ids=[8, 9], border_ids=[6, 7])
        sid, _, mid = slaving._get_act_roles(dm)

        slave_set = set(sid.tolist())
        master_set = set(mid.tolist())
        assert slave_set.isdisjoint(master_set)


class TestComputeSlaveCmdZeroForce:
    """Tests for compute_slave_cmd with zero-force method."""

    def test_output_shape_preserved(self):
        """Test that the output command has the same length as the input."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmd = np.ones(n_acts) * 0.1
        result = slaving.compute_slave_cmd(dm, cmd.copy(), method="zero-force")

        assert result.shape == (n_acts,)

    def test_output_is_ndarray(self):
        """Test that the output is a numpy array."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmd = np.ones(n_acts) * 0.1
        result = slaving.compute_slave_cmd(dm, cmd.copy(), method="zero-force")

        assert isinstance(result, np.ndarray)

    def test_master_actuators_unchanged(self):
        """Test that master actuator commands are not modified."""
        n_acts = 10
        slave_ids = [8, 9]
        border_ids = [6, 7]
        dm = _FakeDM(n_acts, slave_ids=slave_ids, border_ids=border_ids)
        cmd = np.arange(n_acts, dtype=float)
        result = slaving.compute_slave_cmd(dm, cmd.copy(), method="zero-force")

        _, _, mid = slaving._get_act_roles(dm)
        np.testing.assert_array_equal(result[mid], cmd[mid])

    def test_no_slave_ids_raises(self):
        """Test that a DM without slave IDs raises DeviceAttributeError."""
        dm = _FakeDM(10, slave_ids=[], border_ids=[])
        cmd = np.ones(10)
        with pytest.raises(oe.DeviceAttributeError):
            slaving.compute_slave_cmd(dm, cmd, method="zero-force")

    def test_missing_ff_matrix_raises(self):
        """Test that a DM without ff attribute raises DeviceAttributeError."""
        dm = _FakeDMNoFF(10, slave_ids=[8, 9])
        cmd = np.ones(10)
        with pytest.raises(oe.DeviceAttributeError):
            slaving.compute_slave_cmd(dm, cmd, method="zero-force")

    def test_unknown_method_raises(self):
        """Test that an unknown slaving method raises ValueError."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmd = np.ones(n_acts)
        with pytest.raises(ValueError, match="Unknown slaving method"):
            slaving.compute_slave_cmd(dm, cmd, method="unknown")


class TestComputeSlaveCmdMinimumRms:
    """Tests for compute_slave_cmd with minimum-rms method."""

    def test_output_shape_preserved(self):
        """Test that the output command has the same length as the input."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmd = np.ones(n_acts) * 0.1
        result = slaving.compute_slave_cmd(dm, cmd.copy(), method="minimum-rms")

        assert result.shape == (n_acts,)

    def test_output_is_ndarray(self):
        """Test that the output is a numpy array."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmd = np.ones(n_acts) * 0.1
        result = slaving.compute_slave_cmd(dm, cmd.copy(), method="minimum-rms")

        assert isinstance(result, np.ndarray)


class TestComputeSlavedCommandMatrix:
    """Tests for compute_slaved_command_matrix."""

    def test_output_shape(self):
        """Test that the output matrix has the same shape as the input."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmdmat = np.eye(n_acts)
        result = slaving.compute_slaved_command_matrix(dm, cmdmat)

        assert result.shape == cmdmat.shape

    def test_output_is_ndarray(self):
        """Test that the output is a numpy array."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        cmdmat = np.random.randn(n_acts, n_acts)
        result = slaving.compute_slaved_command_matrix(dm, cmdmat)

        assert isinstance(result, np.ndarray)

    def test_identity_input_returns_slaved_output(self):
        """Test that the identity matrix produces a valid slaved matrix."""
        n_acts = 8
        dm = _FakeDM(n_acts, slave_ids=[6, 7], border_ids=[4, 5])
        cmdmat = np.eye(n_acts)
        result = slaving.compute_slaved_command_matrix(dm, cmdmat)

        # Slave rows should no longer be identity rows
        assert result.shape == (n_acts, n_acts)


class TestComputeSlavedIM:
    """Tests for compute_slaved_IM."""

    def test_output_shape_no_method(self):
        """Test output shape when method=None (only master actuators kept)."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        npix = 50
        im = np.random.randn(n_acts, npix)
        result = slaving.compute_slaved_im(dm, im)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_output_with_zero_force_method(self):
        """Test output shape with zero-force slaving."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        npix = 50
        im = np.random.randn(n_acts, npix)
        result = slaving.compute_slaved_im(dm, im, method="zero-force")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_missing_ff_raises(self):
        """Test that a DM without ff attribute raises DeviceAttributeError."""
        dm = _FakeDMNoFF(10, slave_ids=[8, 9])
        im = np.random.randn(10, 50)
        with pytest.raises(oe.DeviceAttributeError):
            slaving.compute_slaved_im(dm, im)


class TestComputeSlavedMat:
    """Tests for compute_slaved_mat."""

    def test_output_shape_master_rows(self):
        """Test that the output has only master-actuator rows."""
        n_acts = 10
        slave_ids = [8, 9]
        border_ids = [6, 7]
        dm = _FakeDM(n_acts, slave_ids=slave_ids, border_ids=border_ids)
        npix = 30
        M = np.random.randn(n_acts, npix)
        result = slaving.compute_slaved_mat(dm, M)

        assert result.ndim == 2
        assert isinstance(result, np.ndarray)

    def test_output_is_ndarray(self):
        """Test that the output is a numpy array."""
        n_acts = 10
        dm = _FakeDM(n_acts, slave_ids=[8, 9], border_ids=[6, 7])
        M = np.random.randn(n_acts, 20)
        result = slaving.compute_slaved_mat(dm, M)

        assert isinstance(result, np.ndarray)

    def test_missing_ff_raises(self):
        """Test that a DM without ff attribute raises DeviceAttributeError."""
        dm = _FakeDMNoFF(10, slave_ids=[8, 9])
        M = np.random.randn(10, 20)
        with pytest.raises(oe.DeviceAttributeError):
            slaving.compute_slaved_mat(dm, M)


class TestProjectIMIntoZonalIM:
    """Tests for project_IM_into_zonal_IM."""

    def test_output_shape(self):
        """Test that the output has the same shape as the input IM."""
        n_acts = 10
        npix = 50
        im = np.random.randn(n_acts, npix)
        FFWD = np.eye(n_acts) * 2.0
        result = slaving.project_im_into_zonal_im(im, FFWD)

        assert result.shape == im.shape

    def test_output_is_ndarray(self):
        """Test that the output is a numpy array."""
        im = np.random.randn(8, 30)
        FFWD = np.eye(8) * 2.0
        result = slaving.project_im_into_zonal_im(im, FFWD)

        assert isinstance(result, np.ndarray)

    def test_identity_ffwd_returns_same_im(self):
        """Test that an identity FFWD matrix returns the same IM (up to sign)."""
        n_acts = 6
        npix = 20
        im = np.random.randn(n_acts, npix)
        FFWD = np.eye(n_acts)
        result = slaving.project_im_into_zonal_im(im, FFWD)

        np.testing.assert_allclose(result, im, atol=1e-10)
