"""Tests for opticalib.devices.wfs (Ingot, Pyramid) and the PupilDetector base."""

import numpy as np
import pytest

from opticalib.core import exceptions as oe
from opticalib.core import _types as ot
from opticalib.devices import wfs as wfs_mod
from opticalib.devices.wfs import Ingot, Pyramid

SHAPE = (200, 200)
RADIUS = 20

# (x, y) centers, origin='upper' (y grows downward)
INGOT_CENTERS = [(100, 50), (60, 150), (140, 150)]  # A top, B bottom-left, C bottom-right
PYR_CENTERS = [(130, 70), (70, 70), (130, 130), (70, 130)]  # A=TR, B=TL, C=BR, D=BL


class FakeCamera:
    """Deterministic camera: constant-flux discs on a zero background."""

    def __init__(self, centers, fluxes, shape=SHAPE, radius=RADIUS, name="fakecam"):
        self._name = name
        self.centers = centers
        self.fluxes = fluxes
        self.shape = shape
        self.radius = radius
        self.exptime_calls = []
        self.fps_calls = []
        self.acquire_calls = []
        self._exptime = None

    def __repr__(self):
        return f"FakeCamera({self._name})"

    def make_frame(self):
        yy, xx = np.mgrid[: self.shape[0], : self.shape[1]]
        frame = np.zeros(self.shape, dtype=float)
        for (xc, yc), f in zip(self.centers, self.fluxes):
            frame[(xx - xc) ** 2 + (yy - yc) ** 2 <= self.radius**2] = f
        return frame

    def acquire_frames(self, nframes=1):
        self.acquire_calls.append(nframes)
        return self.make_frame()

    def set_exptime(self, exptime):
        self.exptime_calls.append(exptime)
        self._exptime = exptime

    def get_exptime(self):
        return self._exptime

    def set_fps(self, fps):
        self.fps_calls.append(fps)


class CameraNoFps(FakeCamera):
    """Camera without set_fps, exposing a ``_cam.setParameter`` fallback."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.params = {}

        class _Inner:
            def setParameter(inner, key, val):
                self.params[key] = val

        self._cam = _Inner()


    def __getattribute__(self, item):
        if item == "set_fps":
            raise AttributeError(item)
        return super().__getattribute__(item)


def base_config(centers, **over):
    cfg = {
        "n_pupils": len(centers),
        "delta_r": 2,
        "sigma_threshold": 2,
        "subapertures": 2 * RADIUS,
        "pupils_centers": [list(c) for c in centers],
        "camera_binning": 1,
        "camera_base_exptime": 5,
        "camera_fps": 0,
        "camera": None,
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def patch_cfg(monkeypatch):
    """Return a function installing a config dict as the WFS device config."""

    def _set(cfg):
        monkeypatch.setattr(wfs_mod, "_gdc", lambda *a, **k: cfg)

    return _set


@pytest.fixture
def ingot(patch_cfg):
    patch_cfg(base_config(INGOT_CENTERS))
    cam = FakeCamera(INGOT_CENTERS, [1.0, 1.0, 1.0])
    return Ingot(camera=cam), cam


@pytest.fixture
def pyramid(patch_cfg):
    patch_cfg(base_config(PYR_CENTERS))
    cam = FakeCamera(PYR_CENTERS, [1.0] * 4)
    return Pyramid(camera=cam), cam


class TestFakeCamera:
    """The helper must satisfy the CameraDevice duck type."""

    def test_is_camera_device(self):
        """FakeCamera is accepted by isinstance_ as CameraDevice."""
        assert ot.isinstance_(FakeCamera([], []), "CameraDevice")

    def test_plain_object_is_not_camera(self):
        """An arbitrary object is not a CameraDevice."""
        assert not ot.isinstance_(object(), "CameraDevice")


class TestConstructor:
    """Constructor, config mapping and camera resolution."""

    @pytest.mark.parametrize("cls,centers", [(Ingot, INGOT_CENTERS), (Pyramid, PYR_CENTERS)])
    def test_attributes_from_config(self, patch_cfg, cls, centers):
        """Config values are mapped on the instance."""
        patch_cfg(base_config(centers, camera_binning=1, camera_fps=30))
        w = cls(camera=FakeCamera(centers, [1] * len(centers)))
        assert w.n_pupils == len(centers)
        assert w._dr == 2
        assert w.sigma_threshold == 2
        assert w._pupil_diameter == 40
        assert w._pupil_radius == 20
        assert w._rmin == 18 and w._rmax == 22
        assert w._camera_binning == 1
        assert w._camera_fps == 30

    @pytest.mark.parametrize("cls,centers", [(Ingot, INGOT_CENTERS), (Pyramid, PYR_CENTERS)])
    def test_pupil_info(self, patch_cfg, cls, centers):
        """pupil_info is a structured array with xc, yc, radius."""
        patch_cfg(base_config(centers))
        w = cls(camera=FakeCamera(centers, [1] * len(centers)))
        info = w.pupil_info
        assert info.dtype.names == ("xc", "yc", "radius")
        assert info.shape == (len(centers),)
        np.testing.assert_array_equal(info["xc"], [c[0] for c in centers])
        np.testing.assert_array_equal(info["yc"], [c[1] for c in centers])
        np.testing.assert_array_equal(info["radius"], [20.0] * len(centers))

    def test_pupil_info_none_without_centers(self, patch_cfg):
        """Missing pupil centers give pupil_info None."""
        patch_cfg(base_config(INGOT_CENTERS, pupils_centers=None))
        w = Ingot(camera=FakeCamera([], []))
        assert w.pupil_info is None

    def test_defaults_with_minimal_config(self, patch_cfg):
        """Only n_pupils comes from the class when config is minimal."""
        patch_cfg({"camera": "x:y"})
        w = Ingot(camera=FakeCamera([], []))
        assert w.n_pupils == 3
        assert w._camera_binning == 1
        assert w.sigma_threshold == 2
        assert w._pupil_radius == 0
        assert w._dr == 0
        assert w._pupil_diameter == 0
        assert w._camera_fps == 1
        assert w._camera_base_exptime == 1
        assert w.pupil_info is None

    def test_device_not_found_with_camera(self, monkeypatch):
        """DeviceNotFoundError from the config is swallowed if a camera is given."""

        def boom(*a, **k):
            raise oe.DeviceNotFoundError("Ingot")

        monkeypatch.setattr(wfs_mod, "_gdc", boom)
        cam = FakeCamera([], [])
        w = Ingot(camera=cam)
        assert w._camera is cam
        assert w.pupil_info is None

    @pytest.mark.parametrize("cls", [Ingot, Pyramid])
    def test_no_camera_anywhere_raises(self, patch_cfg, cls):
        """No camera and no config camera -> DeviceNotFoundError."""
        patch_cfg(base_config(INGOT_CENTERS, camera=None))
        with pytest.raises(oe.DeviceNotFoundError):
            cls()

    @pytest.mark.parametrize("cls", [Ingot, Pyramid])
    def test_no_config_no_camera_raises(self, monkeypatch, cls):
        """Config not found and no camera -> DeviceNotFoundError."""

        def boom(*a, **k):
            raise oe.DeviceNotFoundError("x")

        monkeypatch.setattr(wfs_mod, "_gdc", boom)
        with pytest.raises(oe.DeviceNotFoundError):
            cls()

    @pytest.mark.parametrize("cls", [Ingot, Pyramid])
    def test_camera_string(self, patch_cfg, monkeypatch, cls):
        """A camera given as string is built via GigaVision."""
        patch_cfg(base_config(INGOT_CENTERS))
        made = []

        def stub(name):
            made.append(name)
            return FakeCamera([], [])

        monkeypatch.setattr(wfs_mod._cam, "GigaVision", stub)
        cls(camera="mycam")
        assert made == ["mycam"]

    @pytest.mark.parametrize("cls", [Ingot, Pyramid])
    def test_camera_from_config(self, patch_cfg, monkeypatch, cls):
        """'something:camname' in the config resolves to 'camname'."""
        patch_cfg(base_config(INGOT_CENTERS, camera="CAMERAS:camname"))
        made = []

        def stub(name):
            made.append(name)
            return FakeCamera([], [])

        monkeypatch.setattr(wfs_mod._cam, "GigaVision", stub)
        cls()
        assert made == ["camname"]

    def test_camera_from_config_without_colon(self, patch_cfg, monkeypatch):
        """A plain camera name in the config is used as is."""
        patch_cfg(base_config(INGOT_CENTERS, camera="camname"))
        made = []
        monkeypatch.setattr(
            wfs_mod._cam, "GigaVision", lambda n: made.append(n) or FakeCamera([], [])
        )
        Ingot()
        assert made == ["camname"]

    def test_explicit_camera_has_priority_over_config(self, patch_cfg, monkeypatch):
        """An explicit camera object wins over config['camera']."""
        patch_cfg(base_config(INGOT_CENTERS, camera="a:b"))

        def stub(name):
            raise AssertionError("must not be called")

        monkeypatch.setattr(wfs_mod._cam, "GigaVision", stub)
        cam = FakeCamera([], [])
        assert Ingot(camera=cam)._camera is cam

    def test_invalid_camera_object_falls_back_to_config(self, patch_cfg):
        """A non-camera object is not accepted: raises without config camera."""
        patch_cfg(base_config(INGOT_CENTERS, camera=None))
        with pytest.raises(oe.DeviceNotFoundError):
            Ingot(camera=object())

    @pytest.mark.parametrize("cls", [Ingot, Pyramid])
    def test_device_not_found_message_names_device(self, patch_cfg, cls):
        """The error message should name the right WFS class."""
        patch_cfg(base_config(INGOT_CENTERS, camera=None))
        with pytest.raises(oe.DeviceNotFoundError) as ei:
            cls()
        assert cls.__name__ in str(ei.value)


class TestExptime:
    """Exposure time handling (ms in the WFS, microseconds to the camera)."""

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_initial_exptime_from_config(self, fix, request):
        """Initial exptime comes from camera_base_exptime, passed in us."""
        w, cam = request.getfixturevalue(fix)
        assert w.get_exptime() == 5
        assert cam.exptime_calls == [5000]

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_set_exptime_converts_to_us(self, fix, request):
        """set_exptime passes microseconds to the camera."""
        w, cam = request.getfixturevalue(fix)
        w.set_exptime(12)
        assert w.get_exptime() == 12
        assert cam.exptime_calls[-1] == 12000

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_set_same_exptime_no_camera_call(self, fix, request):
        """Setting an unchanged exptime does not touch the camera."""
        w, cam = request.getfixturevalue(fix)
        n = len(cam.exptime_calls)
        w.set_exptime(5)
        assert len(cam.exptime_calls) == n

    def test_pyramid_float_exptime(self, pyramid):
        """Pyramid accepts fractional milliseconds."""
        w, cam = pyramid
        w.set_exptime(0.5)
        assert cam.exptime_calls[-1] == pytest.approx(500.0)

    def test_base_exptime_zero(self, patch_cfg):
        """A base exptime of 0 is still forwarded (0 != None)."""
        patch_cfg(base_config(INGOT_CENTERS, camera_base_exptime=0))
        cam = FakeCamera([], [])
        w = Ingot(camera=cam)
        assert w.get_exptime() == 0
        assert cam.exptime_calls == [0]

    def test_missing_base_exptime_default(self, patch_cfg):
        """Without camera_base_exptime the effective default is 1 ms."""
        cfg = base_config(INGOT_CENTERS)
        del cfg["camera_base_exptime"]
        patch_cfg(cfg)
        w = Ingot(camera=FakeCamera([], []))
        assert w.get_exptime() == 1


class TestPyramidFps:
    """Pyramid frame rate handling."""

    def test_set_fps_uses_camera_method(self, pyramid):
        """set_fps forwards to camera.set_fps as float."""
        w, cam = pyramid
        w.set_fps(100)
        assert cam.fps_calls == [100.0]
        assert w.get_fps() == 100.0

    def test_set_same_fps_no_call(self, pyramid):
        """Unchanged fps is not re-sent."""
        w, cam = pyramid
        w.set_fps(50)
        w.set_fps(50)
        assert cam.fps_calls == [50.0]

    def test_set_fps_fallback_set_parameter(self, patch_cfg):
        """Cameras without set_fps use _cam.setParameter('fps', ...)."""
        patch_cfg(base_config(PYR_CENTERS))
        cam = CameraNoFps(PYR_CENTERS, [1] * 4)
        w = Pyramid(camera=cam)
        w.set_fps(25)
        assert cam.params == {"fps": 25.0}

    def test_get_fps_from_exptime_when_unset(self, pyramid):
        """With no fps configured, fps = 1/exptime."""
        w, _ = pyramid
        assert w.get_fps() == pytest.approx(1 / 5)


class TestAcquireDetector:
    """Raw detector acquisition."""

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_acquire_detector(self, fix, request):
        """Frames are passed through unmodified; nframes forwarded."""
        w, cam = request.getfixturevalue(fix)
        out = w.acquire_detector(3)
        assert cam.acquire_calls == [3]
        np.testing.assert_array_equal(out, cam.make_frame())

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_acquire_detector_default(self, fix, request):
        """Default nframes is 1."""
        w, cam = request.getfixturevalue(fix)
        w.acquire_detector()
        assert cam.acquire_calls == [1]


class TestExtractPupils:
    """PupilDetector._extract_pupils through the subclasses."""

    def test_shape_and_values_ingot(self, patch_cfg):
        """Cube shape and per-pupil flux values match the injection."""
        patch_cfg(base_config(INGOT_CENTERS))
        cam = FakeCamera(INGOT_CENTERS, [1.0, 2.0, 3.0])
        w = Ingot(camera=cam)
        cube = w._extract_pupils(cam.make_frame())
        assert cube.shape == (3, 41, 41)
        assert isinstance(cube, np.ma.MaskedArray)
        for i, f in enumerate([1.0, 2.0, 3.0]):
            vals = np.asarray(cube[i])[~np.isnan(np.asarray(cube[i]))]
            assert vals.size > 0 and np.all(vals == f)

    def test_mask_is_disc(self, ingot):
        """Mask is False inside the circle, True outside."""
        w, cam = ingot
        cube = w._extract_pupils(cam.make_frame())
        m = np.ma.getmaskarray(cube)[0]
        assert m[20, 20] == False  # noqa: E712
        assert m[0, 0] == True  # noqa: E712
        assert m.sum() < m.size
        # circular pupil: roughly pi r^2 valid pixels
        assert abs((~m).sum() - np.pi * 20**2) < 0.05 * np.pi * 20**2

    def test_pyramid_four_pupils(self, patch_cfg):
        """Pyramid returns four pupils."""
        patch_cfg(base_config(PYR_CENTERS))
        cam = FakeCamera(PYR_CENTERS, [1, 2, 3, 4])
        w = Pyramid(camera=cam)
        cube = w._extract_pupils(cam.make_frame())
        assert cube.shape == (4, 41, 41)
        for i in range(4):
            arr = np.asarray(cube[i])
            assert np.all(arr[~np.isnan(arr)] == i + 1)

    def test_pupil_at_border_fails_or_wraps(self, patch_cfg):
        """A pupil partly outside the frame cannot yield a full 41x41 crop."""
        centers = [(5, 100), (100, 100), (150, 100)]
        patch_cfg(base_config(centers))
        cam = FakeCamera(centers, [1, 1, 1])
        w = Ingot(camera=cam)
        with pytest.raises(ValueError):
            w._extract_pupils(cam.make_frame())

    def test_extract_without_pupil_info(self, patch_cfg):
        """Without pupil info extraction cannot work."""
        patch_cfg(base_config(INGOT_CENTERS, pupils_centers=None))
        w = Ingot(camera=FakeCamera([], []))
        with pytest.raises(TypeError):
            w._extract_pupils(np.zeros(SHAPE))


class TestAcquirePupils:
    """acquire_pupils: int frames vs. arrays, binning."""

    def test_int_frames_acquires(self, ingot):
        """An int triggers a camera acquisition."""
        w, cam = ingot
        cube = w.acquire_pupils(4)
        assert cam.acquire_calls == [4]
        assert cube.shape == (3, 41, 41)

    def test_array_frames_no_acquisition(self, ingot):
        """An array input is used directly, no camera call."""
        w, cam = ingot
        cube = w.acquire_pupils(cam.make_frame())
        assert cam.acquire_calls == []
        assert cube.shape == (3, 41, 41)

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_values(self, fix, request):
        """Extracted values equal the injected flux (NaN outside the disc)."""
        w, cam = request.getfixturevalue(fix)
        arr = np.asarray(w.acquire_pupils(1))
        assert np.all(arr[~np.isnan(arr)] == 1.0)
        assert np.isnan(arr).any()

    def test_binning_int_frames(self, patch_cfg):
        """Binning 2 sums 2x2 pixels; geometry is given in binned pixels."""
        centers = [(50, 25), (30, 75), (70, 75)]
        cfg = base_config(centers, camera_binning=2, subapertures=20)
        patch_cfg(cfg)
        cam = FakeCamera(centers, [1, 1, 1], shape=(100, 100), radius=10)
        # camera produces full-resolution frame; give it in 2x resolution
        cam.shape = (200, 200)
        cam.centers = [(2 * x, 2 * y) for x, y in centers]
        cam.radius = 20
        w = Ingot(camera=cam)
        cube = w.acquire_pupils(1)
        assert cube.shape == (3, 21, 21)
        # binned-by-sum: interior pixels carry 4x the unit flux
        assert np.nanmax(np.asarray(cube)) == pytest.approx(4.0)
        assert np.nanmedian(np.asarray(cube)) == pytest.approx(4.0)

    def test_binning_not_applied_to_array(self, patch_cfg):
        """Array frames are assumed already binned (no rebin)."""
        patch_cfg(base_config(INGOT_CENTERS, camera_binning=2))
        cam = FakeCamera(INGOT_CENTERS, [1, 1, 1])
        w = Ingot(camera=cam)
        cube = w.acquire_pupils(cam.make_frame())
        assert np.nanmax(np.asarray(cube)) == 1.0

    def test_acquire_pupils_detects_when_asked(self, patch_cfg):
        """detect_pupils=True replaces a wrong pupil_info with detected one."""
        wrong = [(90, 60), (50, 140), (150, 140)]
        patch_cfg(base_config(wrong))
        cam = FakeCamera(INGOT_CENTERS, [1, 1, 1])
        w = Ingot(camera=cam)
        w.acquire_pupils(cam.make_frame(), detect_pupils=True)
        xs = sorted(w.pupil_info["xc"])
        assert xs[0] == pytest.approx(60, abs=2)
        assert xs[-1] == pytest.approx(140, abs=2)


class TestSlopes:
    """Slope computation with analytic values."""

    @staticmethod
    def cube(fluxes):
        return np.stack([np.full((5, 5), float(f)) for f in fluxes])

    def test_ingot_equal_flux(self, ingot):
        """Equal flux: Sx = 0, Sy = 1/3."""
        w, _ = ingot
        sx, sy = w._compute_slopes_from_pupils(self.cube([1, 1, 1]))
        np.testing.assert_allclose(sx, 0)
        np.testing.assert_allclose(sy, 1 / 3)

    def test_ingot_bright_B(self, ingot):
        """Only B lit: Sx = +1, Sy = 0."""
        w, _ = ingot
        sx, sy = w._compute_slopes_from_pupils(self.cube([0, 1, 0]))
        np.testing.assert_allclose(sx, 1)
        np.testing.assert_allclose(sy, 0)

    def test_ingot_bright_C(self, ingot):
        """Only C lit: Sx = -1."""
        w, _ = ingot
        sx, _ = w._compute_slopes_from_pupils(self.cube([0, 0, 1]))
        np.testing.assert_allclose(sx, -1)

    def test_ingot_bright_A(self, ingot):
        """Only A lit: Sy = 1, Sx = 0."""
        w, _ = ingot
        sx, sy = w._compute_slopes_from_pupils(self.cube([1, 0, 0]))
        np.testing.assert_allclose(sx, 0)
        np.testing.assert_allclose(sy, 1)

    def test_ingot_zero_flux_masked(self, ingot):
        """Zero total flux pixels are masked, no exception."""
        w, _ = ingot
        sx, sy = w._compute_slopes_from_pupils(self.cube([0, 0, 0]))
        assert isinstance(sx, np.ma.MaskedArray)
        assert sx.mask.all() and sy.mask.all()

    def test_ingot_partial_zero_flux_masked(self, ingot):
        """Only the zero-flux pixel is masked."""
        w, _ = ingot
        c = self.cube([1, 1, 1])
        c[:, 0, 0] = 0
        sx, sy = w._compute_slopes_from_pupils(c)
        assert sx.mask[0, 0] and sy.mask[0, 0]
        assert sx.mask.sum() == 1

    def test_ingot_nan_input_masked(self, ingot):
        """NaN input pixels propagate into the mask."""
        w, _ = ingot
        c = self.cube([1, 1, 1])
        c[1, 2, 2] = np.nan
        sx, sy = w._compute_slopes_from_pupils(c)
        assert sx.mask[2, 2] and sy.mask[2, 2]

    def test_ingot_masked_input(self, ingot):
        """Masked (outside-disc) pixels stay masked in the output."""
        w, cam = ingot
        cube = w._extract_pupils(cam.make_frame())
        sx, sy = w._compute_slopes_from_pupils(cube)
        assert np.ma.getmaskarray(sx)[0, 0]
        assert not np.ma.getmaskarray(sx)[20, 20]

    def test_ingot_does_not_change_global_seterr(self, ingot):
        """Computing slopes should not leave numpy error state altered."""
        w, _ = ingot
        old = np.seterr(invalid="warn")
        try:
            w._compute_slopes_from_pupils(self.cube([1, 1, 1]))
            assert np.geterr()["invalid"] == "warn"
        finally:
            np.seterr(**old)

    def test_pyramid_equal_flux(self, pyramid):
        """Equal flux: both slopes zero."""
        w, _ = pyramid
        sx, sy = w._compute_slopes_from_pupils(self.cube([1, 1, 1, 1]))
        np.testing.assert_allclose(sx, 0)
        np.testing.assert_allclose(sy, 0)

    @pytest.mark.parametrize(
        "fluxes,sx,sy",
        [
            ([1, 0, 0, 0], 1, -1),  # A (TR)
            ([0, 1, 0, 0], 1, 1),  # B (TL)
            ([0, 0, 1, 0], -1, -1),  # C (BR)
            ([0, 0, 0, 1], -1, 1),  # D (BL)
        ],
    )
    def test_pyramid_single_pupil(self, pyramid, fluxes, sx, sy):
        """Single lit pupil gives +/-1 slopes."""
        w, _ = pyramid
        x, y = w._compute_slopes_from_pupils(self.cube(fluxes))
        np.testing.assert_allclose(x, sx)
        np.testing.assert_allclose(y, sy)

    def test_pyramid_zero_flux_no_crash_and_masked(self, pyramid):
        """Zero total flux should give masked output (like Ingot)."""
        w, _ = pyramid
        sx, sy = w._compute_slopes_from_pupils(self.cube([0, 0, 0, 0]))
        assert np.ma.getmaskarray(sx).all()
        assert np.ma.getmaskarray(sy).all()


class TestAcquireMap:
    """acquire_map outputs."""

    def test_ingot_slopes(self, patch_cfg):
        """Slopes output stacks Sx over Sy."""
        patch_cfg(base_config(INGOT_CENTERS))
        cam = FakeCamera(INGOT_CENTERS, [1.0, 2.0, 1.0])
        w = Ingot(camera=cam)
        out = w.acquire_map()
        assert out.shape == (82, 41)
        sx, sy = out[:41], out[41:]
        # A=1, B=2, C=1: Sx = 1/4, Sy = 1/4
        assert np.ma.allclose(sx[20, 20], 0.25)
        assert np.ma.allclose(sy[20, 20], 0.25)

    def test_ingot_pupils(self, ingot):
        """pupils output returns the pupil cube."""
        w, _ = ingot
        out = w.acquire_map(output_type="pupils")
        assert out.shape == (3, 41, 41)

    @pytest.mark.parametrize("fix", ["ingot", "pyramid"])
    def test_slopes_keep_mask(self, fix, request):
        """The stacked slopes preserve the mask (outside the disc)."""
        w, _ = request.getfixturevalue(fix)
        out = w.acquire_map()
        m = np.ma.getmaskarray(out)
        assert m[0, 0] and m[41, 0]
        assert not m[20, 20] and not m[41 + 20, 20]

    def test_pyramid_slopes(self, patch_cfg):
        """Pyramid slopes for a known flux pattern."""
        patch_cfg(base_config(PYR_CENTERS))
        cam = FakeCamera(PYR_CENTERS, [4.0, 2.0, 1.0, 1.0])
        w = Pyramid(camera=cam)
        out = w.acquire_map()
        assert out.shape == (82, 41)
        # A=4, B=2, C=1, D=1, I=8
        # Sx = (A+B-C-D)/I = 0.5 ; Sy = (B+D-A-C)/I = -0.25
        assert np.ma.allclose(out[:41][20, 20], 0.5)
        assert np.ma.allclose(out[41:][20, 20], -0.25)

    def test_pyramid_pupils(self, pyramid):
        """Pyramid pupils output."""
        w, _ = pyramid
        out = w.acquire_map(output_type="pupils")
        assert out.shape == (4, 41, 41)

    def test_nframes_forwarded(self, ingot):
        """nframes goes to the camera."""
        w, cam = ingot
        w.acquire_map(nframes=7)
        assert cam.acquire_calls == [7]

    def test_acquire_map_binning(self, patch_cfg):
        """With binning, the frame is rebinned (sum) before extraction."""
        centers = [(50, 25), (30, 75), (70, 75)]
        patch_cfg(base_config(centers, camera_binning=2, subapertures=20))
        cam = FakeCamera(centers, [1, 1, 1], shape=(200, 200), radius=20)
        cam.centers = [(2 * x, 2 * y) for x, y in centers]
        w = Ingot(camera=cam)
        out = w.acquire_map(output_type="pupils")
        assert out.shape == (3, 21, 21)
        assert np.nanmax(np.asarray(out)) == pytest.approx(4.0)

    def test_unknown_output_type_returns_pupils(self, ingot):
        """Any output_type other than 'slopes' yields the pupils."""
        w, _ = ingot
        assert w.acquire_map(output_type="whatever").shape == (3, 41, 41)

    def test_acquire_map_with_detection(self, patch_cfg):
        """detect_pupils=True works end to end on a clean frame."""
        patch_cfg(base_config(INGOT_CENTERS))
        cam = FakeCamera(INGOT_CENTERS, [1, 1, 1])
        w = Ingot(camera=cam)
        out = w.acquire_map(output_type="pupils", detect_pupils=True)
        assert out.shape == (3, 41, 41)


class TestRepr:
    """String representations."""

    def test_ingot_repr(self, ingot):
        """Ingot repr is Name(camera=<camera repr>, n_pupils=N)."""
        w, _ = ingot
        assert repr(w) == "Ingot(camera=FakeCamera(fakecam), n_pupils=3)"

    def test_pyramid_repr(self, pyramid):
        """Pyramid repr is Name(camera=<camera repr>, n_pupils=N)."""
        w, _ = pyramid
        assert repr(w) == "Pyramid(camera=FakeCamera(fakecam), n_pupils=4)"


@pytest.mark.slow
class TestDetectPupils:
    """Hough-based pupil detection on synthetic clean frames."""

    @staticmethod
    def make(cls, patch_cfg, centers, wrong=None):
        patch_cfg(base_config(wrong or centers, n_pupils=len(centers)))
        cam = FakeCamera(centers, [1.0] * len(centers))
        return cls(camera=cam), cam

    def test_detect_with_config_radii(self, patch_cfg):
        """_detect_pupils works with integer radii straight from the config."""
        patch_cfg(base_config(INGOT_CENTERS))
        cam = FakeCamera(INGOT_CENTERS, [1.0] * 3)
        w = Ingot(camera=cam)
        assert isinstance(w._rmin, int) and isinstance(w._rmax, int)
        w._detect_pupils(cam.make_frame(), pupil_order="+x+y")
        assert len(w.pupil_info) == 3

    def test_ingot_detect_centers(self, patch_cfg):
        """Detected centers and radii match the injected ones."""
        w, cam = self.make(Ingot, patch_cfg, INGOT_CENTERS)
        w._detect_pupils(cam.make_frame(), pupil_order="+x+y")
        info = w.pupil_info
        assert len(info) == 3
        got = sorted(zip(info["xc"], info["yc"]))
        exp = sorted(INGOT_CENTERS)
        for (gx, gy), (ex, ey) in zip(got, exp):
            assert gx == pytest.approx(ex, abs=2)
            assert gy == pytest.approx(ey, abs=2)
        np.testing.assert_allclose(info["radius"], 20, atol=2)

    def test_pyramid_detect_centers(self, patch_cfg):
        """Four pupils are detected within 2 px."""
        w, cam = self.make(Pyramid, patch_cfg, PYR_CENTERS)
        w._detect_pupils(cam.make_frame(), pupil_order="-x+y")
        info = w.pupil_info
        assert len(info) == 4
        got = sorted(zip(info["xc"], info["yc"]))
        exp = sorted(PYR_CENTERS)
        for (gx, gy), (ex, ey) in zip(got, exp):
            assert gx == pytest.approx(ex, abs=2)
            assert gy == pytest.approx(ey, abs=2)

    def test_too_few_pupils_raises(self, patch_cfg):
        """Frame with fewer pupils than expected -> RuntimeError."""
        patch_cfg(base_config(PYR_CENTERS))
        cam = FakeCamera(PYR_CENTERS[:2], [1.0, 1.0])
        w = Pyramid(camera=cam)
        with pytest.raises(RuntimeError, match="Not enough pupils"):
            w._detect_pupils(cam.make_frame(), pupil_order="-x+y")

    def test_unsupported_order_raises(self, patch_cfg):
        """Unsupported pupil_order -> ValueError."""
        w, cam = self.make(Ingot, patch_cfg, INGOT_CENTERS)
        with pytest.raises(ValueError, match="Unsupported pupil_order"):
            w._detect_pupils(cam.make_frame(), pupil_order="xy")

    @pytest.mark.parametrize("order", ["+y-x", "x+y+", "", "+y+y+x"])
    def test_malformed_order_raises(self, patch_cfg, order):
        """Malformed orders are rejected."""
        w, cam = self.make(Ingot, patch_cfg, INGOT_CENTERS)
        with pytest.raises(ValueError):
            w._detect_pupils(cam.make_frame(), pupil_order=order)

    def test_ingot_default_order_matches_abc(self, patch_cfg):
        """Default '+x+y' gives A top, B bottom-left, C bottom-right."""
        w, cam = self.make(Ingot, patch_cfg, INGOT_CENTERS)
        w._detect_pupils(cam.make_frame(), pupil_order="+x+y")
        got = [(round(x / 10) * 10, round(y / 10) * 10) for x, y in
               zip(w.pupil_info["xc"], w.pupil_info["yc"])]
        assert got == [(100, 50), (60, 150), (140, 150)]

    def test_pyramid_default_order_matches_docstring(self, patch_cfg):
        """Default '-x+y' gives A=TR, B=TL, C=BR, D=BL."""
        w, cam = self.make(Pyramid, patch_cfg, PYR_CENTERS)
        w._detect_pupils(cam.make_frame(), pupil_order="-x+y")
        got = [(round(x / 10) * 10, round(y / 10) * 10) for x, y in
               zip(w.pupil_info["xc"], w.pupil_info["yc"])]
        assert got == PYR_CENTERS

    @pytest.mark.parametrize(
        "order", ["+y+x", "+x+y", "-y+x", "-x+y", "-y-x", "-x-y"]
    )
    def test_order_semantics(self, patch_cfg, order):
        """The LAST token is the primary sort key, the first breaks ties.

        y is the array row index (grows downward). Checked on a 2x2 grid.
        """
        w, cam = self.make(Pyramid, patch_cfg, PYR_CENTERS)
        w._detect_pupils(cam.make_frame(), pupil_order=order)
        got = [(round(x / 10) * 10, round(y / 10) * 10) for x, y in
               zip(w.pupil_info["xc"], w.pupil_info["yc"])]
        secondary, primary = order[:2], order[2:]

        def sk(p):
            d = {"x": p[0], "y": p[1]}
            s1 = d[primary[1]] * (1 if primary[0] == "+" else -1)
            s2 = d[secondary[1]] * (1 if secondary[0] == "+" else -1)
            return (s1, s2)

        assert got == sorted(PYR_CENTERS, key=sk)

    def test_detect_warns_on_radius_limit(self, patch_cfg):
        """A wrong expected radius triggers the range-limit warning."""
        patch_cfg(base_config(INGOT_CENTERS, subapertures=30, delta_r=0))
        cam = FakeCamera(INGOT_CENTERS, [1.0] * 3)
        w = Ingot(camera=cam)
        with pytest.warns(UserWarning, match="search range"):
            try:
                w._detect_pupils(cam.make_frame(), pupil_order="+x+y")
            except RuntimeError:
                pass
