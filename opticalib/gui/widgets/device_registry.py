"""
Device registry of CalpyGUI
===========================

Maps the entries of the ``DEVICES`` section of the configuration file to the
opticalib classes that connect them, and checks whether an entry is ready.

How the class of an entry is chosen, in order:

1. an explicit ``class:`` key in the entry (e.g. ``class: PhaseCam``);
2. the entry name, matched case-insensitively against the canonical name
   prefix of each class of the section (``phasecam6110`` -> ``PhaseCam``);
3. the section default, for sections with a single class (``CAMERAS`` ->
   ``GigaVision``, ``WFS`` -> ``Ingot``).

An entry is *ready* when a class was found, its constructor arguments can be
derived from the entry, the class reads this entry (some classes read a
fixed entry, e.g. ``PetalMirror`` always reads ``PetalDM``), and the fields
required by the class are filled in.  Otherwise the problems are reported so
the GUI can explain what to fix.

This module does not import Qt.
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

#: Legacy section names accepted as aliases of the current ones.
SECTION_ALIASES: Dict[str, str] = {"INTERFEROMETER": "INTERFEROMETERS"}

#: Variable bound to a connected device, per section.
SECTION_VARIABLES: Dict[str, str] = {
    "INTERFEROMETERS": "interf",
    "DEFORMABLE.MIRRORS": "dm",
    "CAMERAS": "cam",
    "WFS": "wfs",
    "MOTORS": "motor",
}

#: GUI kind of the devices of each section.
SECTION_KINDS: Dict[str, str] = {
    "INTERFEROMETERS": "interferometer",
    "DEFORMABLE.MIRRORS": "dm",
    "CAMERAS": "camera",
    "WFS": "wfs",
    "MOTORS": "motor",
}


@dataclass(frozen=True)
class DeviceClass:
    """
    An opticalib device class and how it is configured.

    Attributes
    ----------
    name : str
        Class name in :mod:`opticalib.devices`.
    section : str
        ``DEVICES`` subsection of its entries.
    aliases : tuple of str
        Lower-case entry-name prefixes identifying the class.
    args : str
        How the constructor arguments are derived from the entry:
        ``'suffix'`` (name suffix, e.g. the model), ``'alpao'`` (number of
        actuators from the name, or the ``serialNumber``), ``'name'`` (the
        entry name), ``'camera'`` (the ``camera`` field) or ``'none'``.
    reads : str or None
        Entry the class reads: a fixed name, ``'{suffix}'`` patterns (e.g.
        ``'PhaseCam{suffix}'``), ``'*'`` (the entry passed by name) or
        ``None`` (no configuration read).
    required : tuple of tuple of str
        Alternative groups of required fields; one group must be complete.
    optional_config : bool
        Whether the class also works without its entry.
    default : bool
        Whether the class is the default of its section.
    description : str
        Short description.
    """

    name: str
    section: str
    aliases: Tuple[str, ...]
    args: str
    reads: Optional[str]
    required: Tuple[Tuple[str, ...], ...] = ()
    optional_config: bool = False
    default: bool = False
    description: str = ""


#: Known device classes.
DEVICE_CLASSES: List[DeviceClass] = [
    DeviceClass(
        "PhaseCam",
        "INTERFEROMETERS",
        ("phasecam",),
        "suffix",
        "PhaseCam{suffix}",
        (("ip", "port"),),
        description="4D Twyman-Green PhaseCam interferometer",
    ),
    DeviceClass(
        "AccuFiz",
        "INTERFEROMETERS",
        ("accufiz",),
        "suffix",
        "AccuFiz{suffix}",
        (("ip", "port"),),
        description="4D AccuFiz Fizeau interferometer",
    ),
    DeviceClass(
        "Processer4D",
        "INTERFEROMETERS",
        ("4dprocesser",),
        "suffix",
        "4DProcesser{suffix}",
        (("ip", "port"),),
        description="4D processing virtual machine (no acquisition)",
    ),
    DeviceClass(
        "AlpaoDm",
        "DEFORMABLE.MIRRORS",
        ("alpao",),
        "alpao",
        "Alpao{suffix}",
        (("serialNumber",),),
        description="Alpao deformable mirror",
    ),
    DeviceClass(
        "PetalMirror",
        "DEFORMABLE.MIRRORS",
        ("petaldm", "petal"),
        "none",
        "PetalDM",
        tuple((f"ip{i}",) for i in range(6)),
        description="PI petal mirror",
    ),
    DeviceClass(
        "SplattDm",
        "DEFORMABLE.MIRRORS",
        ("splatt",),
        "none",
        "Splatt",
        (("ip", "port"),),
        description="SPLATT deformable mirror",
    ),
    DeviceClass(
        "DP",
        "DEFORMABLE.MIRRORS",
        ("adopticadp", "dp"),
        "none",
        "AdOpticaDP",
        optional_config=True,
        description="AdOptica Demonstration Prototype",
    ),
    DeviceClass(
        "M4AU",
        "DEFORMABLE.MIRRORS",
        ("m4au",),
        "none",
        "M4AU",
        optional_config=True,
        description="M4 adaptive unit",
    ),
    DeviceClass(
        "AdOpticaDm",
        "DEFORMABLE.MIRRORS",
        ("adopticadm", "adoptica"),
        "none",
        None,
        description="Generic AdOptica deformable mirror",
    ),
    DeviceClass(
        "GigaVision",
        "CAMERAS",
        ("gigavision", "avt"),
        "name",
        "*",
        (("id",), ("ip",)),
        default=True,
        description="Allied Vision GigE camera",
    ),
    DeviceClass(
        "Ingot",
        "WFS",
        ("ingot",),
        "camera",
        "INGOT",
        (("camera",),),
        default=True,
        description="INGOT wavefront sensor",
    ),
]


def normalize_section(section: str) -> str:
    """
    Return the canonical upper-case name of a ``DEVICES`` subsection.

    Parameters
    ----------
    section : str
        Section name as written in the YAML file.

    Returns
    -------
    str
        Canonical name, with legacy aliases resolved.
    """
    upper = str(section).upper()
    return SECTION_ALIASES.get(upper, upper)


def variable_name(section: str) -> str:
    """
    Return the variable a device of *section* is bound to.

    Parameters
    ----------
    section : str
        ``DEVICES`` subsection.

    Returns
    -------
    str
        Variable name (``'device'`` for unknown sections).
    """
    return SECTION_VARIABLES.get(normalize_section(section), "device")


def classes_for(section: str) -> List[DeviceClass]:
    """
    Return the device classes of a section.

    Parameters
    ----------
    section : str
        ``DEVICES`` subsection.

    Returns
    -------
    list of DeviceClass
        The classes (empty for sections without known classes).
    """
    section = normalize_section(section)
    return [c for c in DEVICE_CLASSES if c.section == section]


def find_class(name: str) -> Optional[DeviceClass]:
    """
    Find a device class by name, ignoring case and module prefixes.

    Parameters
    ----------
    name : str
        E.g. ``'PhaseCam'``, ``'phasecam'`` or ``'devices.PhaseCam'``.

    Returns
    -------
    DeviceClass or None
        The class, or ``None`` if unknown.
    """
    short = str(name).strip().split(".")[-1].lower()
    return next((c for c in DEVICE_CLASSES if c.name.lower() == short), None)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


@dataclass
class DeviceEntry:
    """
    One entry of the ``DEVICES`` section and how it is connected.

    Attributes
    ----------
    section : str
        Section as written in the YAML file (e.g. ``'INTERFEROMETERS'``).
    name : str
        Entry name.
    conf : dict
        Entry fields.
    device_class : DeviceClass or None
        The class connecting the entry.
    source : str
        How the class was chosen: ``'explicit'`` (``class:`` key),
        ``'name'``, ``'default'`` or ``'none'``.
    args : str or None
        Constructor arguments (``None`` when they cannot be derived).
    reads : str or None
        Entry the class will read (``None``: no configuration).
    missing : list of str
        Required fields that are not filled in.
    problems : list of str
        Why the entry cannot be connected in one click.
    notes : list of str
        Non-blocking remarks.
    """

    section: str
    name: str
    conf: Dict[str, Any]
    device_class: Optional[DeviceClass] = None
    source: str = "none"
    args: Optional[str] = None
    reads: Optional[str] = None
    missing: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique identifier of the entry in the GUI."""
        return f"cfg:{normalize_section(self.section)}:{self.name}"

    @property
    def ready(self) -> bool:
        """Whether the entry can be connected without changes."""
        return (
            self.device_class is not None
            and self.args is not None
            and not self.problems
        )

    @property
    def var_name(self) -> str:
        """Variable the device is bound to."""
        section = self.device_class.section if self.device_class else self.section
        return variable_name(section)


def _suffix(name: str, device_class: DeviceClass) -> Optional[str]:
    """Part of *name* after the longest matching alias (None if no alias matches)."""
    lower = name.lower()
    for alias in sorted(device_class.aliases, key=len, reverse=True):
        if lower.startswith(alias):
            return name[len(alias) :].strip()
    return None


def _match_by_name(section: str, name: str) -> Optional[DeviceClass]:
    lower = name.lower()
    best: Tuple[int, Optional[DeviceClass]] = (0, None)
    for device_class in classes_for(section):
        for alias in device_class.aliases:
            if lower.startswith(alias) and len(alias) > best[0]:
                best = (len(alias), device_class)
    return best[1]


def _constructor_args(
    device_class: DeviceClass, name: str, conf: Dict[str, Any]
) -> Optional[str]:
    suffix = _suffix(name, device_class) or ""
    style = device_class.args
    if style == "none":
        return ""
    if style == "name":
        return repr(name)
    if style == "suffix":
        return repr(suffix) if suffix else None
    if style == "alpao":
        if suffix.isdigit():
            return str(int(suffix))
        serial = conf.get("serialNumber")
        return f"serial_number={str(serial)!r}" if not _is_empty(serial) else None
    if style == "camera":
        camera = conf.get("camera")
        if _is_empty(camera):
            return None
        return repr(str(camera).split(":")[-1].strip())
    return None


def _entry_read(device_class: DeviceClass, name: str) -> Optional[str]:
    reads = device_class.reads
    if reads is None or reads == "*":
        return None if reads is None else name
    if "{suffix}" in reads:
        suffix = _suffix(name, device_class)
        if suffix is None:
            return reads.replace("{suffix}", "<…>")
        return reads.replace("{suffix}", suffix)
    return reads


def _argument_hint(device_class: DeviceClass) -> str:
    return {
        "suffix": (
            (
                f"name the entry {device_class.reads.replace('{suffix}', '<model>')} "
                f"(e.g. {device_class.reads.replace('{suffix}', '6110')})"
            )
            if device_class.reads
            else ""
        ),
        "alpao": "name the entry Alpao<number of actuators> (e.g. Alpao820) or fill in serialNumber",
        "camera": "fill in 'camera: CAMERAS:<camera name>'",
    }.get(device_class.args, "")


def resolve_entry(section: str, name: str, conf: Any) -> DeviceEntry:
    """
    Work out how a configuration entry is connected.

    Parameters
    ----------
    section : str
        ``DEVICES`` subsection, as written in the YAML file.
    name : str
        Entry name.
    conf : dict or None
        Entry fields.

    Returns
    -------
    DeviceEntry
        The class, the arguments and the problems of the entry.
    """
    conf = dict(conf) if isinstance(conf, dict) else {}
    entry = DeviceEntry(section=section, name=str(name), conf=conf)

    explicit = conf.get("class")
    if not _is_empty(explicit):
        entry.device_class = find_class(str(explicit))
        entry.source = "explicit"
        if entry.device_class is None:
            known = ", ".join(c.name for c in DEVICE_CLASSES)
            entry.problems.append(f"Unknown class '{explicit}' (known: {known}).")
            return entry
    else:
        entry.device_class = _match_by_name(section, entry.name)
        entry.source = "name"
        if entry.device_class is None:
            defaults = [c for c in classes_for(section) if c.default]
            if len(defaults) == 1:
                entry.device_class, entry.source = defaults[0], "default"
    if entry.device_class is None:
        entry.source = "none"
        entry.problems.append("Unknown device class: choose it in the connect dialog.")
        return entry

    device_class = entry.device_class
    entry.args = _constructor_args(device_class, entry.name, conf)
    if entry.args is None:
        entry.problems.append(
            f"Cannot build the {device_class.name} arguments: {_argument_hint(device_class)}."
        )

    entry.reads = _entry_read(device_class, entry.name)
    if (
        entry.args is not None
        and entry.reads is not None
        and entry.reads.lower() != entry.name.lower()
    ):
        message = (
            f"{device_class.name} reads the entry '{entry.reads}', not '{entry.name}'"
        )
        if device_class.optional_config:
            entry.notes.append(message + "; its settings here are ignored.")
        else:
            entry.problems.append(message + ": rename this entry.")

    if device_class.required:
        groups = [
            [f for f in group if _is_empty(conf.get(f))]
            for group in device_class.required
        ]
        best = min(groups, key=len)
        if best:
            entry.missing = best
            alternatives = (
                " or ".join(", ".join(group) for group in device_class.required)
                if len(device_class.required) > 1
                else ", ".join(best)
            )
            entry.problems.append(f"Missing: {alternatives}.")
    return entry


def build_command(entry: DeviceEntry, var_name: Optional[str] = None) -> str:
    """
    Return the command that connects *entry*.

    Parameters
    ----------
    entry : DeviceEntry
        The resolved entry.
    var_name : str, optional
        Variable to bind (defaults to the section variable).

    Returns
    -------
    str
        Python source; a commented template printing a hint when the class
        or the arguments are unknown.
    """
    var = var_name or entry.var_name
    if entry.device_class is not None and entry.args is not None:
        return (
            "import opticalib.devices as devices\n"
            f"{var} = devices.{entry.device_class.name}({entry.args})"
        )
    hint = f"Please instantiate {entry.name!r} ({entry.section}) manually."
    return (
        f"# Connect {entry.name!r} ({entry.section})\n"
        f"# Example:\n"
        f"#   import opticalib.devices as devices\n"
        f"#   {var} = devices.<ClassName>(...)\n"
        f"print({hint!r})"
    )


def list_entries(config_path: str) -> List[DeviceEntry]:
    """
    Resolve every entry of the ``DEVICES`` section of *config_path*.

    Entries are listed even when their fields are empty (e.g. the templates
    of the example configuration), so the GUI can say what is missing.

    Parameters
    ----------
    config_path : str
        Path to the ``configuration.yaml`` file.

    Returns
    -------
    list of DeviceEntry
        One entry per device, in file order.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return []
    if not isinstance(config, dict):
        return []
    entries = []
    for section, devices in (config.get("DEVICES") or {}).items():
        if not isinstance(devices, dict):
            continue
        for name, conf in devices.items():
            if conf is None or isinstance(conf, dict):
                entries.append(resolve_entry(str(section), str(name), conf))
    return entries


_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^\s#][^:#]*?)\s*:(?P<rest>.*)$")


def _split_comment(text: str) -> Tuple[str, str]:
    """Split a YAML value into (value, comment), ignoring '#' inside quotes."""
    quote = None
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i].strip(), text[i:].strip()
    return text.strip(), ""


def _parse_line(line: str) -> Optional[Tuple[int, str, str]]:
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    match = _KEY_RE.match(line.rstrip("\n"))
    if match is None:
        return None
    key = match.group("key").strip().strip("'\"")
    return len(match.group("indent")), key, match.group("rest")


def _block_end(lines: List[str], start: int, indent: int) -> int:
    """Index of the first line after the block opened at *start*."""
    for i in range(start + 1, len(lines)):
        parsed = _parse_line(lines[i])
        if parsed is not None and parsed[0] <= indent:
            return i
    return len(lines)


def _find_child(
    lines: List[str], key: str, start: int, end: int, parent_indent: int
) -> Tuple[int, int]:
    """Return (line index, indent) of *key* directly under a parent block."""
    child_indent = None
    for i in range(start, end):
        parsed = _parse_line(lines[i])
        if parsed is None:
            continue
        indent, k, _ = parsed
        if indent <= parent_indent:
            break
        if child_indent is None:
            child_indent = indent
        if indent == child_indent and k == key:
            return i, indent
    raise ValueError(f"'{key}' not found in the configuration file.")


def locate_entry(lines: List[str], section: str, name: str) -> Tuple[int, int]:
    """
    Find a ``DEVICES`` entry in the lines of a configuration file.

    Parameters
    ----------
    lines : list of str
        Lines of the YAML file.
    section : str
        ``DEVICES`` subsection, as written in the file.
    name : str
        Entry name.

    Returns
    -------
    index : int
        Index of the entry line.
    indent : int
        Indentation of the entry line.

    Raises
    ------
    ValueError
        If the entry is not found.
    """
    devices_i, devices_indent = _find_child(lines, "DEVICES", 0, len(lines), -1)
    section_i, section_indent = _find_child(
        lines,
        section,
        devices_i + 1,
        _block_end(lines, devices_i, devices_indent),
        devices_indent,
    )
    return _find_child(
        lines,
        name,
        section_i + 1,
        _block_end(lines, section_i, section_indent),
        section_indent,
    )


def set_entry_class(config_path: str, section: str, name: str, class_name: str) -> None:
    """
    Write ``class: <class_name>`` into a ``DEVICES`` entry of the file.

    Only the ``class`` line is added (or updated); the rest of the file,
    comments included, is left untouched.  The result is checked by parsing
    it before the file is written.

    Parameters
    ----------
    config_path : str
        Path to the ``configuration.yaml`` file.
    section : str
        ``DEVICES`` subsection, as written in the file.
    name : str
        Entry name.
    class_name : str
        Device class name (see :data:`DEVICE_CLASSES`).

    Raises
    ------
    ValueError
        If the entry cannot be found or the edit would not produce the
        expected configuration.
    """
    with open(config_path, "r") as f:
        lines = f.readlines()

    entry_i, entry_indent = locate_entry(lines, section, name)
    value, comment = _split_comment(_parse_line(lines[entry_i])[2])
    if value not in ("", "{}", "null", "~"):
        raise ValueError(
            f"The entry '{name}' is written inline; add 'class: {class_name}' by hand."
        )
    if value:
        # "Name: {}" -> "Name:" (keeping a trailing comment)
        head = lines[entry_i][: lines[entry_i].index(":", entry_indent) + 1]
        lines[entry_i] = head + (f" {comment}" if comment else "") + "\n"
    end = _block_end(lines, entry_i, entry_indent)
    child_indent = None
    class_line = None
    for i in range(entry_i + 1, end):
        parsed = _parse_line(lines[i])
        if parsed is None:
            continue
        if child_indent is None:
            child_indent = parsed[0]
        if parsed[0] == child_indent and parsed[1] == "class":
            class_line = i
    indent = " " * (child_indent if child_indent is not None else entry_indent + 2)
    if class_line is not None:
        _, old_comment = _split_comment(_parse_line(lines[class_line])[2])
        suffix = f"  {old_comment}" if old_comment else ""
        lines[class_line] = f"{indent}class: {class_name}{suffix}\n"
    else:
        lines.insert(entry_i + 1, f"{indent}class: {class_name}\n")

    text = "".join(lines)
    try:
        check = yaml.safe_load(text)
        written = check["DEVICES"][section][name]["class"]
    except (yaml.YAMLError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Could not add the class to the configuration entry: {exc}"
        ) from exc
    if written != class_name:
        raise ValueError("Could not add the class to the configuration entry.")
    # Atomic replacement: a crash while writing never truncates the file.
    folder = os.path.dirname(os.path.abspath(config_path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=".configuration-", suffix=".yaml", dir=folder
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if os.path.exists(config_path):
            shutil.copymode(config_path, tmp_path)
        os.replace(tmp_path, config_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
