"""Landmark layout of IOSLandmarks-1k.

Each arch has 16 tooth positions with five landmarks each. Teeth are ordered
from the right third molar to the left third molar and, within each tooth,
landmarks follow ``LANDMARK_SUFFIXES``, so that the index of a landmark is
``5 * tooth_position + suffix_position``.
"""

DENTITION_TYPES = ("00", "01", "02", "03", "04", "10", "11", "12", "13", "14")

UPPER_TEETH = (18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28)
LOWER_TEETH = (48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38)

LANDMARK_SUFFIXES = ("l", "f", "c", "m", "d")
LANDMARK_CLASSES = {
    "l": "Lingual",
    "f": "Facial",
    "c": "Cusp",
    "m": "Mesial",
    "d": "Distal",
}

NUM_TEETH = len(UPPER_TEETH)
LANDMARKS_PER_TOOTH = len(LANDMARK_SUFFIXES)
NUM_LANDMARKS = NUM_TEETH * LANDMARKS_PER_TOOTH

ARCHES = ("upper", "lower")


def landmark_names(arch: str) -> list[str]:
    """Landmark identifiers (``{FDI}_{suffix}``) of ``arch`` in order."""
    if arch == "upper":
        teeth = UPPER_TEETH
    elif arch == "lower":
        teeth = LOWER_TEETH
    else:
        raise ValueError(f"Unknown arch {arch!r}, expected upper or lower")
    return [
        f"{tooth}_{suffix}" for tooth in teeth for suffix in LANDMARK_SUFFIXES
    ]


def arch_from_scan_id(scan_id: str) -> str:
    """Arch encoded in the ``_upper`` / ``_lower`` suffix of a scan name."""
    for arch in ARCHES:
        if scan_id.endswith(f"_{arch}"):
            return arch
    raise ValueError(f"Scan name must end in _upper or _lower: {scan_id}")


def dentition_type(scan_id: str) -> str:
    """Dentition type encoded in the first two characters of a scan id."""
    prefix = scan_id[:2]
    if prefix not in DENTITION_TYPES:
        raise ValueError(f"Unknown dentition type in scan id {scan_id}")
    return prefix
