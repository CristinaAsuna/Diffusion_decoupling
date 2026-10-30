from __future__ import annotations


SIDE_CONTRACT_VERSION = "anatomical_lps_label8_left_v2"
PATIENT_LEFT_LABEL = 8
PATIENT_RIGHT_LABEL = 9
PROJECTION_CHANNELS = ("left", "right")


def validate(config: dict) -> None:
    """Reject configs that could silently recreate the legacy side swap."""
    labels = config.get("labels", {})
    left_label = int(labels.get("left", -1))
    right_label = int(labels.get("right", -1))
    if left_label == right_label:
        raise ValueError("labels.left and labels.right must differ")
    contract = config.get("side_contract", {})
    if str(contract.get("version")) != SIDE_CONTRACT_VERSION:
        raise ValueError(
            f"side_contract.version must be {SIDE_CONTRACT_VERSION!r}; "
            "legacy outputs must use a separate config/output root"
        )
    if int(contract.get("patient_left_label", -1)) != PATIENT_LEFT_LABEL:
        raise ValueError(f"side_contract.patient_left_label must be {PATIENT_LEFT_LABEL}")
    if int(contract.get("patient_right_label", -1)) != PATIENT_RIGHT_LABEL:
        raise ValueError(f"side_contract.patient_right_label must be {PATIENT_RIGHT_LABEL}")
    if tuple(contract.get("projection_channels", ())) != PROJECTION_CHANNELS:
        raise ValueError("side_contract.projection_channels must be [left, right]")
    if (left_label, right_label) != (PATIENT_LEFT_LABEL, PATIENT_RIGHT_LABEL):
        raise ValueError("labels must follow the anatomical v2 contract: left=8, right=9")


def provenance() -> dict:
    """Return the machine-readable patient-side/channel contract for manifests."""
    return {
        "version": SIDE_CONTRACT_VERSION,
        "physical_coordinate_system": "DICOM LPS millimetres",
        "normal_definition": "centroid(label8) - centroid(label9)",
        "branches": [
            {
                "name": "left",
                "patient_side": "left",
                "label": PATIENT_LEFT_LABEL,
                "plane_sign": "positive",
                "projection_channel_index": 0,
                "projection_suffix": "_left.npy",
            },
            {
                "name": "right",
                "patient_side": "right",
                "label": PATIENT_RIGHT_LABEL,
                "plane_sign": "negative",
                "projection_channel_index": 1,
                "projection_suffix": "_right.npy",
            },
        ],
        "display_orientation_is_independent": True,
    }
