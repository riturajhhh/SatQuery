import numpy as np
from PIL import Image
import pytest

from app.models.base import TaskType, ConfidenceLevel
from app.services.hallucination_guard import HallucinationGuard, EvidenceConsistencyValidator


def test_consistent_vegetation_assertion():
    # Synthetic green forest scene (R=30, G=190, B=40)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 30
    arr[:, :, 1] = 190
    arr[:, :, 2] = 40
    img = Image.fromarray(arr)

    audit = HallucinationGuard.audit_and_calibrate(
        raw_answer="The area features dense vegetation and healthy forest canopy.",
        raw_confidence=0.92,
        task=TaskType.VQA,
        images=[img],
    )
    assert audit.is_consistent is True
    assert audit.consistency_score >= 0.90
    assert audit.advisory_warning is None
    assert audit.calibrated_confidence.level == ConfidenceLevel.HIGH


def test_contradictory_vegetation_assertion():
    # Synthetic red/brown barren soil scene with zero green (R=220, G=40, B=30)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 220
    arr[:, :, 1] = 40
    arr[:, :, 2] = 30
    img = Image.fromarray(arr)

    audit = HallucinationGuard.audit_and_calibrate(
        raw_answer="The region is covered by extensive forest and lush green vegetation.",
        raw_confidence=0.90,
        task=TaskType.VQA,
        images=[img],
    )
    assert audit.is_consistent is False
    assert audit.consistency_score < 0.80
    assert audit.advisory_warning is not None
    assert "Vegetation claimed, but spectral GRDI" in audit.violations[0]
    assert audit.calibrated_confidence.level in (ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN)


def test_contradictory_change_detection_assertion():
    # Answer claims major change, but evidence statistics show 0.0% change
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr)
    evidence_dict = {
        "change_detection_map": {
            "statistics": {
                "changed_percentage": 0.0,
                "changed_pixels": 0,
            }
        }
    }

    audit = HallucinationGuard.audit_and_calibrate(
        raw_answer="Observed significant change and widespread land degradation.",
        raw_confidence=0.88,
        task=TaskType.CHANGE_DETECTION,
        images=[img],
        evidence_dict=evidence_dict,
    )
    assert audit.is_consistent is False
    assert "Major change claimed in text" in audit.violations[0]
