import numpy as np
from PIL import Image
import pytest

from app.models.base import ConfidenceLevel
from app.services.confidence_calibration import ConfidenceCalibrator, CalibratedConfidence


def test_confidence_calibration_high():
    # Crisp image with good dynamic range
    arr = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    calib = ConfidenceCalibrator.calibrate(
        raw_confidence=0.95,
        images=[img],
        consistency_score=1.0,
    )
    assert isinstance(calib, CalibratedConfidence)
    assert calib.level == ConfidenceLevel.HIGH
    assert calib.score >= 0.80
    assert calib.entropy >= 0.0
    assert "LOW_CONTRAST_IMAGERY" not in calib.flags


def test_confidence_calibration_low_contrast_penalty():
    # Low contrast flat gray image
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)

    calib = ConfidenceCalibrator.calibrate(
        raw_confidence=0.90,
        images=[img],
        consistency_score=1.0,
    )
    assert "LOW_CONTRAST_IMAGERY" in calib.flags
    assert calib.score < 0.90


def test_confidence_calibration_evidence_mismatch_uncertain():
    # High raw confidence, but low evidence consistency score
    arr = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    calib = ConfidenceCalibrator.calibrate(
        raw_confidence=0.90,
        images=[img],
        consistency_score=0.40,
    )
    assert "SPECTRAL_EVIDENCE_MISMATCH" in calib.flags
    assert calib.level in (ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN)
    assert calib.score < 0.60
