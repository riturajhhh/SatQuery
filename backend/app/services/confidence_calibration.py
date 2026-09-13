"""SatQuery AI — Confidence Calibration & Uncertainty Quantification.

Provides mathematical confidence calibration, temperature scaling, entropy estimation,
and spectral radiometric quality assessment for remote-sensing vision-language inferences.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from app.models.base import ConfidenceLevel
from app.utils.logging import get_logger

logger = get_logger("services.confidence_calibration")


@dataclass
class CalibratedConfidence:
    """Rigorous uncertainty quantification output for a model analysis."""
    score: float
    level: ConfidenceLevel
    entropy: float
    consistency_score: float
    quality_score: float
    flags: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level.value,
            "entropy": round(self.entropy, 4),
            "consistency_score": round(self.consistency_score, 4),
            "quality_score": round(self.quality_score, 4),
            "flags": self.flags,
            "explanation": self.explanation,
        }


class ConfidenceCalibrator:
    """Calculates temperature-scaled and quality-adjusted confidence scores."""

    DEFAULT_TEMPERATURE = 1.10

    @classmethod
    def assess_image_quality(cls, image: Image.Image) -> Dict[str, float]:
        """Assess input raster image quality based on dynamic range, contrast, and saturation."""
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        if arr.size == 0:
            return {"contrast": 0.0, "snr": 0.0, "saturation": 0.0, "quality_score": 0.5}

        # Contrast: std deviation across luminance & inter-channel dynamic range
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        std_lum = float(np.std(luminance))
        dyn_range = float(np.ptp(arr))
        contrast_score = min(1.0, max(0.2, max(std_lum / 32.0, dyn_range / 128.0)))

        # Saturation clipping check (overexposure or underexposure)
        clipped_low = float(np.mean(arr < 5.0))
        clipped_high = float(np.mean(arr > 250.0))
        clip_penalty = (clipped_low + clipped_high) * 0.5

        # Overall raster quality score
        quality_score = max(0.3, min(1.0, (0.8 + 0.2 * contrast_score) * (1.0 - 0.5 * clip_penalty)))

        return {
            "contrast": round(contrast_score, 3),
            "std_lum": round(std_lum, 3),
            "clipped_ratio": round(clipped_low + clipped_high, 3),
            "quality_score": round(quality_score, 3),
        }

    @classmethod
    def calibrate(
        cls,
        raw_confidence: float,
        images: Optional[List[Image.Image]] = None,
        consistency_score: float = 1.0,
        temperature: float = DEFAULT_TEMPERATURE,
        task_specific_penalty: float = 0.0,
    ) -> CalibratedConfidence:
        """Calibrate raw confidence using temperature scaling, quality metrics, and evidence consistency."""
        raw_conf = max(0.01, min(0.99, raw_confidence))
        flags: List[str] = []

        # 1. Quality Assessment
        quality_score = 1.0
        if images and len(images) > 0:
            quality_metrics = cls.assess_image_quality(images[0])
            quality_score = quality_metrics["quality_score"]
            if quality_metrics["std_lum"] < 2.0 and quality_metrics["contrast"] < 0.3:
                flags.append("LOW_CONTRAST_IMAGERY")
            if quality_metrics["clipped_ratio"] > 0.40:
                flags.append("HIGH_RADIOMETRIC_CLIPPING")

        # 2. Temperature Scaling on Logit
        logit = math.log(raw_conf / (1.0 - raw_conf))
        scaled_logit = logit / max(0.1, temperature)
        scaled_conf = 1.0 / (1.0 + math.exp(-scaled_logit))

        # 3. Shannon Entropy of Binary Confidence Distribution
        p = max(1e-5, min(1.0 - 1e-5, scaled_conf))
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

        # 4. Synthesize Penalties & Consistency Scaling
        adjusted_score = scaled_conf * (0.85 + 0.15 * quality_score) * consistency_score
        adjusted_score = max(0.1, min(0.99, adjusted_score - task_specific_penalty))

        # 5. Consistency Validation Flags
        if consistency_score < 0.6:
            flags.append("SPECTRAL_EVIDENCE_MISMATCH")
        elif consistency_score < 0.85:
            flags.append("PARTIAL_EVIDENCE_DISCREPANCY")

        # 6. Categorize into Strict Confidence Levels
        if adjusted_score >= 0.85 and not any("MISMATCH" in f for f in flags):
            level = ConfidenceLevel.HIGH
            explanation = "High model confidence backed by robust spectral evidence and clear image quality."
        elif adjusted_score >= 0.70:
            level = ConfidenceLevel.MEDIUM
            explanation = "Adequate confidence with minor radiometric variance or moderate spectral alignment."
        elif adjusted_score >= 0.50:
            level = ConfidenceLevel.LOW
            flags.append("LOW_CONFIDENCE_INFERENCE")
            explanation = "Low confidence: analysis is tentative due to imaging constraints or weak spectral corroboration."
        else:
            level = ConfidenceLevel.UNCERTAIN
            flags.append("UNCERTAIN_PREDICTION")
            explanation = "Uncertain prediction: significant evidence contradiction or severe imaging degradation detected."

        return CalibratedConfidence(
            score=round(adjusted_score, 4),
            level=level,
            entropy=round(entropy, 4),
            consistency_score=round(consistency_score, 4),
            quality_score=round(quality_score, 4),
            flags=flags,
            explanation=explanation,
        )
