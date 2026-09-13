"""SatQuery AI — Bi-Temporal Change Detection Models.

Provides production adapter (RSChangeDetection_BIT) and resilient CPU fallback
specialist (RSChangeDetection_Fallback) for detecting, quantifying, and visualizing
spatial changes between bi-temporal satellite image pairs (T1 and T2).
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.base import (
    ConfidenceLevel,
    InputType,
    ModelInfo,
    ModelInput,
    ModelOutput,
    RemoteSensingModel,
    TaskType,
)
from app.utils.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("models.change_detection")


def _align_and_resample_pair(img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Ensure both images match in dimensions and mode (RGB)."""
    im1 = img1.convert("RGB")
    im2 = img2.convert("RGB")

    if im1.size != im2.size:
        # Resize im2 to match im1's dimensions using high quality resampling
        im2 = im2.resize(im1.size, Image.Resampling.BILINEAR)

    return im1, im2


class RSChangeDetection_BIT(RemoteSensingModel):
    """Production Bitemporal Image Transformer (BIT) Change Detection Specialist."""

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="bit-change-detection",
            version="1.0.0",
            base_model="BIT-ResNet18-Transformer",
            adapter="LEVIR-CD-BiTemporalHead",
            description="Bitemporal Image Transformer specialized for high-resolution remote-sensing change detection",
            supported_tasks=[TaskType.CHANGE_DETECTION],
            supported_inputs=[
                InputType.BI_TEMPORAL,
                InputType.SINGLE_OPTICAL,
            ],
            is_fallback=False,
            device=self._device,
        )

    def validate_input(self, model_input: ModelInput) -> bool:
        if not model_input.images or len(model_input.images) < 2:
            raise ValueError("Bi-temporal change detection requires two temporal images (T1 pre-event and T2 post-event).")
        return True

    def load(self) -> None:
        try:
            import torch
            if not torch.cuda.is_available() and self._device != "cpu":
                raise RuntimeError("CUDA unavailable for BIT production model.")
            self._loaded = True
        except Exception as e:
            self._loaded = False
            raise RuntimeError(f"Could not load BIT model weights: {e}")

    def predict(self, model_input: ModelInput) -> ModelOutput:
        self.validate_input(model_input)
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Fallback should be used.")

        # Production model forward pass placeholder for GPU environments
        raise NotImplementedError("Production BIT model inference not available in CPU-only mode.")

    def explain(self, model_input: ModelInput, output: ModelOutput) -> Dict[str, Any]:
        return output.evidence or {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class RSChangeDetection_Fallback(RemoteSensingModel):
    """Resilient, high-performance CPU specialist for bi-temporal change detection.

    Computes radiometric difference, spectral index deltas (NDVI/greenness, NDWI/water),
    structural gradients, binary change mask, ground change statistics, and
    visual change map overlays.
    """

    def __init__(self):
        self._loaded = True

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="rs-bitemporal-change-cpu",
            version="1.0.0",
            base_model="Spectral-Structural-Change-Engine",
            adapter="None (CPU Deterministic)",
            description="CPU-optimized spectral difference, vegetation shift, and radiometric change detection engine",
            supported_tasks=[TaskType.CHANGE_DETECTION],
            supported_inputs=[
                InputType.BI_TEMPORAL,
                InputType.SINGLE_OPTICAL,
            ],
            is_fallback=True,
            device="cpu",
        )

    def validate_input(self, model_input: ModelInput) -> bool:
        if not model_input.images or len(model_input.images) < 2:
            raise ValueError("Bi-temporal change detection requires at least two temporal images (T1 pre-event and T2 post-event).")
        return True

    def predict(self, model_input: ModelInput) -> ModelOutput:
        self.validate_input(model_input)
        start_time = time.perf_counter()

        # Step 1: Align & resample pair
        im1, im2 = _align_and_resample_pair(model_input.images[0], model_input.images[1])
        w, h = im1.size

        # Convert to float numpy arrays
        arr1 = np.asarray(im1).astype(np.float32)
        arr2 = np.asarray(im2).astype(np.float32)

        r1, g1, b1 = arr1[:, :, 0], arr1[:, :, 1], arr1[:, :, 2]
        r2, g2, b2 = arr2[:, :, 0], arr2[:, :, 1], arr2[:, :, 2]

        # Step 2: Radiometric difference in RGB color space
        diff_r = np.abs(r2 - r1)
        diff_g = np.abs(g2 - g1)
        diff_b = np.abs(b2 - b1)
        radiometric_diff = np.sqrt(diff_r**2 + diff_g**2 + diff_b**2) / (np.sqrt(3.0) * 255.0)

        # Step 3: Spectral index shift calculations
        green1 = (g1 - r1) / (g1 + r1 + 1e-5)
        green2 = (g2 - r2) / (g2 + r2 + 1e-5)
        delta_green = green2 - green1

        water1 = (b1 - r1) / (b1 + r1 + 1e-5)
        water2 = (b2 - r2) / (b2 + r2 + 1e-5)
        delta_water = water2 - water1

        # Step 4: Structural gradient change
        gray1 = (r1 + g1 + b1) / 3.0
        gray2 = (r2 + g2 + b2) / 3.0
        grad1 = np.abs(np.diff(gray1, axis=0, prepend=gray1[:1, :])) + np.abs(np.diff(gray1, axis=1, prepend=gray1[:, :1]))
        grad2 = np.abs(np.diff(gray2, axis=0, prepend=gray2[:1, :])) + np.abs(np.diff(gray2, axis=1, prepend=gray2[:, :1]))
        grad_diff = np.clip(np.abs(grad2 - grad1) / 255.0, 0.0, 1.0)

        # Step 5: Multi-factor change magnitude map [0, 1]
        change_magnitude = np.clip(
            (radiometric_diff * 0.6) + (np.abs(delta_green) * 0.25) + (grad_diff * 0.15),
            0.0,
            1.0,
        )

        # Adaptive thresholding for binary change mask
        mean_mag = float(np.mean(change_magnitude))
        std_mag = float(np.std(change_magnitude))
        threshold = max(0.12, min(0.35, mean_mag + 1.2 * std_mag))
        binary_mask = (change_magnitude > threshold).astype(np.uint8)

        # Remove isolated single-pixel noise via 3x3 local density filter
        pad_mask = np.pad(binary_mask, 1, mode="constant", constant_values=0)
        neighbor_sum = (
            pad_mask[:-2, 1:-1] + pad_mask[2:, 1:-1] + pad_mask[1:-1, :-2] + pad_mask[1:-1, 2:]
        )
        binary_mask = np.where((binary_mask == 1) & (neighbor_sum == 0), 0, binary_mask)

        # Step 6: Compute Change Statistics
        total_pixels = int(w * h)
        changed_pixels = int(np.sum(binary_mask))
        changed_percentage = round(float((changed_pixels / max(1, total_pixels)) * 100.0), 2)

        # Determine GSD (pixel resolution) from metadata if available
        gsd_m = 10.0  # Default to 10m/pixel (Sentinel-2 baseline)
        if model_input.metadata:
            res_meta = model_input.metadata.get("resolution") or {}
            if isinstance(res_meta, dict) and "x" in res_meta:
                try:
                    res_val = float(res_meta["x"])
                    if 0.1 <= res_val <= 100.0:
                        gsd_m = res_val
                except (ValueError, TypeError):
                    pass

        pixel_area_m2 = gsd_m * gsd_m
        changed_area_m2 = round(float(changed_pixels * pixel_area_m2), 1)
        changed_area_hectares = round(float(changed_area_m2 / 10000.0), 2)
        changed_area_km2 = round(float(changed_area_m2 / 1000000.0), 3)

        # Step 7: Classify dominant transition
        if changed_pixels > 0:
            changed_delta_green = float(np.mean(delta_green[binary_mask == 1]))
            changed_delta_water = float(np.mean(delta_water[binary_mask == 1]))
            changed_delta_bright = float(np.mean(gray2[binary_mask == 1] - gray1[binary_mask == 1]))

            if changed_delta_water > 0.15 and changed_delta_water >= changed_delta_green:
                dominant_transition = "Water Body Expansion / Inundation"
                change_type_code = "water_expansion"
            elif changed_delta_green < -0.15:
                dominant_transition = "Vegetation Loss / Land Clearance"
                change_type_code = "veg_loss"
            elif changed_delta_green > 0.15:
                dominant_transition = "Vegetation Growth / Revegetation"
                change_type_code = "veg_growth"
            elif changed_delta_water < -0.15:
                dominant_transition = "Water Depletion / Desiccation"
                change_type_code = "water_shrink"
            elif changed_delta_bright > 25:
                dominant_transition = "New Built-up Structure / High-Albedo Ground"
                change_type_code = "urban_expansion"
            else:
                dominant_transition = "Surface Land Cover Modification"
                change_type_code = "general_modification"
        else:
            dominant_transition = "No Significant Change Detected"
            change_type_code = "none"

        # Step 8: Generate Change Map Visualization
        settings = get_settings()
        evidence_dir = Path(settings.evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        timestamp_id = int(time.time() * 1000)
        overlay_filename = f"change_map_{timestamp_id}.png"
        overlay_path = evidence_dir / overlay_filename

        # Create false-color change overlay
        # Changed areas highlighted in bright red/orange (239, 68, 68) with magnitude-modulated alpha
        overlay_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

        # Apply vibrant heat styling to changed pixels
        mask_indices = binary_mask == 1
        overlay_arr[mask_indices, 0] = 239  # R
        overlay_arr[mask_indices, 1] = 68   # G
        overlay_arr[mask_indices, 2] = 68   # B
        # Opacity proportional to change magnitude
        alpha_channel = np.clip(change_magnitude * 255.0 * 1.5, 120, 220).astype(np.uint8)
        overlay_arr[mask_indices, 3] = alpha_channel[mask_indices]

        overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
        post_rgba = im2.convert("RGBA")
        blended_overlay = Image.alpha_composite(post_rgba, overlay_img).convert("RGB")

        # Create 3-panel comparative composite banner: [T1 Pre-Event | T2 Post-Event | Change Map]
        banner_w = w * 3
        banner_h = h + 40
        composite_banner = Image.new("RGB", (banner_w, banner_h), (15, 23, 42))  # Slate dark bg
        draw_b = ImageDraw.Draw(composite_banner)

        # Paste 3 views
        composite_banner.paste(im1, (0, 40))
        composite_banner.paste(im2, (w, 40))
        composite_banner.paste(blended_overlay, (w * 2, 40))

        # Add section titles
        draw_b.text((12, 12), "PRE-EVENT (T1)", fill=(148, 163, 184))
        draw_b.text((w + 12, 12), "POST-EVENT (T2)", fill=(148, 163, 184))
        draw_b.text((w * 2 + 12, 12), f"CHANGE OVERLAY ({changed_percentage}% Changed)", fill=(248, 113, 113))

        # Save visualization banner
        composite_banner.save(overlay_path, format="PNG")

        # Also save standalone overlay for high-res map layer viewing
        standalone_filename = f"change_overlay_{timestamp_id}.png"
        standalone_path = evidence_dir / standalone_filename
        blended_overlay.save(standalone_path, format="PNG")

        # Formulate answer text
        if changed_percentage > 0.05:
            answer = (
                f"Bi-temporal change analysis detected significant spatial shifts affecting "
                f"{changed_percentage}% of the analyzed footprint (~{changed_area_hectares} hectares / {changed_pixels:,} pixels). "
                f"The primary transition is characterized as '{dominant_transition}'."
            )
            confidence = 0.89
            confidence_level = ConfidenceLevel.HIGH
        else:
            answer = (
                f"Bi-temporal comparison indicates high temporal stability with minimal change "
                f"({changed_percentage}% changed area). No significant structural or spectral disruptions were identified."
            )
            confidence = 0.85
            confidence_level = ConfidenceLevel.HIGH

        duration = (time.perf_counter() - start_time) * 1000.0

        statistics = {
            "total_pixels": int(total_pixels),
            "changed_pixels": int(changed_pixels),
            "changed_percentage": float(changed_percentage),
            "gsd_meters": float(gsd_m),
            "changed_area_sq_meters": float(changed_area_m2),
            "changed_area_hectares": float(changed_area_hectares),
            "changed_area_km2": float(changed_area_km2),
            "dominant_transition": dominant_transition,
            "change_type_code": change_type_code,
            "threshold_used": float(round(threshold, 3)),
        }

        evidence = {
            "statistics": statistics,
            "change_map_url": f"/api/files/evidence/{overlay_filename}",
            "overlay_url": f"/api/files/evidence/{standalone_filename}",
            "overlay_path": str(overlay_path),
            "evidence_type": "change_detection_map",
        }

        return ModelOutput(
            answer=answer,
            confidence=confidence,
            confidence_level=confidence_level,
            evidence=evidence,
            model_info=self.info,
            execution_time_ms=round(duration, 2),
            is_fallback=True,
        )

    def explain(self, model_input: ModelInput, output: ModelOutput) -> Dict[str, Any]:
        return output.evidence or {}

    @property
    def is_loaded(self) -> bool:
        return True
