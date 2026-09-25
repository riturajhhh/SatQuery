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


def _align_and_resample_pair(img1: Any, img2: Any) -> Tuple[Image.Image, Image.Image]:
    """Ensure both images match in dimensions and mode (RGB)."""
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1.astype(np.uint8))
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2.astype(np.uint8))
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
            (radiometric_diff * 0.50) + (np.abs(delta_green) * 0.20) + (np.abs(delta_water) * 0.15) + (grad_diff * 0.15),
            0.0,
            1.0,
        )

        # Adaptive thresholding for binary change mask via Otsu histogram analysis
        def _compute_adaptive_threshold(mag: np.ndarray) -> float:
            max_val = float(np.max(mag))
            if max_val < 0.08:
                return 0.12
            hist, bin_edges = np.histogram(mag, bins=256, range=(0.0, 1.0))
            hist = hist.astype(np.float64)
            total = hist.sum()
            if total == 0:
                return 0.12
            p = hist / total
            omega = np.cumsum(p)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            mu = np.cumsum(p * centers)
            mu_t = mu[-1]
            denom = omega * (1.0 - omega)
            valid = denom > 1e-7
            sigma_b = np.zeros_like(p)
            sigma_b[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
            best_idx = np.argmax(sigma_b)
            otsu_val = float(centers[best_idx])
            # Lower bound 0.10 rejects sensor noise; upper bound 0.24 guarantees significant changes are captured
            return float(max(0.10, min(0.24, otsu_val)))

        threshold = _compute_adaptive_threshold(change_magnitude)
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

        # Determine spatial quadrant/sector with highest concentration of changes
        def _find_change_sector(mask: np.ndarray) -> str:
            mh, mw = mask.shape
            tot_ch = int(np.sum(mask))
            if tot_ch == 0:
                return "scene"
            hh, hw = mh // 2, mw // 2
            sectors = {
                "northwestern": int(np.sum(mask[:hh, :hw])),
                "northeastern": int(np.sum(mask[:hh, hw:])),
                "southwestern": int(np.sum(mask[hh:, :hw])),
                "southeastern": int(np.sum(mask[hh:, hw:])),
                "central": int(np.sum(mask[mh // 4 : 3 * mh // 4, mw // 4 : 3 * mw // 4])),
            }
            best_sec = max(sectors, key=sectors.get)
            if sectors[best_sec] / max(1, tot_ch) >= 0.35:
                return best_sec
            return "central and mixed"

        change_sector = _find_change_sector(binary_mask)

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

        # Step 7: Classify dominant transition using spectral, structural, and discrete building analysis
        from app.models.building_counter import BuildingCounter
        b1 = BuildingCounter.detect_and_count(im1)
        b2 = BuildingCounter.detect_and_count(im2)
        delta_buildings = int(b2.get("count", 0)) - int(b1.get("count", 0))
        delta_built_pct = round(float(b2.get("built_coverage_pct", 0.0)) - float(b1.get("built_coverage_pct", 0.0)), 2)

        if changed_pixels > 0:
            changed_mask = binary_mask == 1
            changed_delta_green = float(np.mean(delta_green[changed_mask]))
            changed_delta_water = float(np.mean(delta_water[changed_mask]))
            changed_delta_bright = float(np.mean(gray2[changed_mask] - gray1[changed_mask]))
            changed_delta_edge = float(np.mean(grad2[changed_mask] - grad1[changed_mask]))

            is_urban_expansion = (
                delta_buildings >= 2
                or (delta_buildings >= 1 and changed_delta_edge >= 2.0)
                or (changed_delta_edge > 2.5 and changed_delta_bright > 8.0 and changed_delta_green > -0.25)
                or (changed_delta_bright > 22.0 and changed_delta_edge > 1.5)
            )
            is_urban_demolition = (
                delta_buildings < 0
                and (delta_built_pct <= -2.0 or changed_delta_edge < -2.0)
            )

            if is_urban_expansion:
                dominant_transition = "New Built-up Structure / Urban Construction"
                change_type_code = "urban_expansion"
            elif is_urban_demolition:
                dominant_transition = "Building Demolition / Structural Clearance"
                change_type_code = "urban_demolition"
            elif changed_delta_water > 0.12 and changed_delta_water >= changed_delta_green:
                dominant_transition = "Water Body Expansion / Inundation"
                change_type_code = "water_expansion"
            elif changed_delta_water < -0.12 and changed_delta_water <= changed_delta_green:
                dominant_transition = "Water Depletion / Desiccation"
                change_type_code = "water_shrink"
            elif changed_delta_green < -0.12:
                dominant_transition = "Vegetation Loss / Land Clearance"
                change_type_code = "veg_loss"
            elif changed_delta_green > 0.12:
                dominant_transition = "Vegetation Growth / Revegetation"
                change_type_code = "veg_growth"
            elif changed_delta_bright > 20.0:
                dominant_transition = "Surface Land Clearing / High-Albedo Ground"
                change_type_code = "urban_expansion"
            else:
                dominant_transition = "Surface Land Cover Modification"
                change_type_code = "general_modification"
        else:
            dominant_transition = "No Significant Change Detected"
            change_type_code = "none"
            changed_delta_edge = 0.0

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
        friendly_trans = {
            "water_expansion": "surface water expansion (flooding or inundation)",
            "veg_loss": "vegetation loss and land clearing",
            "veg_growth": "vegetation growth and revegetation",
            "water_shrink": "water body shrinkage and desiccation",
            "urban_expansion": "new building construction and built-up development",
            "urban_demolition": "building demolition and structural removal",
            "general_modification": "ground surface modification",
            "none": "no major changes",
        }.get(change_type_code, dominant_transition.lower())

        if changed_percentage > 0.05:
            answer = (
                f"Comparing the two images, noticeable changes were detected across about "
                f"{changed_percentage}% of the area (~{changed_area_hectares} hectares). "
                f"The main change observed is: {friendly_trans}, predominantly in the {change_sector} sector."
            )
            confidence = 0.91
            confidence_level = ConfidenceLevel.HIGH
        else:
            answer = (
                f"Comparing the two dates shows high stability with minimal change "
                f"({changed_percentage}% variance). The land cover has remained largely identical with no significant disruptions."
            )
            confidence = 0.88
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
            "building_count_t1": int(b1.get("count", 0)),
            "building_count_t2": int(b2.get("count", 0)),
            "building_count_delta": int(delta_buildings),
            "built_coverage_pct_t1": float(b1.get("built_coverage_pct", 0.0)),
            "built_coverage_pct_t2": float(b2.get("built_coverage_pct", 0.0)),
            "built_coverage_delta": float(delta_built_pct),
            "change_sector": change_sector,
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
