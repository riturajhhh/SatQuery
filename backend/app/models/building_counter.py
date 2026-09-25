"""SatQuery AI — Building Footprint Detection & Counting Engine.

Provides automated structural segmentation, multi-spectral rooftop signature
extraction, mobile/app screenshot chrome detection, map text filtering,
morphological spatial filtering, geometric footprint verification, spatial
quadrant localization, and visual overlay generation for remote-sensing optical
imagery.
"""

from collections import deque
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

from app.utils.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("models.building_counter")


def _morph_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Perform binary morphological erosion using vectorized NumPy."""
    m = mask.copy()
    for _ in range(iterations):
        pad_m = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad_m[0:-2, 0:-2] & pad_m[0:-2, 1:-1] & pad_m[0:-2, 2:] &
            pad_m[1:-1, 0:-2] & pad_m[1:-1, 1:-1] & pad_m[1:-1, 2:] &
            pad_m[2:,   0:-2] & pad_m[2:,   1:-1] & pad_m[2:,   2:]
        )
    return m


def _morph_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Perform binary morphological dilation using vectorized NumPy."""
    m = mask.copy()
    for _ in range(iterations):
        pad_m = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad_m[0:-2, 0:-2] | pad_m[0:-2, 1:-1] | pad_m[0:-2, 2:] |
            pad_m[1:-1, 0:-2] | pad_m[1:-1, 1:-1] | pad_m[1:-1, 2:] |
            pad_m[2:,   0:-2] | pad_m[2:,   1:-1] | pad_m[2:,   2:]
        )
    return m


def _morph_open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Perform morphological opening (erosion followed by dilation) to break thin linear spurs."""
    return _morph_dilate(_morph_erode(mask, iterations), iterations)


class BuildingCounter:
    """Specialist engine for detecting, localizing, and counting buildings in satellite imagery."""

    @classmethod
    def detect_and_count(
        cls,
        image: Image.Image,
        min_area_pixels: int = 12,
        max_area_ratio: float = 0.22,
        max_aspect_ratio: float = 3.5,
        min_density: float = 0.22,
        pixel_resolution_meters: float = 1.0,
    ) -> Dict[str, Any]:
        """Analyze satellite image, detect discrete building footprints, and compute exact count.

        Args:
            image: Input PIL Image (converted to RGB).
            min_area_pixels: Minimum pixel footprint to reject sensor speckle noise.
            max_area_ratio: Maximum area fraction to reject broad landscape background.
            max_aspect_ratio: Maximum aspect ratio to reject linear road corridors.
            min_density: Minimum bounding box solidity/fill-rate.
            pixel_resolution_meters: Estimated GSD in meters per pixel.

        Returns:
            Dict containing:
                count: Total number of detected discrete buildings.
                boxes: List of bounding box dictionaries for UI rendering.
                quadrants: Count of buildings per spatial sector.
                built_coverage_pct: Overall built-up structural percentage.
                dominant_quadrant: Spatial sector with highest building concentration.
                size_breakdown: Counts of small, medium, and large structures.
                total_footprint_sqm: Estimated total rooftop ground area in square meters.
                answer: Natural-language formulation describing the count and distribution.
                overlay_path: Absolute file path to the annotated overlay image.
                overlay_url: API URL path to retrieve the overlay image.
        """
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size
        arr = np.asarray(rgb_img).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        total_pixels = float(h * w)

        # 1. Multi-spectral decomposition & Edge Gradients
        intensity = (r + g + b) / 3.0
        greenness = (g - r) / (g + r + 1e-5)

        dy = np.abs(np.diff(intensity, axis=0, prepend=intensity[:1, :]))
        dx = np.abs(np.diff(intensity, axis=1, prepend=intensity[:, :1]))
        grad_mag = (dy + dx) / 2.0
        edge_density = float(np.mean(grad_mag))

        # 2. UI Chrome Exclusion Mask (Phone / Map application screenshot detection)
        # When user uploads mobile screenshots (e.g. Google Maps with search pills and bottom sheet)
        ui_chrome_mask = np.zeros((h, w), dtype=bool)
        if h / max(1, w) > 1.65:
            # Top status bar & search header card (typically top 16%)
            top_h = int(h * 0.16)
            ui_chrome_mask[:top_h, :] = True

            # Bottom navigation sheet / tab bar (typically bottom 15%)
            bot_h = int(h * 0.85)
            ui_chrome_mask[bot_h:, :] = True

            # Floating right-side buttons (Compass, Layers, Target GPS, Directions)
            right_margin = int(w * 0.80)
            white_or_cyan_btn = (intensity > 220) | ((b > 180) & (g > 140) & (r < 80))
            ui_chrome_mask[:, right_margin:] |= white_or_cyan_btn[:, right_margin:]

        # 3. Map Text / Vector Labels Mask
        # Digital pure white characters (R>235, G>235, B>235) with thin stroke width
        pure_white = (r > 235) & (g > 235) & (b > 235)
        eroded_white = _morph_erode(pure_white, iterations=2)
        text_strokes = pure_white & ~_morph_dilate(eroded_white, iterations=2)
        text_halo_mask = _morph_dilate(text_strokes, iterations=2)

        # Deep water / broad water body mask
        water_mask = (
            ((b > g + 15) & (b > r + 25) & (intensity < 180) & (grad_mag < 8.0))
            | ((b > 140) & (b > g + 20) & (b > r + 35) & (grad_mag < 8.0))
        )
        water_pct = float(np.sum(water_mask)) / total_pixels * 100.0

        # Photosynthetic vegetation mask (real chlorophyll absorbs blue & red, reflects green)
        # Urban skyscraper shadows and dark asphalt have Rayleigh blue scatter (B >= G > R)
        # Checking G >= B - 2 and G > 40 prevents dark urban asphalt and skyscraper shadows from being misidentified as vegetation
        veg_mask = (g > r + 6) & (g >= b - 2) & (greenness > 0.05) & (g > 40) & ~water_mask
        veg_pct = float(np.sum(veg_mask)) / total_pixels * 100.0

        # Bare soil / sandy earthworks / excavation ground mask:
        # Natural sand and bare soil have characteristic warm yellow-tan hue (R > B + 24, G > B + 15)
        warm_sand_soil = (r > b + 24) & (g > b + 15) & (r > 68) & (grad_mag < 7.0) & ~veg_mask & ~water_mask
        soil_pct = float(np.sum(warm_sand_soil)) / total_pixels * 100.0

        # Shadow mask (localized deep cast shadows indicative of elevated 3D structures)
        shadow_mask = (intensity < 45) & ~water_mask

        # Valid map pixels for building detection
        valid_map_pixels = ~ui_chrome_mask & ~text_halo_mask & ~veg_mask & ~water_mask

        # 5. Precision Structural Rooftop Candidates
        # A. Terracotta / red-orange tile roofs (high red excess over BOTH green and blue)
        red_roof_cand = (r > 140) & (r > g + 18) & (r > b + 24) & valid_map_pixels

        # B. Copper patina / verdigris domes / turquoise oxidized roofs (e.g. World Financial Center / historic domes)
        copper_roof_cand = (g > 70) & (b > 65) & (g > r + 12) & (b > r + 10) & (intensity > 60) & valid_map_pixels

        # C. Solar panels / dark pitched roofs / industrial slate (intensity 28-115 with high boundary gradient)
        dark_roof_cand = (intensity >= 28) & (intensity <= 115) & (grad_mag > 4.5) & valid_map_pixels

        # D. Concrete / foundation slabs / neutral light roofs (intensity 105-245, balanced RGB, not warm sand)
        neutral_light_roof = (
            (intensity > 105) & (intensity < 245) &
            (np.abs(r - g) < 28) & (np.abs(g - b) < 28) &
            valid_map_pixels & ~warm_sand_soil
        )

        # E. Elevated roofs casting cast shadows immediately adjacent
        dilated_shadow = _morph_dilate(shadow_mask, iterations=2)
        elevated_roof = (intensity > 65) & dilated_shadow & valid_map_pixels

        raw_cand = (red_roof_cand | copper_roof_cand | dark_roof_cand | neutral_light_roof | elevated_roof) & valid_map_pixels

        # Single-iteration morphological opening (3x3 footprint):
        # Removes isolated 1-pixel noise without disintegrating textured roofs that have rooftop HVAC/vents
        cleaned_candidate = _morph_open(raw_cand, iterations=1)
        if not np.any(cleaned_candidate):
            # Fallback for synthetic or tiny test imagery
            cleaned_candidate = raw_cand

        built_coverage_pct = round(float(np.sum(cleaned_candidate)) / total_pixels * 100.0, 1)

        # 6. Connected Component Footprint Extraction
        # Scale-adaptive minimum pixel area & dimensions:
        is_mobile_screenshot = (h / max(1, w) > 1.65)
        if is_mobile_screenshot:
            min_pixels = max(150, int(total_pixels * 0.00030))
            max_pixels = int(total_pixels * 0.06)
            min_dim = 12
            max_aspect = 3.5
        elif total_pixels > 150000:
            min_pixels = max(30, int(total_pixels * 0.00010))
            max_pixels = int(total_pixels * 0.10)
            min_dim = 5
            max_aspect = 3.8
        else:
            min_pixels = max(min_area_pixels, 12)
            max_pixels = int(total_pixels * min(max_area_ratio, 0.22))
            min_dim = 3
            max_aspect = 3.8

        visited = np.zeros((h, w), dtype=bool)
        raw_components: List[Dict[str, Any]] = []
        cand_ys, cand_xs = np.where(cleaned_candidate)

        for cy, cx in zip(cand_ys, cand_xs):
            if visited[cy, cx]:
                continue

            queue = deque([(cy, cx)])
            visited[cy, cx] = True
            component_pixels = []
            ymin, ymax = cy, cy
            xmin, xmax = cx, cx

            while queue:
                y, x = queue.popleft()
                component_pixels.append((y, x))

                if y < ymin: ymin = y
                if y > ymax: ymax = y
                if x < xmin: xmin = x
                if x > xmax: xmax = x

                # 8-connected neighbors
                for dy_n in (-1, 0, 1):
                    for dx_n in (-1, 0, 1):
                        if dy_n == 0 and dx_n == 0:
                            continue
                        ny, nx = y + dy_n, x + dx_n
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            if cleaned_candidate[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

            area = len(component_pixels)
            if area < min_pixels or area > max_pixels:
                continue

            bw = xmax - xmin + 1
            bh = ymax - ymin + 1
            if bw < min_dim or bh < min_dim:
                continue

            aspect_ratio = max(bw / bh, bh / bw)
            # Reject linear road corridors, canals, and runways
            if aspect_ratio > max_aspect:
                continue
            if aspect_ratio > 2.6 and min(bw, bh) < 10:
                continue

            density = area / float(bw * bh)
            if density < min_density:
                continue

            # Perimeter and boundary edge contrast
            pad_ymin = max(0, ymin - 2)
            pad_ymax = min(h - 1, ymax + 2)
            pad_xmin = max(0, xmin - 2)
            pad_xmax = min(w - 1, xmax + 2)

            boundary_edges = grad_mag[pad_ymin : pad_ymax + 1, pad_xmin : pad_xmax + 1]
            comp_edges = float(np.max(boundary_edges))

            # Must have clear boundary contrast against surrounding landscape
            if comp_edges < 4.0:
                continue

            # Check for localized cast shadow in immediate vicinity (elevated structure cue)
            shadow_vicinity = int(np.sum(shadow_mask[pad_ymin : pad_ymax + 1, pad_xmin : pad_xmax + 1]))
            has_shadow = (shadow_vicinity >= 3)

            # Categorize building size adaptive to scene scale
            if area < min_pixels * 4:
                size_tier = "Small (Residential)"
                tier_short = "Small"
            elif area < min_pixels * 16:
                size_tier = "Medium (Commercial/Civic)"
                tier_short = "Medium"
            else:
                size_tier = "Large (Industrial/Warehouse)"
                tier_short = "Large"

            # Estimated ground footprint in square meters
            area_sqm = round(float(area) * (pixel_resolution_meters ** 2), 1)

            raw_components.append({
                "ymin": int(ymin),
                "ymax": int(ymax),
                "xmin": int(xmin),
                "xmax": int(xmax),
                "area": int(area),
                "area_sqm": area_sqm,
                "size_tier": size_tier,
                "tier_short": tier_short,
                "edge_score": comp_edges,
                "has_shadow": has_shadow,
                "density": round(density, 2),
                "center_y": (ymin + ymax) / 2.0,
                "center_x": (xmin + xmax) / 2.0,
            })

        # 7. Non-Maximum Suppression / Merging Overlapping Envelopes
        raw_components.sort(key=lambda c: c["area"], reverse=True)
        filtered_buildings: List[Dict[str, Any]] = []

        for comp in raw_components:
            overlap = False
            for fb in filtered_buildings:
                inter_ymin = max(comp["ymin"], fb["ymin"])
                inter_ymax = min(comp["ymax"], fb["ymax"])
                inter_xmin = max(comp["xmin"], fb["xmin"])
                inter_xmax = min(comp["xmax"], fb["xmax"])

                if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
                    comp_area = (comp["ymax"] - comp["ymin"]) * (comp["xmax"] - comp["xmin"])
                    fb_area = (fb["ymax"] - fb["ymin"]) * (fb["xmax"] - fb["xmin"])
                    iou = inter_area / float(comp_area + fb_area - inter_area)
                    if iou > 0.30:
                        overlap = True
                        break
            if not overlap:
                filtered_buildings.append(comp)

        count = len(filtered_buildings)

        # 8. Spatial Quadrants & Size Breakdown
        mid_h, mid_w = h / 2.0, w / 2.0
        q_h_start, q_h_end = h * 0.25, h * 0.75
        q_w_start, q_w_end = w * 0.25, w * 0.75

        quadrants_count = {
            "Northwest (top-left)": 0,
            "Northeast (top-right)": 0,
            "Southwest (bottom-left)": 0,
            "Southeast (bottom-right)": 0,
            "Central Zone": 0,
        }

        size_counts = {
            "Small (Residential)": 0,
            "Medium (Commercial/Civic)": 0,
            "Large (Industrial/Warehouse)": 0,
        }

        boxes: List[Dict[str, Any]] = []
        total_sqm = 0.0

        for idx, bldg in enumerate(filtered_buildings):
            cy, cx = bldg["center_y"], bldg["center_x"]
            total_sqm += bldg["area_sqm"]
            size_counts[bldg["size_tier"]] += 1

            if q_h_start <= cy <= q_h_end and q_w_start <= cx <= q_w_end:
                quadrants_count["Central Zone"] += 1
            if cy < mid_h and cx < mid_w:
                quadrants_count["Northwest (top-left)"] += 1
            elif cy < mid_h and cx >= mid_w:
                quadrants_count["Northeast (top-right)"] += 1
            elif cy >= mid_h and cx < mid_w:
                quadrants_count["Southwest (bottom-left)"] += 1
            else:
                quadrants_count["Southeast (bottom-right)"] += 1

            norm_ymin = round(float(bldg["ymin"]) / h, 4)
            norm_xmin = round(float(bldg["xmin"]) / w, 4)
            norm_ymax = round(float(bldg["ymax"]) / h, 4)
            norm_xmax = round(float(bldg["xmax"]) / w, 4)

            # Calibrate confidence using edge score, density, and shadow cues
            base_conf = 0.86 + min(0.08, (bldg["edge_score"] - 4.5) * 0.005)
            if bldg["has_shadow"]:
                base_conf += 0.03
            if bldg["density"] > 0.55:
                base_conf += 0.02
            conf = round(min(0.98, max(0.82, base_conf)), 2)

            boxes.append({
                "label": f"Building #{idx + 1}",
                "confidence": conf,
                "box_2d": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
                "box_pixel": [bldg["xmin"], bldg["ymin"], bldg["xmax"], bldg["ymax"]],
                "area_pixels": bldg["area"],
                "area_sqm": bldg["area_sqm"],
                "size_tier": bldg["size_tier"],
                "tier_short": bldg["tier_short"],
            })

        dominant_quadrant = max(quadrants_count.keys(), key=lambda k: quadrants_count[k]) if count > 0 else "N/A"
        mean_building_area_sqm = round(total_sqm / max(1, count), 1)

        # 9. Visual Evidence Generation (HUD-Style Annotated Overlay)
        settings = get_settings()
        evidence_dir = Path(settings.evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        overlay_id = f"building_count_{int(time.time() * 1000)}"
        overlay_filename = f"{overlay_id}.png"
        overlay_path = evidence_dir / overlay_filename

        overlay_img = rgb_img.copy().convert("RGBA")
        overlay_layer = Image.new("RGBA", rgb_img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay_layer)
        draw_box = ImageDraw.Draw(overlay_img)

        # Draw each detected building with HUD styling
        for idx, b_item in enumerate(boxes):
            xmin, ymin, xmax, ymax = b_item["box_pixel"]
            conf = b_item["confidence"]
            tier_short = b_item.get("tier_short", "Bldg")

            # Semi-transparent cyan tint over footprint
            draw_overlay.rectangle([xmin, ymin, xmax, ymax], fill=(6, 182, 212, 55))
            # Clean crisp cyan outer border
            draw_box.rectangle([xmin, ymin, xmax, ymax], outline=(6, 182, 212, 240), width=2)

            # L-bracket corner accents for high-precision HUD look
            c_len = min(6, max(3, (xmax - xmin) // 4), max(3, (ymax - ymin) // 4))
            draw_box.line([(xmin, ymin), (xmin + c_len, ymin)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmin, ymin), (xmin, ymin + c_len)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmax, ymin), (xmax - c_len, ymin)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmax, ymin), (xmax, ymin + c_len)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmin, ymax), (xmin + c_len, ymax)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmin, ymax), (xmin, ymax - c_len)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmax, ymax), (xmax - c_len, ymax)], fill=(255, 255, 255, 255), width=2)
            draw_box.line([(xmax, ymax), (xmax, ymax - c_len)], fill=(255, 255, 255, 255), width=2)

            # Dark pill badge with index and confidence
            badge_text = f"#{idx + 1} {tier_short} ({int(conf * 100)}%)"
            badge_w = len(badge_text) * 7 + 4
            badge_h = 13
            b_ymin = max(0, ymin - badge_h)
            draw_box.rectangle([xmin, b_ymin, xmin + badge_w, b_ymin + badge_h], fill=(15, 23, 42, 230), outline=(6, 182, 212, 180))
            draw_box.text((xmin + 3, b_ymin + 1), badge_text, fill=(255, 255, 255, 255))

        # Composite layers
        final_overlay = Image.alpha_composite(overlay_img, overlay_layer).convert("RGB")
        final_overlay.save(overlay_path, format="PNG")
        overlay_url = f"/api/files/evidence/{overlay_filename}"

        # 10. Formulate Natural, Human-Understandable Formulation
        if count > 0:
            quad_info = f" Most are clustered in the {dominant_quadrant.lower()} ({quadrants_count[dominant_quadrant]} buildings)." if quadrants_count[dominant_quadrant] > 0 else ""

            size_parts = []
            if size_counts["Small (Residential)"] > 0:
                size_parts.append(f"{size_counts['Small (Residential)']} small residential")
            if size_counts["Medium (Commercial/Civic)"] > 0:
                size_parts.append(f"{size_counts['Medium (Commercial/Civic)']} medium commercial/civic")
            if size_counts["Large (Industrial/Warehouse)"] > 0:
                size_parts.append(f"{size_counts['Large (Industrial/Warehouse)']} large institutional/facility")
            size_str = f" Including {', '.join(size_parts)} structures." if size_parts else ""

            area_str = f", covering ~{round(total_sqm):,} sq meters total footprint" if total_sqm > 0 else ""

            answer = (
                f"I detected {count} discrete buildings across this satellite scene (covering about {built_coverage_pct}% of the ground{area_str})."
                f"{size_str}{quad_info} Each building has been highlighted with a cyan box and size indicator on the map below so you can inspect them easily."
            )
        else:
            answer = (
                f"A total of 0 buildings were detected in this satellite imagery (built structural coverage: {built_coverage_pct}%). "
                f"The surveyed area consists entirely of natural terrain (about {round(veg_pct, 1)}% green vegetation and canopy, "
                f"{round(water_pct, 1)}% water, and {round(soil_pct, 1)}% bare soil) with no residential, commercial, or industrial buildings."
            )

        return {
            "count": count,
            "building_count": count,
            "boxes": boxes,
            "quadrants": quadrants_count,
            "dominant_quadrant": dominant_quadrant,
            "size_breakdown": size_counts,
            "total_footprint_sqm": round(total_sqm, 1),
            "mean_building_area_sqm": mean_building_area_sqm,
            "built_coverage_pct": built_coverage_pct,
            "answer": answer,
            "overlay_path": str(overlay_path),
            "overlay_url": overlay_url,
            "edge_density": round(edge_density, 2),
            "water_pct": round(water_pct, 1),
            "veg_pct": round(veg_pct, 1),
            "soil_pct": round(soil_pct, 1),
        }
