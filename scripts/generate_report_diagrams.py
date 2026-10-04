import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram(output_path="reports/fig_4_1_architecture.png"):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    def draw_box(x, y, w, h, title, subtitle, fillcolor, edgecolor, title_color="#0f172a", sub_color="#334155"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     linewidth=1.8, edgecolor=edgecolor, facecolor=fillcolor)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_color)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=7.5, color=sub_color, multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=None, label_side="right"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.25", color="#1e40af", lw=1.6))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            offset = 0.15 if label_side == "right" else -0.15
            ax.text(mx + offset, my, label, ha="center" if label_side == "center" else ("left" if label_side == "right" else "right"),
                    va="center", fontsize=6.8, fontstyle="italic", color="#1e3a8a",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", lw=0.6))

    # Title Banner
    ax.text(5.0, 10.2, "SatQuery AI: End-to-End System Architecture", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#0f172a")

    # Tier 1: User Interface
    draw_box(2.2, 9.1, 5.6, 0.75, "React 18 User Interface (Client)",
             "Drag-and-Drop Ingestion | Query Input | Image Swipe Slider | Opacity Layer Controls",
             "#eff6ff", "#3b82f6", "#1e3a8a")

    # Tier 2: API Gateway Server
    draw_box(2.2, 7.9, 5.6, 0.75, "FastAPI Application Server (API Gateway)",
             "REST API Endpoints | Session Management | SQLite ORM | Async Job Dispatcher",
             "#f0fdfa", "#0d9488", "#115e59")

    # Tier 3: Geospatial Ingestion
    draw_box(2.2, 6.7, 5.6, 0.75, "Geospatial Ingestion & Sensor Normalizer",
             "Rasterio/GDAL CRS Parser | ISRO Cartosat Percentile Stretch | RISAT SAR dB Calibrator",
             "#ecfeff", "#0891b2", "#155e75")

    # Tier 4: Agentic Router
    draw_box(2.2, 5.5, 5.6, 0.75, "Agentic Query Router & Task Classifier",
             "Regex & Semantic Intent Parsing | Modality Verification | Dynamic Specialist Dispatcher",
             "#faf5ff", "#9333ea", "#581c87")

    # Tier 5: Dual-Engine Model Branches (Left = Production GPU, Right = Heuristic CPU)
    draw_box(0.5, 3.8, 4.2, 1.15, "Production Engine (GPU Accelerated)",
             "• BLIP-2 / Qwen2-VL (VQA & Captioning)\n• BIT Siamese Transformer (LEVIR-CD)\n• Florence-2 / Grounding DINO (Localization)",
             "#eef2ff", "#4f46e5", "#312e81")

    draw_box(5.3, 3.8, 4.2, 1.15, "Deterministic Fallback Engine (CPU)",
             "• Geophysical Spectral Indices (NDVI, NDWI, NDBI)\n• Otsu Bitemporal Structural Difference\n• Morphological Contour Bounding Boxes",
             "#fff7ed", "#ea580c", "#9a3412")

    # Tier 6: Evidence & Confidence Calibration
    draw_box(2.2, 2.3, 5.6, 0.85, "Visual Evidence Synthesis & Calibration Guard",
             "Pixel Evidence Masks | Composite Score C_final = 0.50 C_m + 0.35 C_e + 0.15 C_s\nMulti-Factor Hallucination Guard Cross-Check",
             "#ecfdf5", "#059669", "#064e3b")

    # Tier 7: Output Delivery
    draw_box(0.7, 0.9, 3.9, 0.8, "Interactive Dashboard Feedback",
             "Visual Heatmap Overlay | Execution Trace Timeline", "#fef2f2", "#dc2626", "#991b1b")

    draw_box(5.4, 0.9, 3.9, 0.8, "Auditable Intelligence Briefing",
             "Automated ReportLab PDF & Print HTML Export", "#fef2f2", "#dc2626", "#991b1b")

    # Connecting Arrows
    draw_arrow(5.0, 9.1, 5.0, 8.65, "HTTP POST Payload")
    draw_arrow(5.0, 7.9, 5.0, 7.45, "GeoTIFF Rasters")
    draw_arrow(5.0, 6.7, 5.0, 6.25, "Calibrated Tensors")

    # Branching to Models
    ax.annotate('', xy=(2.6, 4.95), xytext=(4.0, 5.5),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#4f46e5", lw=1.5))
    ax.text(2.8, 5.25, "GPU Available", fontsize=6.8, fontstyle="italic", color="#4f46e5",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#c7d2fe", lw=0.5))

    ax.annotate('', xy=(7.4, 4.95), xytext=(6.0, 5.5),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#ea580c", lw=1.5))
    ax.text(6.6, 5.25, "CPU / Fallback", fontsize=6.8, fontstyle="italic", color="#ea580c",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#fed7aa", lw=0.5))

    # Converging to Evidence
    ax.annotate('', xy=(3.8, 3.15), xytext=(2.6, 3.8),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#059669", lw=1.5))
    ax.text(2.6, 3.35, "Predictions", fontsize=6.8, fontstyle="italic", color="#059669",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#a7f3d0", lw=0.5))

    ax.annotate('', xy=(6.2, 3.15), xytext=(7.4, 3.8),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#059669", lw=1.5))
    ax.text(6.8, 3.35, "Spectral Indices", fontsize=6.8, fontstyle="italic", color="#059669",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#a7f3d0", lw=0.5))

    # Branching to Output
    ax.annotate('', xy=(2.65, 1.7), xytext=(3.9, 2.3),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#dc2626", lw=1.5))
    ax.annotate('', xy=(7.35, 1.7), xytext=(6.1, 2.3),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color="#dc2626", lw=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def generate_hallucination_guard_diagram(output_path="reports/fig_4_2_hallucination_guard.png"):
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def draw_box(x, y, w, h, title, subtitle, fillcolor, edgecolor, title_color="#0f172a", sub_color="#334155"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     linewidth=1.8, edgecolor=edgecolor, facecolor=fillcolor)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=7.2, color=sub_color, multialignment="center")

    def draw_diamond(cx, cy, rx, ry, title, subtitle):
        diamond = patches.Polygon([[cx - rx, cy], [cx, cy + ry], [cx + rx, cy], [cx, cy - ry]],
                                  linewidth=1.8, edgecolor="#d97706", facecolor="#fef3c7")
        ax.add_patch(diamond)
        ax.text(cx, cy + 0.12, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#78350f")
        ax.text(cx, cy - 0.22, subtitle, ha="center", va="center", fontsize=6.8, color="#92400e", multialignment="center")

    ax.text(5.0, 8.7, "Multi-Factor Hallucination Guard & Confidence Calibration Flowchart",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a")

    # Step 1: Input
    draw_box(2.2, 7.5, 5.6, 0.75, "Candidate Model Output Ingestion",
             "Raw Textual Claim A_raw + Predicted Visual Evidence Mask M",
             "#eff6ff", "#3b82f6", "#1e3a8a")

    # Step 2: Sub-scores
    draw_box(2.2, 6.2, 5.6, 0.75, "Compute Component Sub-Scores",
             "C_model (Softmax Logits) | C_evidence (Mask Quality) | C_sensor (GeoTIFF Metadata)",
             "#eff6ff", "#3b82f6", "#1e3a8a")

    # Step 3: Decision Diamond
    draw_diamond(5.0, 4.7, 2.3, 0.75, "Hallucination Guard Audit", "Does claim contradict physical\nspectral indices (NDVI / NDWI)?")

    # Step 4 Left: Fail Warning Box
    draw_box(0.5, 2.8, 3.8, 1.0, "Flag Semantic Contradiction",
             "• Apply penalty factor (0.60x)\n• Append audit warning to execution trace\n• Force categorical status -> UNCERTAIN",
             "#fef2f2", "#ef4444", "#991b1b")

    # Step 4 Right: Pass Box
    draw_box(5.7, 2.8, 3.8, 1.0, "Verify Spectral Consistency",
             "• Retain baseline composite weights\n• C_final = 0.50 C_m + 0.35 C_e + 0.15 C_s\n• Map to HIGH / MEDIUM / LOW rating",
             "#ecfdf5", "#10b981", "#064e3b")

    # Step 5: Final Output
    draw_box(2.0, 1.0, 6.0, 0.85, "Verified, Auditable Analysis Response",
             "Grounded Answer Text | Calibrated Confidence Badge | Spatial Evidence Overlay | Trace Audit",
             "#faf5ff", "#a855f7", "#581c87")

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#1e40af", lw=1.5))

    arrow(5.0, 7.5, 5.0, 6.95)
    arrow(5.0, 6.2, 5.0, 5.45)

    # Branch arrows
    ax.annotate('', xy=(2.4, 3.8), xytext=(2.7, 4.7),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#ef4444", lw=1.5))
    ax.text(1.9, 4.4, "YES (Mismatch)", fontsize=7.2, fontweight="bold", color="#dc2626",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#fca5a5", lw=0.5))

    ax.annotate('', xy=(7.6, 3.8), xytext=(7.3, 4.7),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#10b981", lw=1.5))
    ax.text(7.6, 4.4, "NO (Consistent)", fontsize=7.2, fontweight="bold", color="#059669",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#86efac", lw=0.5))

    # Converging arrows
    ax.annotate('', xy=(4.0, 1.85), xytext=(2.4, 2.8),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#6b7280", lw=1.4))
    ax.annotate('', xy=(6.0, 1.85), xytext=(7.6, 2.8),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#6b7280", lw=1.4))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
    generate_hallucination_guard_diagram()
