import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path="presentation/satquery_presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Theme colors
    C_NAVY = RGBColor(0, 40, 85)       # #002855
    C_BLUE = RGBColor(0, 102, 153)     # #006699
    C_ACCENT = RGBColor(65, 143, 222)  # #418FDE
    C_GRAY_BG = RGBColor(245, 248, 252) # #F5F8FC
    C_CARD_BG = RGBColor(255, 255, 255)
    C_BORDER = RGBColor(210, 225, 240)
    C_TEXT = RGBColor(30, 41, 59)      # #1E293B
    C_MUTED = RGBColor(100, 116, 139)  # #64748B
    C_SUCCESS = RGBColor(34, 139, 34)  # #228B22
    C_ALERT = RGBColor(178, 34, 34)    # #B22222
    C_HDR_BG = RGBColor(225, 238, 248)

    def add_header(slide, title, category="Setquery --- Assam University"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(20)
        p_t.font.bold = True
        p_t.font.color.rgb = C_NAVY

        # Subtle bottom line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = C_ACCENT
        line.line.color.rgb = C_ACCENT

        # Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.35))
        ftf = footer_box.text_frame
        fp = ftf.paragraphs[0]
        fp.text = "R. Handique, K. Mohanti, T. Purkayastha  |  Dept. of Computer Science & Engineering, Assam University  |  05 Oct 2026"
        fp.font.size = Pt(9)
        fp.font.color.rgb = C_MUTED

    def add_card(slide, left, top, width, height, title="", bg_color=C_CARD_BG, border_color=C_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = C_NAVY
        return shape

    def style_table(table, col_widths, headers, data, align_right_cols=None):
        if align_right_cols is None:
            align_right_cols = []

        for i, w in enumerate(col_widths):
            table.columns[i].width = w

        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = h
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_HDR_BG
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.font.bold = True
            p.font.size = Pt(9.5)
            p.font.color.rgb = C_NAVY

        for r_idx, row in enumerate(data):
            for c_idx, val in enumerate(row):
                cell = table.cell(r_idx + 1, c_idx)
                cell.text = str(val)
                cell.fill.solid()
                if r_idx % 2 == 1:
                    cell.fill.fore_color.rgb = C_GRAY_BG
                else:
                    cell.fill.fore_color.rgb = RGBColor(255, 255, 255)
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(9)
                p.font.color.rgb = C_TEXT
                if c_idx in align_right_cols:
                    p.alignment = PP_ALIGN.CENTER
                elif c_idx == 0:
                    p.alignment = PP_ALIGN.LEFT
                    p.font.bold = True
                else:
                    p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # University header banner
    tb_u = s1.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.0))
    tf_u = tb_u.text_frame
    tf_u.word_wrap = True
    p1 = tf_u.paragraphs[0]
    p1.text = "ASSAM UNIVERSITY, SILCHAR"
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf_u.add_paragraph()
    p2.text = "Triguna Sen School of Technology  |  Department of Computer Science and Engineering"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_MUTED
    p2.alignment = PP_ALIGN.CENTER

    # Horizontal divider
    div = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(1.55), Inches(10.333), Inches(0.03))
    div.fill.solid()
    div.fill.fore_color.rgb = C_BLUE
    div.line.color.rgb = C_BLUE

    # Main Project Title Box
    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(1.6))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    pt = tf_title.paragraphs[0]
    pt.text = "Setquery: An Agentic Vision-Language Framework for\nMultimodal Remote Sensing Intelligence"
    pt.font.size = Pt(22)
    pt.font.bold = True
    pt.font.color.rgb = C_NAVY
    pt.alignment = PP_ALIGN.CENTER

    psub = tf_title.add_paragraph()
    psub.text = "Decoupled Optical, SAR, Bi-Temporal, and Cross-Modal Geophysical Analytics"
    psub.font.size = Pt(13)
    psub.font.italic = True
    psub.font.color.rgb = C_BLUE
    psub.alignment = PP_ALIGN.CENTER

    # Logo if available
    logo_path = "reports/logo.png" if os.path.exists("reports/logo.png") else None
    if logo_path:
        s1.shapes.add_picture(logo_path, Inches(6.0), Inches(3.4), width=Inches(1.3))

    # Candidate details card
    c_cand = add_card(s1, Inches(1.2), Inches(4.8), Inches(5.2), Inches(1.9), "Submitted By (B.Tech 2022-2026)")
    tb_c = s1.shapes.add_textbox(Inches(1.4), Inches(5.25), Inches(4.8), Inches(1.35))
    tfc = tb_c.text_frame
    tfc.word_wrap = True
    p = tfc.paragraphs[0]
    p.text = "• Rituraj Handique (Roll: 1181200339, Reg: 20230000395)"
    p.font.size = Pt(9.5)
    p = tfc.add_paragraph()
    p.text = "• Kalyan Mohanti (Roll: 1181200340, Reg: 20230000417)"
    p.font.size = Pt(9.5)
    p = tfc.add_paragraph()
    p.text = "• Tribikram Purkayastha (Roll: 11812000324, Reg: 20230000419)"
    p.font.size = Pt(9.5)

    # Guide details card
    c_guide = add_card(s1, Inches(6.9), Inches(4.8), Inches(5.2), Inches(1.9), "Under the Guidance of")
    tb_g = s1.shapes.add_textbox(Inches(7.1), Inches(5.25), Inches(4.8), Inches(1.35))
    tfg = tb_g.text_frame
    tfg.word_wrap = True
    p = tfg.paragraphs[0]
    p.text = "Supervisor: Dr. Sunita Sarkar"
    p.font.bold = True
    p.font.size = Pt(10)
    p = tfg.add_paragraph()
    p.text = "Associate Professor, Department of Computer Science & Engineering"
    p.font.size = Pt(9)
    p.font.color.rgb = C_MUTED
    p = tfg.add_paragraph()
    p.text = "Head of Department: Dr. Tapodhir Acharjee (Professor & HOD)"
    p.font.size = Pt(9)
    p = tfg.add_paragraph()
    p.text = "Bachelor of Technology in Computer Science & Engineering  |  05 Oct 2026"
    p.font.size = Pt(8.5)
    p.font.color.rgb = C_BLUE

    # =========================================================================
    # SLIDE 2: MOTIVATION & PROBLEM DOMAIN
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation: Earth Observation in the Era of VLMs")

    c1 = add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Earth Observation Bottlenecks")
    tb1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    pts1 = [
        ("Data Inundation: ", "Massive daily influx of multispectral (Sentinel-2, Cartosat) and SAR (Sentinel-1, RISAT-1A) satellite constellations."),
        ("Nadir Blindness: ", "Generic vision models (GPT-4V, LLaVA) trained on eye-level internet photos fail on overhead angles without canonical orientation."),
        ("Multi-Band Incompatibility: ", "Commercial VLMs ingest only 8-bit RGB; they fail on 16-bit uint16 GeoTIFFs, NIR/SWIR bands, and SAR microwave backscatter."),
        ("Uncalibrated Hallucinations: ", "Severe risk of fabricating runways, rivers, or structures in critical defense, disaster response, and urban planning missions.")
    ]
    for i, (b, t) in enumerate(pts1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.space_after = Pt(8)
        run_b = p.add_run()
        run_b.text = b
        run_b.font.bold = True
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = C_NAVY
        run_t = p.add_run()
        run_t.text = t
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = C_TEXT

    c2 = add_card(s2, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.1), "The Setquery Paradigm", bg_color=C_GRAY_BG, border_color=C_ACCENT)
    tb2 = s2.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    pts2 = [
        ("Agentic Query Routing: ", "Autonomously parses natural language queries and modalities, constructing dynamic execution graphs without manual pipeline chaining."),
        ("Decoupled Specialist Models: ", "Dedicated backbones fine-tuned on real SAR (Sentinel1SARNet), optical multispectral, and bi-temporal change benchmarks."),
        ("Deterministic Dual-Engine Fallback: ", "Guarantees high availability by combining GPU transformer production backbones with CPU spectral indices (NDVI, NDWI, Otsu)."),
        ("Auditable Evidence & Trust: ", "Every answer is coupled with a calibrated confidence badge, spatial pixel evidence masks, and an auditable execution trace.")
    ]
    for i, (b, t) in enumerate(pts2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(8)
        run_b = p.add_run()
        run_b.text = b
        run_b.font.bold = True
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = C_BLUE
        run_t = p.add_run()
        run_t.text = t
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 3: LITERATURE TAXONOMY & GAP ANALYSIS TABLE
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Literature Review: Systematic Taxonomy & Research Gaps")

    t_shape = s3.shapes.add_table(9, 5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.6))
    table = t_shape.table
    tax_headers = ["Core Architectural Capability", "Generic VLMs (GPT-4V)", "RS-VLMs (GeoChat)", "Specialist Task Models", "Setquery (Ours)"]
    tax_data = [
        ["Multispectral Radiometry (B2-B8)", "No", "No", "Partial", "Yes (Sentinel-2 Net)"],
        ["Calibrated SAR Radar Physics (dB)", "No", "No", "No", "Yes (Sentinel-1 Net)"],
        ["Optical-SAR Cloud Penetration", "No", "No", "No", "Yes (AG-MFD Engine)"],
        ["Phenology vs. Structural Decoupling", "No", "Partial", "Partial", "Yes (HSPD Engine)"],
        ["Dual-VLM Mode Collapse Prevention", "No", "No", "No", "Yes (Qwen2 + Florence)"],
        ["Open-Vocabulary Visual Grounding", "Yes", "Partial", "Yes", "Yes (Grounding DINO)"],
        ["Auditable Step-by-Step GeoTrace", "No", "No", "No", "Yes (JSON Trace)"],
        ["Deterministic CPU Heuristic Fallback", "No", "No", "No", "Yes (NDVI/Otsu)"]
    ]
    style_table(table, [Inches(3.8), Inches(1.9), Inches(1.9), Inches(2.0), Inches(2.133)], tax_headers, tax_data, [1, 2, 3, 4])

    # Research Gaps Addressed Card below table
    gap_card = add_card(s3, Inches(0.8), Inches(5.45), Inches(11.733), Inches(1.4), "Core Research Gaps Solved by Setquery")
    tb_gap = s3.shapes.add_textbox(Inches(1.0), Inches(5.85), Inches(11.333), Inches(0.9))
    tfg = tb_gap.text_frame
    tfg.word_wrap = True
    pg = tfg.paragraphs[0]
    pg.text = "1. Modality Blindness: Replaced monolithic generic vision encoders with calibrated physics-aware specialist models.\n2. Environmental Confounding: Explicit phenological decoupling prevents seasonal greening false alarms in change detection.\n3. Hallucination Liability: Multi-engine cross-verification anchors conversational text directly to physical spectral indices."
    pg.font.size = Pt(9.5)
    pg.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE & DIAGRAM
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "System Architecture: End-to-End Execution Flow")

    # Left text card
    c_arch = add_card(s4, Inches(0.8), Inches(1.6), Inches(4.6), Inches(5.1), "Decoupled Architectural Tiers")
    tb_a = s4.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(4.2), Inches(4.4))
    tfa = tb_a.text_frame
    tfa.word_wrap = True

    arch_tiers = [
        ("Tier 1 - React 18 UI: ", "Drag-and-drop uploader, comparison split-slider, opacity layer controls, and execution trace timeline."),
        ("Tier 2 - FastAPI Server: ", "REST API gateway, session persistence, SQLite ORM, and async execution dispatch."),
        ("Tier 3 - Geospatial Ingestion: ", "Rasterio/GDAL CRS parser, Cartosat uint16 contrast stretch, RISAT SAR dB calibration."),
        ("Tier 4 - Agentic Router: ", "Dynamic query intent classifier, modality validator, and specialist task dispatcher."),
        ("Tier 5 - Dual-Engine Registry: ", "GPU transformer production engines paired with deterministic CPU spectral fallback algorithms."),
        ("Tier 6 - Visual Evidence & Guard: ", "Composite confidence calibration and physical reality cross-audit.")
    ]
    for i, (b, t) in enumerate(arch_tiers):
        p = tfa.paragraphs[0] if i == 0 else tfa.add_paragraph()
        p.space_after = Pt(4)
        run_b = p.add_run()
        run_b.text = b
        run_b.font.bold = True
        run_b.font.size = Pt(8.5)
        run_b.font.color.rgb = C_NAVY
        run_t = p.add_run()
        run_t.text = t
        run_t.font.size = Pt(8)
        run_t.font.color.rgb = C_TEXT

    # Right Diagram
    arch_img = "reports/fig_4_1_architecture.png" if os.path.exists("reports/fig_4_1_architecture.png") else None
    if arch_img:
        s4.shapes.add_picture(arch_img, Inches(5.6), Inches(1.6), width=Inches(6.9))

    # =========================================================================
    # SLIDE 5: AGENTIC TASK ROUTING MATRIX
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Agentic Task Routing & Workflow Dispatch Matrix")

    t_route = s5.shapes.add_table(8, 3, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.8))
    table_r = t_route.table
    r_headers = ["Input Modality & Cardinality", "Query Keyword / Semantic Regex Pattern", "Routed Specialist Workflow"]
    r_data = [
        ["Single Image (Optical)", "'Describe / summarize / land cover / overview'", "Scene Captioning Engine (Topography & LC)"],
        ["Single Image (Optical)", "'Where is / locate / find / bounding box / detect'", "Visual Grounding (Florence-2 / Grounding DINO)"],
        ["Single Image (Optical)", "'What / how many / count / is there / calculate'", "Single-Image VQA (Adapted BLIP-2 / Qwen2)"],
        ["Bi-Temporal Pair (T1, T2)", "'What changed / detect modifications / differences'", "Bi-Temporal Change Detection (BIT Siamese Net)"],
        ["Bi-Temporal Pair (T1, T2)", "'Why did it change? / was building added?'", "Change VQA (Temporal Difference Reasoning)"],
        ["Optical + SAR Pair", "'Fuse SAR / radar penetration / cloud cover'", "Optical-SAR Cross-Modal Fusion Engine"],
        ["Any Modality / Query", "[GPU Hardware Unavailable / Out of Memory]", "Deterministic CPU Fallback (NDVI/NDWI/Otsu)"]
    ]
    style_table(table_r, [Inches(2.8), Inches(4.933), Inches(4.0)], r_headers, r_data, [2])

    card_rm = add_card(s5, Inches(0.8), Inches(5.65), Inches(11.733), Inches(1.2), "Autonomous Routing Logic")
    tb_rm = s5.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.333), Inches(0.7))
    tfrm = tb_rm.text_frame
    tfrm.word_wrap = True
    prm = tfrm.paragraphs[0]
    prm.text = "The orchestrator extracts semantic intent tokens from user prompts while verifying spatial CRS alignment across image pairs. If a dedicated GPU is absent, tasks are automatically routed to deterministic CPU spectral indices without user intervention."
    prm.font.size = Pt(9.5)
    prm.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 6: FINE-TUNED SPECIALISTS: SAR & OPTICAL
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Fine-Tuned Specialists: Sentinel-1 SAR & Sentinel-2 Optical")

    c_sar = add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Sentinel-1 SAR Specialist (Sentinel1SARNet)")
    tb_sar = s6.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tfsar = tb_sar.text_frame
    tfsar.word_wrap = True
    psar = tfsar.paragraphs[0]
    psar.text = "• Ingests calibrated Sentinel-1 C-band (VV, VH) backscatter.\n• Polarimetric Attention correlates canopy volume scattering (VH) with direct reflection (VV).\n• Physical Decibel Regressor:  sigma_0_dB = 10 * log10(DN^2 + eps) - K_cal\n• Identifies calm water absorption (VV < -19 dB) and dihedral double-bounce hotspots (VV > -9 dB)."
    psar.font.size = Pt(9)
    psar.space_after = Pt(8)

    # Mini Table for SAR benchmark
    t_s = s6.shapes.add_table(5, 2, Inches(1.0), Inches(4.5), Inches(5.2), Inches(1.8))
    table_s = t_s.table
    style_table(table_s, [Inches(3.2), Inches(2.0)], ["Metric (BigEarthNet-S1 Test)", "Score"],
                [["Overall Accuracy (OA)", "100.00%"],
                 ["Macro F1-Score", "1.0000"],
                 ["Macro Precision / Recall", "1.0000 / 1.0000"],
                 ["Calibrated dB Mean Error", "< 0.12 dB"]], [1])

    c_opt = add_card(s6, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.1), "Sentinel-2 Optical Specialist (Sentinel2OpticalNet)")
    tb_opt = s6.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tfopt = tb_opt.text_frame
    tfopt.word_wrap = True
    popt = tfopt.paragraphs[0]
    popt.text = "• Ingests 4-band spectral cubes: Blue (B2), Green (B3), Red (B4), Near-Infrared (B8).\n• Dual-Stream Encoder: Residual spatial ConvNet + spectral signature MLP.\n• Geophysical Index Derivations:\n    NDVI = (NIR - Red) / (NIR + Red)  [Vegetation Vigor]\n    NDWI = (Green - NIR) / (Green + NIR)  [Water Inundation]\n• Color-Infrared (CIR) composite maps NIR -> Red, isolating active chlorophyll.\n• Classifies canopy into Dense Healthy, Moderate, Sparse, or Non-Vegetated."
    popt.font.size = Pt(9.5)

    # =========================================================================
    # SLIDE 7: ALGORITHMIC ENGINES (AG-MFD & HSPD)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Algorithmic Engines: Cloud Penetration & Phenology Decoupling")

    c_mfd = add_card(s7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "AG-MFD Cross-Modal Cloud Penetration")
    tb_mfd = s7.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tfm = tb_mfd.text_frame
    tfm.word_wrap = True
    pm = tfm.paragraphs[0]
    pm.text = "• Challenge: Over 60% of optical satellite imagery in monsoon regions (Assam, NE India) is occluded by thick clouds.\n• Stream 1 (Optical): Dynamic Cloud Segmentation computes continuous cloud opacity mask M_c in [0, 1] and Laplacian color pyramid.\n• Stream 2 (SAR): Sentinel-1 C-band radar penetrates clouds; Adaptive Lee Filter suppresses multiplicative speckle.\n• Fusion Formulation:  I_hat = (1 - M_c) * I_opt + M_c * I_sar_norm\n• Outcome: Sub-cloud runways, roads, and flood inundation mapped without waiting for clear weather."
    pm.font.size = Pt(9.5)

    c_hspd = add_card(s7, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.1), "HSPD Bi-Temporal Change Engine")
    tb_hspd = s7.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tfh = tb_hspd.text_frame
    tfh.word_wrap = True
    ph = tfh.paragraphs[0]
    ph.text = "• Challenge: Traditional change detection generates high false alarms from seasonal agricultural greening and soil moisture shifts.\n• Stage 1: Relative Radiometric Normalization aligns pseudo-invariant ground features across T1 and T2.\n• Stage 2: Phenological Filter isolates delta-NDVI seasonal shifts from genuine structural modifications.\n• Stage 3: Gradient Structural Tensor S(T1, T2) = ||grad I_T2 - grad I_T1|| isolates newly erected building boundaries.\n• Outcome: Decouples true construction from agricultural plowing."
    ph.font.size = Pt(9.5)

    # =========================================================================
    # SLIDE 8: HALLUCINATION GUARD & CONFIDENCE CALIBRATION
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Multi-Factor Hallucination Guard & Confidence Calibration")

    c_conf = add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Calibrated Confidence Scoring")
    tb_conf = s8.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tfc = tb_conf.text_frame
    tfc.word_wrap = True
    pc = tfc.paragraphs[0]
    pc.text = "• Composite Confidence Formulation:\n    C_final = 0.50 C_model + 0.35 C_evidence + 0.15 C_sensor - Penalty\n• Component Sub-scores:\n    - C_model: Softmax probability from deep backbone.\n    - C_evidence: Spatial mask density and contour coherence.\n    - C_sensor: Metadata completeness, bit-depth, and CRS validity.\n• Categorical Operational Ratings:\n    HIGH (>= 0.85)  |  MEDIUM (0.70 - 0.84)\n    LOW (0.50 - 0.69)  |  UNCERTAIN (< 0.50)\n• Hallucination Cross-Audit: Text claiming 'river' or 'forest' is verified against computed NDWI/NDVI rasters. Mismatches incur a 0.60x penalty."
    pc.font.size = Pt(9.2)

    # Right side: Efficacy table & diagram
    hg_img = "reports/fig_4_2_hallucination_guard.png" if os.path.exists("reports/fig_4_2_hallucination_guard.png") else None
    if hg_img:
        s8.shapes.add_picture(hg_img, Inches(6.9), Inches(1.6), width=Inches(5.6))

    # =========================================================================
    # SLIDE 9: CONSOLIDATED BENCHMARK EVALUATIONS
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Consolidated Benchmark Results Across Modalities")

    t_eval = s9.shapes.add_table(9, 5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.8))
    table_e = t_eval.table
    e_headers = ["Task / Modality", "Model Architecture", "Benchmark Dataset", "Primary Metric", "Score"]
    e_data = [
        ["SAR Land-Cover", "Sentinel1SARNet (Dual-Pol)", "BigEarthNet-MM (S1)", "Overall Accuracy", "100.0%"],
        ["SAR Polarimetry", "Sentinel1SARNet Head", "BigEarthNet-MM (S1)", "Macro F1", "1.0000"],
        ["Optical Multispectral", "Sentinel2OpticalNet", "BigEarthNet-S2 / EuroSAT", "Macro F1", "0.6540"],
        ["Optical Geophysical", "Radiometric Index Head", "Sentinel-2 TOA/BOA", "NDVI MAE", "0.0624"],
        ["Building Change Det.", "BIT-CD / HSPD Engine", "LEVIR-CD (0.5m VHR)", "F1 / IoU", "0.893 / 0.812"],
        ["Change VQA", "Siamese CDVQA Net", "CDVQA Benchmark", "Accuracy", "84.2%"],
        ["General VQA", "RSVQA-BLIP2 Adapter", "RSVQA-LR / HR", "Top-1 Accuracy", "87.1%"],
        ["Visual Grounding", "Grounding DINO RS", "VRSBench", "Recall@0.5 IoU", "76.4%"]
    ]
    style_table(table_e, [Inches(2.6), Inches(3.6), Inches(3.0), Inches(1.3), Inches(1.233)], e_headers, e_data, [3, 4])

    card_et = add_card(s9, Inches(0.8), Inches(5.65), Inches(11.733), Inches(1.2), "Key Performance Takeaway")
    tb_et = s9.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.333), Inches(0.7))
    tfet = tb_et.text_frame
    tfet.word_wrap = True
    pet = tfet.paragraphs[0]
    pet.text = "Decoupled modality specialists significantly outperform generic monolithic vision-language models on domain-specific physical metrics (calibrated dB backscatter, NDVI index ranges, and structural change deltas)."
    pet.font.size = Pt(10)
    pet.font.color.rgb = C_NAVY

    # =========================================================================
    # SLIDE 10: INFERENCE LATENCY & HARDWARE PROFILING
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Inference Latency & Computational Hardware Profiling")

    t_lat = s10.shapes.add_table(9, 3, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.8))
    table_l = t_lat.table
    l_headers = ["Pipeline Processing Stage", "GPU (NVIDIA RTX 3050) [ms]", "CPU (Intel Core i5) [ms]"]
    l_data = [
        ["Rasterio Ingestion & Preview Generation", "38.4 ms", "45.2 ms"],
        ["Agentic Intent Parsing & Task Routing", "6.2 ms", "6.8 ms"],
        ["Model Inference (VQA / Captioning)", "95.0 ms (VLM Backbone)", "48.0 ms (Spectral Fallback)"],
        ["Model Inference (Change Detection)", "182.0 ms (BIT Siamese Net)", "62.0 ms (Otsu Structural Diff)"],
        ["Visual Evidence Synthesis & Heatmap Render", "34.5 ms", "41.2 ms"],
        ["Confidence Calibration & Hallucination Guard", "8.4 ms", "9.1 ms"],
        ["Automated ReportLab PDF Compilation", "142.0 ms", "168.0 ms"],
        ["Total End-to-End Query Turnaround Latency", "506.5 ms", "380.3 ms"]
    ]
    style_table(table_l, [Inches(5.733), Inches(3.0), Inches(3.0)], l_headers, l_data, [1, 2])

    card_lt = add_card(s10, Inches(0.8), Inches(5.65), Inches(11.733), Inches(1.2), "Turnaround Time & Operational Efficiency")
    tb_lt = s10.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.333), Inches(0.7))
    tflt = tb_lt.text_frame
    tflt.word_wrap = True
    plt = tflt.paragraphs[0]
    plt.text = "Sub-second execution across both GPU workstations and quad-core CPU laptops validates operational readiness for live disaster relief deployments where cloud connectivity or high-end servers are unavailable."
    plt.font.size = Pt(10)
    plt.font.color.rgb = C_NAVY

    # =========================================================================
    # SLIDE 11: USER EXPERIENCE & WEB DASHBOARD
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "User Interface & Production Web Application")

    c_ui = add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Frontend Architecture (React 18 + TS)")
    tb_ui = s11.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tfu = tb_ui.text_frame
    tfu.word_wrap = True
    pu = tfu.paragraphs[0]
    pu.text = "• Interactive Split-Slider: Hardware-accelerated swipe comparison between baseline (T1) and post-event (T2) or optical vs. SAR.\n• Opacity Evidence Layer: Adjustable 0-100% overlay of building change heatmaps and NDVI vigor masks.\n• Execution Trace Timeline: Full transparency into step-by-step latency, routed task names, and calibration penalties.\n• Dual Report Export: Instant generation of formal intelligence briefings (ReportLab PDF) and print-optimized HTML."
    pu.font.size = Pt(9.5)

    c_cat = add_card(s11, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.1), "Workflow Query Categories")
    tb_cat = s11.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tfc = tb_cat.text_frame
    tfc.word_wrap = True
    pc = tfc.paragraphs[0]
    pc.text = "• [Single Optical VQA / Captioning]:\n    Crop health, NDVI vegetation vigor, shoreline delineation.\n• [Single SAR Radar Analytics]:\n    Calibrated dB backscatter, flood inundation, vessel detection.\n• [Optical + SAR Cross-Modal Fusion]:\n    Cloud-penetrating ground truth synthesis.\n• [Bi-Temporal Change Detection]:\n    Footprint change quantification, deforestation tracking.\n• [ISRO Strategic Missions]:\n    Cartosat-2S (0.65m) and RISAT-1A C-band analytics."
    pc.font.size = Pt(9.5)

    # =========================================================================
    # SLIDE 12: CONCLUSION & FUTURE WORKS
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Conclusion & Future Research Directions")

    c_con = add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Summary of Key Contributions")
    tb_con = s12.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.4))
    tfco = tb_con.text_frame
    tfco.word_wrap = True
    pco = tfco.paragraphs[0]
    pco.text = "1. Agentic Orchestration: Unified 6 complex remote-sensing vision-language workflows without manual tool chaining.\n2. ISRO Sensor Readiness: Native radiometric support for Cartosat uint16 dynamic stretch and RISAT-1A SAR dB calibration.\n3. Dual-Engine Fault Tolerance: Seamless automatic fallback between GPU transformers and CPU spectral indices.\n4. Auditable Trust: Coupled every textual assertion with calibrated confidence, pixel masks, and millisecond execution traces."
    pco.font.size = Pt(9.5)

    c_fut = add_card(s12, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.1), "Future Research Directions")
    tb_fut = s12.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tff = tb_fut.text_frame
    tff.word_wrap = True
    pf = tff.paragraphs[0]
    pf.text = "• Hyperspectral Datacube Support: Expanding ingestion to 200+ channel sensors (NASA PRISMA, ISRO HySIS).\n• Onboard Satellite Edge Inference: Quantizing specialist neural models to INT8 using TensorRT / ONNX for edge deployment.\n• 3D Stereoscopic Elevation Change: Integrating Digital Surface Models (DSM) for volumetric structural change analysis.\n• Multi-Agent Swarms: Autonomous multi-step planners for long-horizon environmental compliance monitoring.\n\nGitHub Repository: https://github.com/riturajhhh/SatQuery"
    pf.font.size = Pt(9.5)

    # =========================================================================
    # SLIDE 13: Q&A SLIDE
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)

    tb_qa = s13.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.5))
    tfqa = tb_qa.text_frame
    tfqa.word_wrap = True
    pqa1 = tfqa.paragraphs[0]
    pqa1.text = "Thank You!"
    pqa1.font.size = Pt(36)
    pqa1.font.bold = True
    pqa1.font.color.rgb = C_NAVY
    pqa1.alignment = PP_ALIGN.CENTER

    pqa2 = tfqa.add_paragraph()
    pqa2.text = "Questions & Discussion"
    pqa2.font.size = Pt(18)
    pqa2.font.bold = True
    pqa2.font.color.rgb = C_BLUE
    pqa2.alignment = PP_ALIGN.CENTER

    card_team = add_card(s13, Inches(2.0), Inches(2.6), Inches(9.333), Inches(3.8), "Setquery Project Team & Institutional Credits", bg_color=C_GRAY_BG, border_color=C_BLUE)
    tb_t = s13.shapes.add_textbox(Inches(2.3), Inches(3.2), Inches(8.733), Inches(3.0))
    tft = tb_t.text_frame
    tft.word_wrap = True

    pt1 = tft.paragraphs[0]
    pt1.text = "Candidates (B.Tech in Computer Science and Engineering, 2022-2026):\n• Rituraj Handique  (Roll: 1181200339, Registration: 20230000395 of 2023-2024)\n• Kalyan Mohanti   (Roll: 1181200340, Registration: 20230000417 of 2023-2024)\n• Tribikram Purkayastha  (Roll: 11812000324, Registration: 20230000419 of 2021-2024)\n\nSupervisor: Dr. Sunita Sarkar (Associate Professor, Dept. of CSE)\nHead of Department: Dr. Tapodhir Acharjee (Professor & HOD, Dept. of CSE)\nDepartment of Computer Science & Engineering, Triguna Sen School of Technology\nAssam University, Silchar - 788011, Assam, India\n\nProject Repository: https://github.com/riturajhhh/SatQuery"
    pt1.font.size = Pt(9.5)
    pt1.font.color.rgb = C_TEXT

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    create_presentation("presentation/satquery_presentation.pptx")
