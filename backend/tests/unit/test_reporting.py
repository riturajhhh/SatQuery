"""Unit tests for Phase 13 — Report & Audit service.

Validates:
1. Complete audit trail compilation via generate_report_data.
2. Standalone HTML audit report rendering with print styles.
3. Publication-grade PDF report rendering via ReportLab with zero errors.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Analysis, UploadedFile, ModelRun, ExecutionTrace, Evidence
from app.services.report_service import generate_report_data, render_html_report, render_pdf_report


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def populated_analysis(in_memory_db):
    """Create a sample completed analysis session with full relational entities."""
    now = datetime.now(timezone.utc)
    analysis = Analysis(
        id="audit_test_analysis_001",
        query="Verify urban expansion and flood extent in Ahmedabad region",
        task="optical_sar_analysis",
        input_type="optical_sar_pair",
        status="complete",
        confidence=0.885,
        confidence_level="HIGH",
        answer_text="Cross-modal Optical-SAR analysis detected 14.2 hectares of new built infrastructure along the western corridor, with 3.8 hectares of seasonal floodwater inundation.",
        is_fallback=0,
        started_at=now,
        completed_at=now,
    )
    in_memory_db.add(analysis)

    # Add 2 files (Cartosat optical and RISAT SAR)
    f1 = UploadedFile(
        id="f1_cartosat",
        analysis_id=analysis.id,
        upload_id="upload_audit_001",
        original_name="Cartosat_2S_Ahmedabad.tif",
        stored_name="Cartosat_2S_Ahmedabad.tif",
        stored_path="uploads/Cartosat_2S_Ahmedabad.tif",
        file_format="GeoTIFF",
        file_size_bytes=45000000,
        width=2048,
        height=2048,
        bands=4,
        dtype="uint16",
        crs="EPSG:32643",
        resolution={"x": 1.6, "y": 1.6},
        bounds={"left": 300000.0, "bottom": 2500000.0, "right": 303276.8, "top": 2503276.8},
        modality="multispectral",
        metadata_json={
            "isro": {
                "platform": "Cartosat-2S",
                "sensor_type": "4-Band VNIR",
                "gsd_nominal_m": 1.6,
                "bit_depth_effective": 11,
            }
        },
    )
    f2 = UploadedFile(
        id="f2_risat",
        analysis_id=analysis.id,
        upload_id="upload_audit_001",
        original_name="RISAT1A_EOS04_FRS.tif",
        stored_name="RISAT1A_EOS04_FRS.tif",
        stored_path="uploads/RISAT1A_EOS04_FRS.tif",
        file_format="GeoTIFF",
        file_size_bytes=32000000,
        width=1024,
        height=1024,
        bands=2,
        dtype="uint16",
        crs="EPSG:32643",
        resolution={"x": 3.0, "y": 3.0},
        bounds={"left": 300000.0, "bottom": 2500000.0, "right": 303072.0, "top": 2503072.0},
        modality="sar",
        metadata_json={
            "isro": {
                "platform": "RISAT-1A (EOS-04)",
                "sensor_type": "C-band SAR",
                "polarization": "Circular/Hybrid (RH/RV)",
            }
        },
    )
    in_memory_db.add_all([f1, f2])

    # Add model run
    mr = ModelRun(
        id="mr_001",
        analysis_id=analysis.id,
        model_name="OpticalSARFusionNet-ViT",
        model_version="1.2.0",
        task="optical_sar_analysis",
        is_fallback=0,
        confidence=0.89,
        execution_time_ms=142.5,
        device="cpu",
        status="success",
    )
    in_memory_db.add(mr)

    # Add trace steps
    t1 = ExecutionTrace(
        id="tr_001",
        analysis_id=analysis.id,
        step_number=1,
        action="Input Ingestion & Modality Verification",
        status="success",
        duration_ms=45.2,
        details="Verified 2 raster input(s). Modality: multispectral, sar. ISRO Platforms: Cartosat-2S, RISAT-1A (EOS-04).",
    )
    t2 = ExecutionTrace(
        id="tr_002",
        analysis_id=analysis.id,
        step_number=2,
        action="Cross-Modal Optical-SAR Alignment & Backscatter Calibration",
        status="success",
        duration_ms=112.8,
        details="Calibrated sigma0 backscatter in dB and verified sub-pixel co-registration footprint.",
    )
    in_memory_db.add_all([t1, t2])

    # Add evidence
    ev = Evidence(
        id="ev_001",
        analysis_id=analysis.id,
        evidence_type="optical_sar_fusion",
        file_path="outputs/processed/evidence/ev_fusion.png",
        description="Cross-modal Optical-SAR fusion heatmap identifying non-cloud-occluded urban backscatter.",
        metadata_json={
            "statistics": {
                "cloud_occlusion_percent": 34.2,
                "penetrated_cloud_hectares": 14.8,
            }
        },
    )
    in_memory_db.add(ev)

    in_memory_db.commit()
    return analysis


def test_generate_report_data_audit_completeness(in_memory_db, populated_analysis):
    """Verify that generate_report_data gathers all forensic audit fields."""
    report_data = generate_report_data(in_memory_db, populated_analysis.id)

    assert report_data["analysis_id"] == "audit_test_analysis_001"
    assert "urban expansion" in report_data["query"]
    assert report_data["status"] == "complete"
    assert report_data["confidence_level"] == "HIGH"
    assert report_data["confidence"] == 0.885
    assert len(report_data["files"]) == 2
    assert report_data["files"][0]["isro"]["platform"] == "Cartosat-2S"
    assert report_data["files"][1]["isro"]["platform"] == "RISAT-1A (EOS-04)"
    assert len(report_data["models_used"]) == 1
    assert report_data["models_used"][0]["model_name"] == "OpticalSARFusionNet-ViT"
    assert len(report_data["traces"]) == 2
    assert len(report_data["evidence"]) == 1
    assert "disclaimer" in report_data


def test_render_html_report(in_memory_db, populated_analysis):
    """Verify that render_html_report produces complete standalone HTML document."""
    report_data = generate_report_data(in_memory_db, populated_analysis.id)
    html_output = render_html_report(report_data)

    assert isinstance(html_output, str)
    assert "<!DOCTYPE html>" in html_output
    assert "SatQuery AI" in html_output
    assert report_data["analysis_id"] in html_output
    assert "Cartosat-2S" in html_output
    assert "RISAT-1A (EOS-04)" in html_output
    assert "Cross-modal Optical-SAR" in html_output
    assert "@media print" in html_output
    assert "System Audit Disclaimer" in html_output


def test_render_pdf_report(in_memory_db, populated_analysis):
    """Verify that render_pdf_report generates non-empty binary PDF stream."""
    report_data = generate_report_data(in_memory_db, populated_analysis.id)
    pdf_bytes = render_pdf_report(report_data)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    # PDF specification magic byte header
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_report_data_not_found(in_memory_db):
    """Verify that generate_report_data raises ValueError on missing ID."""
    with pytest.raises(ValueError, match="not found"):
        generate_report_data(in_memory_db, "nonexistent_id_999")
