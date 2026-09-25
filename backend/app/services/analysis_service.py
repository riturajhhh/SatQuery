"""SatQuery AI — Analysis Orchestration Service.

Coordinates query execution, specialist model selection, execution traces,
model run logging, evidence generation, and database updates.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from PIL import Image
from sqlalchemy.orm import Session

from app.database import crud
from app.database.models import Analysis, UploadedFile
from app.models import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    ModelOutput,
    TaskType,
    get_model_registry,
)
from app.utils.logging import get_logger

from app.services.agentic_planner import AgenticPlanner, ExecutionPlan, PlanStep, PlanStrategy
from app.services.hallucination_guard import HallucinationGuard

logger = get_logger("services.analysis")


def _determine_task_type(query: str, num_files: int = 1, is_optical_sar: bool = False) -> TaskType:
    """Infer task type from query intent using AgenticPlanner."""
    modalities = ["sar", "optical"] if is_optical_sar else []
    plan = AgenticPlanner.create_plan(query=query, num_files=num_files, modalities=modalities)
    # Return the primary/culminating task
    return plan.steps[-1].task


def execute_analysis(
    db: Session,
    upload_id: str,
    query: str,
    options: Optional[Dict[str, Any]] = None,
) -> Analysis:
    """Execute an agentic single-image, bi-temporal, or cross-modal analysis pipeline.

    Decomposes queries into execution plans, chains specialist models, logs full
    auditable execution traces, and synthesizes multi-modal answers.

    Args:
        db: SQLAlchemy database session.
        upload_id: Upload identifier from /api/upload.
        query: Natural-language query string.
        options: Optional execution configuration (confidence threshold, evidence generation).

    Returns:
        The completed Analysis ORM record.
    """
    options = options or {}
    start_total = time.perf_counter()

    # 1. Look up uploaded files
    uploaded_files: List[UploadedFile] = crud.get_files_by_upload_id(db, upload_id)
    if not uploaded_files:
        raise ValueError(f"No uploaded files found for upload_id '{upload_id}'")

    primary_file = uploaded_files[0]

    # 2. Create the Analysis record
    analysis = crud.create_analysis(db, query=query)
    analysis.started_at = datetime.now(timezone.utc)
    analysis.status = "processing"
    db.commit()
    db.refresh(analysis)

    # Associate files with this analysis
    for uf in uploaded_files:
        uf.analysis_id = analysis.id
    db.commit()

    try:
        # Step 1: Input Ingestion & Modality Verification
        t0 = time.perf_counter()
        modality = primary_file.modality or "optical"
        modalities = [f.modality for f in uploaded_files if f.modality]
        is_optical_sar_pair = (
            len(uploaded_files) >= 2
            and ("sar" in modalities)
            and ("optical" in modalities or "multispectral" in modalities)
        )

        if is_optical_sar_pair:
            input_type = InputType.OPTICAL_SAR_PAIR
        elif len(uploaded_files) > 1:
            input_type = InputType.BI_TEMPORAL
        elif modality == "sar":
            input_type = InputType.SINGLE_SAR
        elif modality == "multispectral":
            input_type = InputType.SINGLE_MULTISPECTRAL
        else:
            input_type = InputType.SINGLE_OPTICAL

        step1_duration = (time.perf_counter() - t0) * 1000.0
        isro_platforms = [
            f.metadata_json.get("isro", {}).get("platform")
            for f in uploaded_files
            if f.metadata_json and f.metadata_json.get("isro")
        ]
        isro_suffix = f" ISRO Platform(s): {', '.join(set(isro_platforms))}." if isro_platforms else ""

        crud.create_trace_step(
            db=db,
            analysis_id=analysis.id,
            step_number=1,
            action="Input Ingestion & Modality Verification",
            status="success",
            duration_ms=round(step1_duration, 2),
            details=f"Verified {len(uploaded_files)} raster input(s). Modality: {modality}. Configuration: {input_type.value}.{isro_suffix}",
        )


        # Step 2: Agentic Planning & Intent Decomposition
        t1 = time.perf_counter()
        plan: ExecutionPlan = AgenticPlanner.create_plan(
            query=query,
            num_files=len(uploaded_files),
            modalities=modalities,
        )
        registry = get_model_registry()

        plan_steps_summary = " -> ".join(s.task.value for s in plan.steps)
        step2_duration = (time.perf_counter() - t1) * 1000.0
        crud.create_trace_step(
            db=db,
            analysis_id=analysis.id,
            step_number=2,
            action="Agentic Planning & Intent Decomposition",
            status="success",
            duration_ms=round(step2_duration, 2),
            details=f"Decomposed query into strategy '{plan.strategy.value}'. Execution plan: [{plan_steps_summary}].",
        )

        # Load images for inference
        pil_images = []
        if is_optical_sar_pair:
            opt_file = next((f for f in uploaded_files if f.modality in ("optical", "multispectral")), uploaded_files[0])
            sar_file = next((f for f in uploaded_files if f.modality == "sar"), uploaded_files[1])
            for uf in [opt_file, sar_file]:
                img_p = uf.preview_path or uf.stored_path
                pil_images.append(Image.open(img_p))
        else:
            for uf in uploaded_files:
                img_p = uf.preview_path or uf.stored_path
                pil_images.append(Image.open(img_p).convert("RGB"))

        # Step 3: Chained Multi-Step Execution
        executed_outputs: List[Tuple[PlanStep, Any, ModelOutput]] = []

        for idx, plan_step in enumerate(plan.steps):
            t_step_start = time.perf_counter()

            # Determine appropriate sub-input type and images
            if plan_step.task in (TaskType.GROUNDING, TaskType.CAPTIONING, TaskType.VQA) and len(uploaded_files) > 1:
                sub_input_type = InputType.SINGLE_OPTICAL
                sub_images = [pil_images[0]]
            else:
                sub_input_type = input_type
                sub_images = pil_images

            step_model = registry.select_best_model(plan_step.task, sub_input_type)

            sub_query = (
                plan_step.target_phrase
                if (plan_step.task == TaskType.GROUNDING and plan_step.target_phrase)
                else query
            )

            sub_input = ModelInput(
                images=sub_images,
                query=sub_query,
                metadata=primary_file.metadata_json,
                file_paths=[f.stored_path for f in uploaded_files],
            )

            sub_output: ModelOutput = step_model.predict(sub_input)
            t_step_duration = (time.perf_counter() - t_step_start) * 1000.0

            executed_outputs.append((plan_step, step_model, sub_output))

            # Trace individual sub-task step
            crud.create_trace_step(
                db=db,
                analysis_id=analysis.id,
                step_number=3 + idx,
                action=f"Tool Execution: {plan_step.task.value} ({step_model.info.name})",
                status="success",
                duration_ms=round(t_step_duration, 2),
                details=f"Step {idx+1}/{len(plan.steps)} ({plan_step.description}): Completed in {round(t_step_duration, 1)}ms. Confidence: {round(sub_output.confidence * 100)}%.",
            )

            # Log Model Run for this tool
            crud.create_model_run(
                db=db,
                analysis_id=analysis.id,
                model_name=step_model.info.name,
                model_version=step_model.info.version,
                task=plan_step.task.value,
                is_fallback=1 if sub_output.is_fallback else 0,
                parameters=options,
                output_data=sub_output.evidence or {},
                confidence=sub_output.confidence,
                execution_time_ms=sub_output.execution_time_ms,
                device=step_model.info.device,
                status="complete",
            )

            # Generate and persist evidence for this tool
            if options.get("generate_evidence", True):
                evidence_data = step_model.explain(sub_input, sub_output) or {}
                if sub_output.evidence and isinstance(sub_output.evidence, dict):
                    for k, v in sub_output.evidence.items():
                        if k not in evidence_data and not (k == "overlay_image" and hasattr(v, "save")):
                            evidence_data[k] = v

                ev_type = evidence_data.get("evidence_type") or (
                    "optical_sar_fusion"
                    if plan_step.task == TaskType.OPTICAL_SAR
                    else "change_detection_map"
                    if plan_step.task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA)
                    else "grounding_overlay"
                    if plan_step.task == TaskType.GROUNDING
                    else "spectral_indices"
                )
                ev_path = (
                    evidence_data.get("overlay_path")
                    or primary_file.preview_path
                    or primary_file.stored_path
                )
                crud.create_evidence(
                    db=db,
                    analysis_id=analysis.id,
                    evidence_type=ev_type,
                    file_path=str(ev_path),
                    description=f"Visual evidence for {plan_step.task.value}",
                    metadata_json=evidence_data,
                )

        # Step 4: Multi-Task Synthesis & Response Calibration
        t_synth = time.perf_counter()
        final_trace_step_num = 3 + len(plan.steps)

        if len(executed_outputs) > 1:
            # Multi-step narrative synthesis
            narrative_parts = [out.answer for _, _, out in executed_outputs]
            synthesized_text = " ".join(narrative_parts)
            avg_confidence = round(
                sum(out.confidence for _, _, out in executed_outputs) / len(executed_outputs), 2
            )
            is_any_fallback = any(out.is_fallback for _, _, out in executed_outputs)
            final_task_name = "agentic_multi_task"
            conf_level = ConfidenceLevel.HIGH if avg_confidence >= 0.85 else ConfidenceLevel.MEDIUM
        else:
            _, last_model, last_output = executed_outputs[0]
            synthesized_text = last_output.answer
            avg_confidence = last_output.confidence
            is_any_fallback = last_output.is_fallback
            final_task_name = plan.steps[0].task.value
            conf_level = last_output.confidence_level

        synth_duration = (time.perf_counter() - t_synth) * 1000.0
        crud.create_trace_step(
            db=db,
            analysis_id=analysis.id,
            step_number=final_trace_step_num,
            action="Cross-Task Synthesis & Response Calibration",
            status="success",
            duration_ms=round(synth_duration, 2),
            details=f"Synthesized {len(executed_outputs)} tool output(s). Initial confidence: {round(avg_confidence * 100)}% ({conf_level.value}).",
        )

        # Step 5: Evidence Consistency & Hallucination Audit
        t_audit = time.perf_counter()
        audit_step_num = final_trace_step_num + 1

        all_evidence_records = crud.get_evidence_for_analysis(db, analysis.id)
        all_evidence_dict = {
            item.evidence_type: item.metadata_json for item in all_evidence_records
        }

        audit_task = (
            plan.steps[0].task
            if len(plan.steps) == 1
            else TaskType.VQA
        )
        guard_result = HallucinationGuard.audit_and_calibrate(
            raw_answer=synthesized_text,
            raw_confidence=avg_confidence,
            task=audit_task,
            images=pil_images,
            evidence_dict=all_evidence_dict,
        )

        calibrated = guard_result.calibrated_confidence
        final_confidence = calibrated.score if calibrated else avg_confidence
        final_conf_level = calibrated.level if calibrated else conf_level

        # Persist confidence calibration evidence
        crud.create_evidence(
            db=db,
            analysis_id=analysis.id,
            evidence_type="confidence_calibration",
            file_path=str(primary_file.preview_path or primary_file.stored_path),
            description="Multi-factor confidence calibration and evidence consistency audit",
            metadata_json=guard_result.to_dict(),
        )

        audit_duration = (time.perf_counter() - t_audit) * 1000.0
        flags_str = ", ".join(calibrated.flags) if calibrated and calibrated.flags else "none"
        claims_str = ", ".join(guard_result.detected_claims) if guard_result.detected_claims else "none"

        crud.create_trace_step(
            db=db,
            analysis_id=analysis.id,
            step_number=audit_step_num,
            action="Evidence Consistency & Hallucination Audit",
            status="success" if guard_result.is_consistent else "warning",
            duration_ms=round(audit_duration, 2),
            details=(
                f"Consistency: {round(guard_result.consistency_score * 100)}%. "
                f"Calibrated: {round(final_confidence * 100)}% ({final_conf_level.value}). "
                f"Flags: [{flags_str}]. Claims verified: [{claims_str}]."
            ),
        )

        if guard_result.advisory_warning:
            synthesized_text = f"{synthesized_text}\n\n{guard_result.advisory_warning}"

        # Finalize Analysis record
        analysis.task = final_task_name
        analysis.input_type = input_type.value
        analysis.status = "complete"
        analysis.answer_text = synthesized_text
        analysis.confidence = final_confidence
        analysis.confidence_level = final_conf_level.value
        analysis.is_fallback = 1 if is_any_fallback else 0
        analysis.completed_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(analysis)

        logger.info(
            "agentic_analysis_completed",
            analysis_id=analysis.id,
            strategy=plan.strategy.value,
            steps_count=len(plan.steps),
            duration_ms=round((time.perf_counter() - start_total) * 1000.0, 2),
        )
        return analysis

    except Exception as e:
        logger.error("analysis_failed", analysis_id=analysis.id, error=str(e))
        analysis.status = "failed"
        analysis.error_message = str(e)
        analysis.completed_at = datetime.now(timezone.utc)
        db.commit()

        crud.create_trace_step(
            db=db,
            analysis_id=analysis.id,
            step_number=99,
            action="Error Handling",
            status="failed",
            details=f"Analysis failed: {e}",
        )
        raise
