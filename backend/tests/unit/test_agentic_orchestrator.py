import pytest
from app.services.agentic_planner import AgenticPlanner, PlanStrategy

def test_planner_single_step_vqa():
    planner = AgenticPlanner()
    plan = planner.create_plan("What is the water surface condition?", ["img-1"])
    assert plan.strategy == PlanStrategy.DIRECT_SINGLE
    assert len(plan.steps) == 1
    assert plan.steps[0].task == "vqa"

def test_planner_single_step_captioning():
    planner = AgenticPlanner()
    plan = planner.create_plan("Describe this satellite scene in detail", ["img-1"])
    assert plan.strategy == PlanStrategy.DIRECT_SINGLE
    assert len(plan.steps) == 1
    assert plan.steps[0].task == "captioning"

def test_planner_single_step_grounding():
    planner = AgenticPlanner()
    plan = planner.create_plan("Locate the runway in this image", ["img-1"])
    assert plan.strategy == PlanStrategy.DIRECT_SINGLE
    assert len(plan.steps) == 1
    assert plan.steps[0].task == "grounding"
    assert "runway" in plan.steps[0].parameters.get("target_expression", "").lower()

def test_planner_single_step_change_detection():
    planner = AgenticPlanner()
    plan = planner.create_plan("Show difference between these two images", ["img-1", "img-2"])
    assert plan.strategy == PlanStrategy.DIRECT_SINGLE
    assert len(plan.steps) == 1
    assert plan.steps[0].task == "change_detection"

def test_planner_single_step_change_vqa():
    planner = AgenticPlanner()
    plan = planner.create_plan("Did urban buildings expand between 2020 and 2023?", ["img-1", "img-2"])
    assert plan.strategy == PlanStrategy.DIRECT_SINGLE
    assert len(plan.steps) == 1
    assert plan.steps[0].task == "change_vqa"

def test_planner_multi_step_chain():
    planner = AgenticPlanner()
    plan = planner.create_plan(
        "Locate the forest and analyze temporal changes over time",
        ["img-1", "img-2"]
    )
    assert plan.strategy == PlanStrategy.MULTI_STEP_CHAIN
    assert len(plan.steps) >= 2
    tasks = [step.task for step in plan.steps]
    assert "grounding" in tasks
    assert "change_detection" in tasks
