import pytest
from app.evaluation.metrics import (
    compute_exact_match,
    compute_token_accuracy,
    compute_bleu,
    compute_rouge_l,
    compute_box_iou,
    compute_grounding_map,
    compute_routing_accuracy,
    normalize_text,
)


def test_normalize_text():
    raw = "  Hello, World! Dense Vegetation... "
    norm = normalize_text(raw)
    assert norm == "hello world dense vegetation"


def test_exact_match():
    assert compute_exact_match("Dense forest canopy", "dense forest canopy.") == 1.0
    assert compute_exact_match("Dense forest canopy", "sparse trees") == 0.0


def test_token_accuracy():
    ref = "dense forest with tree canopy"
    cand = "dense forest canopy"
    acc = compute_token_accuracy(cand, ref)
    assert acc == 0.6  # 3 matching out of 5 reference tokens


def test_bleu():
    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the quick brown fox jumps over the lazy dog"
    bleu = compute_bleu(cand, ref)
    assert bleu["bleu_1"] == 1.0
    assert bleu["bleu_4"] == 1.0

    cand_partial = "the quick brown fox"
    bleu_partial = compute_bleu(cand_partial, ref)
    assert bleu_partial["bleu_1"] > 0.0


def test_rouge_l():
    ref = "dense forest canopy in the scene"
    cand = "dense canopy in the scene"
    rouge = compute_rouge_l(cand, ref)
    assert rouge["rouge_l_f1"] > 0.70


def test_box_iou():
    box_a = [0.0, 0.0, 1.0, 1.0]
    box_b = [0.0, 0.0, 1.0, 1.0]
    assert compute_box_iou(box_a, box_b) == 1.0

    box_c = [0.0, 0.0, 0.5, 0.5]  # Area 0.25, inter 0.25, union 1.0 -> IoU 0.25
    assert compute_box_iou(box_a, box_c) == 0.25

    box_disjoint = [2.0, 2.0, 3.0, 3.0]
    assert compute_box_iou(box_a, box_disjoint) == 0.0


def test_grounding_map():
    target_boxes = [[0.0, 0.0, 0.5, 0.5]]
    pred_boxes = [[0.0, 0.0, 0.5, 0.5]]
    m_ap = compute_grounding_map(pred_boxes, target_boxes, iou_threshold=0.5)
    assert m_ap == 1.0


def test_routing_accuracy():
    preds = ["vqa", "captioning", "grounding"]
    targets = ["vqa", "captioning", "vqa"]
    acc = compute_routing_accuracy(preds, targets)
    assert acc == pytest.approx(2 / 3, 0.01)
