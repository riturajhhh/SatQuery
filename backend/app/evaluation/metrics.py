"""SatQuery AI — Evaluation Metrics Engine.

Provides quantitative metric computation for remote-sensing vision-language tasks:
- NLP Text Metrics: BLEU (1-4), ROUGE-L, Token Accuracy, Exact Match
- Spatial Metrics: Bounding Box IoU, mAP@0.5
- Agentic Metrics: Routing Accuracy, Task Success Rate, Latency Profiling
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Sequence, Tuple


# ==============================================================================
# 1. NLP Text Generation Metrics
# ==============================================================================

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for standardized evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compute_token_accuracy(candidate: str, reference: str) -> float:
    """Token-level exact match / soft overlap ratio (0.0 to 1.0)."""
    cand_tokens = set(normalize_text(candidate).split())
    ref_tokens = set(normalize_text(reference).split())

    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0

    intersection = cand_tokens.intersection(ref_tokens)
    return len(intersection) / len(ref_tokens)


def compute_exact_match(candidate: str, reference: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if normalize_text(candidate) == normalize_text(reference) else 0.0


def compute_bleu(
    candidate: str,
    reference: str,
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 with brevity penalty."""
    cand_tokens = normalize_text(candidate).split()
    ref_tokens = normalize_text(reference).split()

    c_len = len(cand_tokens)
    r_len = len(ref_tokens)

    if c_len == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    # Brevity Penalty
    bp = 1.0 if c_len > r_len else math.exp(1.0 - (r_len / max(1, c_len)))

    results = {}
    precisions = []

    for n in range(1, max_n + 1):
        if c_len < n:
            precisions.append(0.0)
            results[f"bleu_{n}"] = 0.0
            continue

        cand_ngrams = Counter(
            tuple(cand_tokens[i : i + n]) for i in range(c_len - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(r_len - n + 1)
        )

        clipped_matches = sum(
            min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items()
        )
        total_cand_ngrams = max(1, c_len - n + 1)
        p_n = clipped_matches / total_cand_ngrams
        precisions.append(p_n)

        # Geometric mean of precisions up to n
        if all(p > 0 for p in precisions):
            log_prec_sum = sum(math.log(p) for p in precisions) / len(precisions)
            score = bp * math.exp(log_prec_sum)
        else:
            score = 0.0

        results[f"bleu_{n}"] = round(score, 4)

    return results


def compute_rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L (Longest Common Subsequence Precision, Recall, F1)."""
    cand_tokens = normalize_text(candidate).split()
    ref_tokens = normalize_text(reference).split()

    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f1": 0.0}

    # DP Table for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if cand_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "rouge_l_p": round(prec, 4),
        "rouge_l_r": round(rec, 4),
        "rouge_l_f1": round(f1, 4),
    }


# ==============================================================================
# 2. Spatial Grounding Metrics
# ==============================================================================

def compute_box_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Compute Intersection over Union between two bounding boxes [ymin, xmin, ymax, xmax]."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    y_min_a, x_min_a, y_max_a, x_max_a = box_a[:4]
    y_min_b, x_min_b, y_max_b, x_max_b = box_b[:4]

    inter_ymin = max(y_min_a, y_min_b)
    inter_xmin = max(x_min_a, x_min_b)
    inter_ymax = min(y_max_a, y_max_b)
    inter_xmax = min(x_max_a, x_max_b)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x_max_a - x_min_a) * max(0.0, y_max_a - y_min_a)
    area_b = max(0.0, x_max_b - x_min_b) * max(0.0, y_max_b - y_min_b)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return round(inter_area / union_area, 4)


def compute_grounding_map(
    pred_boxes: List[Sequence[float]],
    target_boxes: List[Sequence[float]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute average precision for predicted bounding boxes against ground truth at IoU threshold."""
    if not target_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    matched = 0
    used_targets = set()

    for p_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, t_box in enumerate(target_boxes):
            if idx in used_targets:
                continue
            iou = compute_box_iou(p_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx != -1:
            matched += 1
            used_targets.add(best_idx)

    precision = matched / len(pred_boxes)
    recall = matched / len(target_boxes)

    return round((2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0, 4)


# ==============================================================================
# 3. Agentic Routing & Performance Metrics
# ==============================================================================

def compute_routing_accuracy(
    predicted_tasks: Sequence[str],
    ground_truth_tasks: Sequence[str],
) -> float:
    """Fraction of queries correctly routed to the expected specialist task."""
    if not predicted_tasks:
        return 0.0
    correct = sum(
        1 for p, g in zip(predicted_tasks, ground_truth_tasks) if p.lower() == g.lower()
    )
    return round(correct / len(predicted_tasks), 4)
