"""推理执行"""

import random

import torch

from lottery.data import build_sequence_tensor, denormalize_prediction
from lottery.domain.types import PredictionCandidate, PredictionResult
from lottery_data import LotteryRecord
from lottery.models import LotteryLSTM

DEFAULT_SEQ_LEN = 10
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_BACKTEST_PERIODS = 30


def _match_stats(
    red_balls: list[int], blue_ball: int, actual: LotteryRecord
) -> tuple[float, float, float]:
    red_hits = len(set(red_balls) & set(actual.red_balls))
    blue_hit = 1.0 if blue_ball == actual.blue_ball else 0.0
    hit_rate = (red_hits / 6 + blue_hit) / 7 * 100
    return hit_rate, red_hits, blue_hit


def compute_candidate_hit_rates(
    model: LotteryLSTM,
    records: list[LotteryRecord],
    candidates: list[PredictionCandidate],
    seq_len: int,
    device: torch.device,
    backtest_periods: int = DEFAULT_BACKTEST_PERIODS,
) -> tuple[list[PredictionCandidate], int]:
    """基于历史滑动窗口回测，为每组备选计算命中率。"""
    count = len(candidates)
    slot_hit_sum = [0.0] * count
    slot_red_sum = [0.0] * count
    slot_blue_sum = [0.0] * count
    periods = 0

    start_idx = max(0, len(records) - seq_len - backtest_periods)
    end_idx = len(records) - seq_len

    model.eval()
    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            actual = records[idx + seq_len]
            features = build_sequence_tensor(records, idx, seq_len).to(device)
            output = model(features)
            normalized = output.squeeze(0).cpu().tolist()
            historical = generate_candidates(normalized, count=count)

            for slot, cand in enumerate(historical):
                rate, red_hits, blue_hit = _match_stats(
                    cand.red_balls, cand.blue_ball, actual
                )
                slot_hit_sum[slot] += rate
                slot_red_sum[slot] += red_hits
                slot_blue_sum[slot] += blue_hit
            periods += 1

    if periods == 0:
        return candidates, 0

    rated = [
        PredictionCandidate(
            red_balls=cand.red_balls,
            blue_ball=cand.blue_ball,
            hit_rate=round(slot_hit_sum[i] / periods, 1),
            red_hit_avg=round(slot_red_sum[i] / periods, 2),
            blue_hit_rate=round(slot_blue_sum[i] / periods * 100, 1),
        )
        for i, cand in enumerate(candidates)
    ]
    return rated, periods


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _candidate_key(red_balls: list[int], blue_ball: int) -> tuple[tuple[int, ...], int]:
    return (tuple(red_balls), blue_ball)


def generate_candidates(
    normalized: list[float],
    count: int = DEFAULT_NUM_CANDIDATES,
) -> list[PredictionCandidate]:
    """基于模型归一化输出生成多组不重复备选号码。"""
    candidates: list[PredictionCandidate] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    def add_candidate(red_balls: list[int], blue_ball: int) -> None:
        key = _candidate_key(red_balls, blue_ball)
        if key in seen:
            return
        seen.add(key)
        candidates.append(PredictionCandidate(red_balls=red_balls, blue_ball=blue_ball))

    add_candidate(*denormalize_prediction(normalized))

    noise_scales = [0.02, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.10, 0.12]
    attempt = 0
    while len(candidates) < count and attempt < 200:
        scale = noise_scales[attempt % len(noise_scales)]
        perturbed = [
            max(0.0, min(1.0, v + random.gauss(0, scale)))
            for v in normalized
        ]
        add_candidate(*denormalize_prediction(perturbed))
        attempt += 1

    nudge_idx = 0
    while len(candidates) < count and nudge_idx < 50:
        perturbed = list(normalized)
        dim = nudge_idx % 7
        delta = (1 if (nudge_idx // 7) % 2 == 0 else -1) * (0.01 * ((nudge_idx // 14) + 1))
        perturbed[dim] = max(0.0, min(1.0, perturbed[dim] + delta))
        add_candidate(*denormalize_prediction(perturbed))
        nudge_idx += 1

    return candidates[:count]


def predict_next(
    model: LotteryLSTM,
    records: list[LotteryRecord],
    model_dir: str,
    metadata: dict,
    seq_len: int = DEFAULT_SEQ_LEN,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
) -> PredictionResult:
    """基于最近 seq_len 期历史数据预测下一期号码。"""
    if len(records) < seq_len:
        raise ValueError(
            f"历史数据不足 {seq_len} 期（当前 {len(records)} 期），无法构造推理输入"
        )

    start_idx = len(records) - seq_len
    input_issues = [records[start_idx + i].issue for i in range(seq_len)]
    last_issue = input_issues[-1]

    features = build_sequence_tensor(records, start_idx, seq_len)
    device = _resolve_device()
    model = model.to(device)
    features = features.to(device)

    with torch.no_grad():
        output = model(features)

    normalized = output.squeeze(0).cpu().tolist()
    candidates = generate_candidates(normalized, count=num_candidates)
    candidates, backtest_periods = compute_candidate_hit_rates(
        model,
        records,
        candidates,
        seq_len=seq_len,
        device=device,
    )

    return PredictionResult(
        model_dir=model_dir,
        model_timestamp=metadata.get("timestamp"),
        seq_len=seq_len,
        input_issues=input_issues,
        last_issue=last_issue,
        candidates=candidates,
        normalized=normalized,
        backtest_periods=backtest_periods,
    )
