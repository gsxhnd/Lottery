"""推理执行"""

import torch

from lottery.data import build_sequence_tensor, denormalize_prediction
from lottery.domain.types import LotteryRecord, PredictionResult
from lottery.models import LotteryLSTM

DEFAULT_SEQ_LEN = 10


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_next(
    model: LotteryLSTM,
    records: list[LotteryRecord],
    model_dir: str,
    metadata: dict,
    seq_len: int = DEFAULT_SEQ_LEN,
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
    red_balls, blue_ball = denormalize_prediction(normalized)

    return PredictionResult(
        model_dir=model_dir,
        model_timestamp=metadata.get("timestamp"),
        seq_len=seq_len,
        input_issues=input_issues,
        last_issue=last_issue,
        predicted_red_balls=red_balls,
        predicted_blue_ball=blue_ball,
        normalized=normalized,
    )
