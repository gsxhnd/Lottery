"""命令行主入口"""

import argparse
import sys


def main() -> int:
    """主入口函数"""
    parser = argparse.ArgumentParser(
        prog="lottery", description="双色球历史数据实验项目"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, help="配置文件路径")

    from lottery.cli.data import register_data_commands

    register_data_commands(subparsers)

    # predict 命令
    predict_parser = subparsers.add_parser("predict", help="执行推理")
    predict_parser.add_argument(
        "--model", type=str, required=True, help="模型目录或 model.pt 路径"
    )
    predict_parser.add_argument("--config", type=str, help="配置文件路径")

    args = parser.parse_args()

    if args.command == "train":
        return _train(args)
    elif args.command == "predict":
        return _predict(args)
    elif args.command == "data":
        from lottery.cli.data import run_data_command

        return run_data_command(args)
    else:
        parser.print_help()
        return 1


def _train(args) -> int:
    """训练命令处理"""
    import torch
    import platform
    from torch.utils.data import DataLoader, random_split
    from lottery.config import load_config
    from lottery.data import LotteryDataset
    from lottery.models import LotteryLSTM
    from lottery.training import Trainer, save_model

    # 显示环境信息
    print("=" * 50)
    print("环境信息:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  系统平台: {platform.system()} {platform.machine()}")
    print(f"  Python 版本: {platform.python_version()}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and mps_backend.is_available()
    print(f"  MPS 可用: {mps_available}")
    print("=" * 50)

    config = load_config(args.config)

    data_source = config["data"].get("source", "auto")
    db_file = config["data"].get("db_file", "data/lottery.duckdb")
    print(f"数据源: {data_source} (raw={config['data']['raw_file']}, db={db_file})")
    dataset = LotteryDataset.from_config(config, seq_len=10)
    print(f"加载 {len(dataset.records)} 条记录")
    val_split = config["training"]["val_split"]
    batch_size = config["training"]["batch_size"]

    val_size = int(len(dataset) * val_split)
    if 0 < val_size < len(dataset):
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"训练样本: {train_size}, 验证样本: {val_size}")
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        print("验证集划分被跳过（val_split 导致验证样本为 0）")

    model = LotteryLSTM()
    trainer = Trainer(model, config)
    print(f"训练设备: {trainer.device}")

    print(f"开始训练 {config['training']['epochs']} 轮...")
    summary = trainer.train(
        train_loader, config["training"]["epochs"], val_loader=val_loader
    )

    model_path = save_model(model, config, summary, trainer.timestamp)
    print(f"模型已保存: {model_path}")
    print(f"TensorBoard: tensorboard --logdir={config['output']['logs_dir']}")

    return 0


def _predict(args) -> int:
    """推理命令处理"""
    import json
    from datetime import datetime

    from lottery.config import load_config
    from lottery.data import load_lottery_records
    from lottery.inference import (
        DEFAULT_SEQ_LEN,
        load_model_artifact,
        predict_next,
        save_prediction,
    )

    config = load_config(args.config)

    print(f"加载模型: {args.model}")
    model, artifact = load_model_artifact(args.model)
    model_dir = artifact.model_path

    records = load_lottery_records(config)
    print(f"加载 {len(records)} 条记录")

    result = predict_next(
        model,
        records,
        model_dir=model_dir,
        metadata=artifact.metadata,
        seq_len=DEFAULT_SEQ_LEN,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = save_prediction(
        result, config["output"]["summaries_dir"], timestamp=timestamp
    )
    print(f"推理结果已保存: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
