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

    # predict 命令（占位）
    predict_parser = subparsers.add_parser("predict", help="执行推理")
    predict_parser.add_argument("--model", type=str, required=True, help="模型路径")

    args = parser.parse_args()

    if args.command == "train":
        return _train(args)
    elif args.command == "predict":
        print("推理功能将在里程碑三实现")
        return 0
    else:
        parser.print_help()
        return 1


def _train(args) -> int:
    """训练命令处理"""
    import torch
    import platform
    from torch.utils.data import DataLoader
    from lottery.config import load_config
    from lottery.data import load_lottery_data, LotteryDataset
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
    print(f"  训练设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 50)

    config = load_config(args.config)

    print(f"加载数据: {config['data']['raw_file']}")
    records = load_lottery_data(config["data"]["raw_file"])
    print(f"加载 {len(records)} 条记录")

    dataset = LotteryDataset(records, seq_len=10)
    train_loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )

    model = LotteryLSTM()
    trainer = Trainer(model, config)

    print(f"开始训练 {config['training']['epochs']} 轮...")
    summary = trainer.train(train_loader, config["training"]["epochs"])

    model_path = save_model(model, config, summary, trainer.timestamp)
    print(f"模型已保存: {model_path}")
    print(f"TensorBoard: tensorboard --logdir={config['output']['logs_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
