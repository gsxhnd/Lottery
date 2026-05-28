"""兼容导出：业务服务已迁移到 business 层。"""

from lottery_api.business.prediction_service import PredictionService

__all__ = ["PredictionService"]
