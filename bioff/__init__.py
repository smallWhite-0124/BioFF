__version__ = "0.2.6"
# 导出核心类/函数（移除高变异基因筛选相关）
from .classifier import BioFFClassifier
from .pipeline import run_prediction
from .preprocess import (
    load_good_bad_data,
    standardize_data,
    handle_missing_values,
    split_train_test
)
from .core import get_device  # 导出设备检测函数（生信用户可能需要）

# 包级别的说明
__author__ = "smallwhite"
__description__ = "基于Forward-Forward算法的生物信息学分类工具（适配基因/蛋白质表达数据）"
