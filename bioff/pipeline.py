# bioff/pipeline.py

from .classifier import BioFFClassifier
from .preprocess import (
    load_good_bad_data,
    standardize_data,
    handle_missing_values,
    split_train_test
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def run_prediction(
        good_path: str = None,
        bad_path: str = None,
        data_path: str = None,          # 新增：单文件路径
        label_col: int = -1,            # 新增：标签所在列（默认最后一列）
        pos_label: int = 0,             # 新增：正样本的标签值
        # 保留核心预处理/模型参数，移除高变异基因相关
        scale_method: str = "zscore",
        missing_fill_method: str = "median",
        test_size: float = 0.2,
        # 模型超参数透传
        hidden_dim: int = 256,
        lr: float = 0.01,
        threshold: float = 2.0,
        num_epochs: int = 500,
        random_state: int = 42
):
    """
    生信数据Forward-Forward分类一键预测接口
    适配基因/蛋白质表达数据的分类任务（如正常/肿瘤样本）

    参数：
        good_path: 正样本文件路径（txt，每行一个样本，最后一列是标签）
        bad_path: 负样本文件路径（同上）
        scale_method: 标准化方法（zscore/minmax/robust），默认zscore（适配多数生物数据）
        missing_fill_method: 缺失值填充方式（mean/median/drop），默认median（抗生信数据噪音）
        test_size: 测试集比例，默认0.2（小样本可设0.1）
        hidden_dim: 模型隐藏层维度，默认256（高维生物数据可设512）
        lr: 学习率，默认0.01（生信小样本避免过拟合）
        threshold: FF算法阈值，默认2.0
        num_epochs: 训练轮数，默认500（生信小样本减少过拟合）
        random_state: 随机种子，保证结果可复现

    返回：
        model: 训练好的BioFFClassifier模型
        results: 字典，包含准确率、分类报告、混淆矩阵、预测值、真实值
    """
    # 1. 加载数据（增加异常捕获）
    """
    生信数据Forward-Forward分类一键预测接口
    ...
    """
    # ========== 新增：单文件自动拆分 ==========
    if data_path is not None:
        import tempfile
        data = np.loadtxt(data_path)
        X = data[:, :-1]
        # 提取标签列（支持负数索引）
        y = data[:, label_col]
        # 拆分正负样本
        pos_mask = (y == pos_label)
        X_good = X[pos_mask]
        X_bad = X[~pos_mask]
        
    # 检查是否为空
    if len(X_good) == 0:
        raise ValueError("未找到正样本，请检查 pos_label 设置")
    if len(X_bad) == 0:
        raise ValueError("未找到负样本，请检查标签分布")
        
    # 创建带标签的数据：正样本标签为0，负样本标签为1（与原 load_good_bad_data 约定一致）
    y_good = np.zeros((X_good.shape[0], 1))
    y_bad = np.ones((X_bad.shape[0], 1))
    data_good = np.hstack([X_good, y_good])
    data_bad = np.hstack([X_bad, y_bad])
        
    # 保存为临时文件
    f_good = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(f_good, data_good)
    good_path = f_good.name
    f_bad = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(f_bad, data_bad)
    bad_path = f_bad.name
    # ========== 新增结束 ==========
    try:
        X, y = load_good_bad_data(good_path, bad_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到！请检查路径：good_path={good_path}, bad_path={bad_path}")
    except Exception as e:
        raise RuntimeError(f"数据加载失败（生信数据建议用txt格式，每行一个样本）：{str(e)}")

    # 2. 预处理（简化：缺失值→标准化）
    X = handle_missing_values(X, method=missing_fill_method)
    X_scaled, scaler = standardize_data(X, method=scale_method)

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = split_train_test(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # 4. 训练模型（透传超参数）
    model = BioFFClassifier(
        hidden_dim=hidden_dim,
        lr=lr,
        threshold=threshold,
        num_epochs=num_epochs,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 5. 评估（返回详细生信分析需要的指标）
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    # 打印友好输出
    print(f"\n===== 生信分类预测结果 =====")
    print(f"测试集准确率: {acc:.4f}")
    print(f"使用归一化方法: {scale_method}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
    print(f"混淆矩阵:\n{conf_mat}")

    # 返回结构化结果（方便用户后续分析）
    results = {
        "accuracy": acc,
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        "y_pred": y_pred,
        "y_test": y_test
    }
    return model, results
