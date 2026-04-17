import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

def load_good_bad_data(good_path: str, bad_path: str):
    """自动加载正负样本并拆分标签（适配基因/蛋白质表达数据）"""
    data_good = np.loadtxt(good_path)
    data_bad = np.loadtxt(bad_path)

    # 生信数据校验：避免维度不一致
    if data_good.ndim != 2 or data_bad.ndim != 2:
        raise ValueError("生信数据必须是二维矩阵！（每行一个样本，每列一个特征，最后一列标签）")
    if data_good.shape[1] != data_bad.shape[1]:
        raise ValueError(f"正负样本特征数不一致！good={data_good.shape[1]}, bad={data_bad.shape[1]}")

    all_data = np.vstack([data_good, data_bad])
    X = all_data[:, :-1]
    y = all_data[:, -1].astype(int)

    # 适配基因/蛋白质数据的提示
    print(f"数据加载完成：总样本 {X.shape[0]}, 特征数（基因/蛋白数） {X.shape[1]}")
    return X, y

def standardize_data(X, method="zscore"):
    """
    生物数据标准化（支持多种方法，默认Z-score）
    参数：
        X: 表达矩阵 (样本数, 特征数/基因数/蛋白数)
        method: 标准化方法
            - zscore:  Z-score 标准化（默认，均值0方差1，适合大部分生物数据）
            - minmax:  Min-Max 归一化（压缩到 [0,1]）
            - robust:  鲁棒标准化（抗异常值，适合有噪音的生信数据）
    返回：
        X_scaled: 标准化后矩阵
        scaler: 标准化器
    """
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method 仅支持 zscore / minmax / robust")

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def handle_missing_values(X, method="mean"):
    """
    处理生物数据中的缺失值
    参数：
        X: 表达矩阵 (样本数, 特征数/基因数/蛋白数)
        method: 填充方式，支持"mean"（均值填充）、"median"（中位数填充）、"drop"（删除样本）
    返回：
        X_clean: 无缺失值的矩阵
    """
    if method == "drop":
        return X[~np.isnan(X).any(axis=1)]
    elif method == "mean":
        return np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    elif method == "median":
        return np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    else:
        raise ValueError("method 只能是 mean/median/drop")

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集（适配生信数据，保持类别分布）
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
