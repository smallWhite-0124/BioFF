import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score  #  修复导入规范
from .core import Net, inject_label, get_device


class BioFFClassifier(BaseEstimator, ClassifierMixin):
    #  开放所有核心超参数，设生信友好默认值
    def __init__(self, hidden_dims=[256,128], lr=0.01, threshold=2.0, num_epochs=500, random_state=42,reweight_gamma=0.5):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.model = None
        self.num_classes = None
        self.device = get_device()
        self.reweight_gamma = reweight_gamma
        torch.manual_seed(random_state)

    def fit(self, X, y):
        #  数据校验（生信数据必做）
        if len(X) == 0:
            raise ValueError("输入数据为空！请检查生信数据文件路径/格式")
        if len(np.unique(y)) < 2:
            raise ValueError("标签类别数<2！生信分类任务至少需要2类（如正常/肿瘤）")

        input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))
        #  透传超参数到Net
        self.model = Net(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            device=self.device,
            num_classes=self.num_classes,  # 新增
            lr=self.lr,
            threshold=self.threshold,
            num_epochs=self.num_epochs
        )

        X_tensor = torch.tensor(X, dtype=torch.float32)  # 去掉了 .to(self.device)，后面在 model.train 内部会转移
        y_tensor = torch.tensor(y, dtype=torch.long)

        # ---- 计算类别权重（预算不变重分配） ----
        unique, counts = np.unique(y, return_counts=True)
        gamma = self.reweight_gamma
        a = counts.max()  # 多数类样本数
        b = counts.min()  # 少数类样本数
        w_maj = 1 - gamma*(a-b)/(a+b)  # 多数类权重（<1）
        w_min = 1 + gamma*(a-b)/(a+b)  # 少数类权重（>1）
        major_class = unique[counts.argmax()]

        sample_weights = np.where(y == major_class, w_maj, w_min)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        # 构造真实例和反例（变量名不变）
        x_pos = inject_label(X_tensor, y_tensor, self.num_classes)
        wrong_y = 1 - y_tensor
        x_neg = inject_label(X_tensor, wrong_y, self.num_classes)

        self.model.train(x_pos, x_neg, sample_weights)

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model.predict(X_tensor, self.num_classes).cpu().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
