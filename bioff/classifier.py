# classifier.py 改造后
import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score  #  修复导入规范
from core import Net, overlay_y_on_x, get_device


class BioFFClassifier(BaseEstimator, ClassifierMixin):
    #  开放所有核心超参数，设生信友好默认值
    def __init__(self, hidden_dim=256, lr=0.01, threshold=2.0, num_epochs=500, random_state=42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.model = None
        self.num_classes = None
        self.device = get_device()
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
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        # 数据转tensor并适配设备
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        x_pos = overlay_y_on_x(X_tensor, y_tensor, self.num_classes)
        x_neg = overlay_y_on_x(X_tensor, y_tensor[torch.randperm(X_tensor.size(0))], self.num_classes)
        self.model.train(x_pos, x_neg)

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model.predict(X_tensor, self.num_classes).cpu().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
