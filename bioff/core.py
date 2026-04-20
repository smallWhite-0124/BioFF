# ==============================
# Forward-Forward Algorithm Core
# Source: https://github.com/mpezeshki/pytorch_forward_forward
# ==============================
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

# 新增：设备自动检测函数
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Layer(nn.Linear):
    # 开放超参数：lr、threshold、num_epochs，设生信友好默认值
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 lr=0.01, threshold=2.0, num_epochs=500):  # 生信小样本默认调小lr/epochs
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = nn.ReLU()
        self.lr = lr
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.device = get_device()  # 设备适配

    def forward(self, x):
        x = x.to(self.device)  # 数据移到对应设备
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T.to(self.device)) +
            self.bias.unsqueeze(0).to(self.device)
        )

    def train(self, x_pos, x_neg):
        x_pos, x_neg = x_pos.to(self.device), x_neg.to(self.device)
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], device=None, num_classes=2,
                 lr=0.01, threshold=2.0, num_epochs=500):
        super().__init__()
        self.device = device or get_device()
        self.num_classes = num_classes

        # 拼接 One-Hot 后的真实输入维度
        actual_input_dim = input_dim + num_classes

        # 动态构建隐藏层
        layers = []
        prev_dim = actual_input_dim
        for h_dim in hidden_dims:
            layers.append(Layer(prev_dim, h_dim, device=self.device,
                                lr=lr, threshold=threshold, num_epochs=num_epochs))
            prev_dim = h_dim
        self.layers = nn.ModuleList(layers)

    def predict(self, x, num_classes):
        x = x.to(self.device)
        goodness_per_label = []
        for label in range(num_classes):
            y = torch.full((x.size(0),), label, device=self.device, dtype=torch.long)
            h = inject_label(x, y, num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos.to(self.device), x_neg.to(self.device)
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

def inject_label(x, y, num_classes):
    """
    将标签以 One-Hot 形式拼接到输入特征末尾。
    输入：
        x: (batch_size, input_dim)
        y: (batch_size,)
        num_classes: 类别数
    返回：
        (batch_size, input_dim + num_classes)
    """
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device, dtype=x.dtype)
    y_onehot[range(x.size(0)), y] = 1.0
    return torch.cat([x, y_onehot], dim=1)
