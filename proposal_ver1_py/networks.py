# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ClippedCriticNet(nn.Module):

    def __init__(self, input_num, output_num, hidden_size):

        super().__init__()

        self.linear1 = nn.Linear(input_num, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_num)

        self.linear4 = nn.Linear(input_num, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, output_num)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu)) ##network1
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu)) ##network2
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2  #network1, network2の出力


class SoftActorNet(nn.Module):

    def __init__(self, input_num, output_num, hidden_size, action_scale):

        super().__init__()

        self.linear1 = nn.Linear(input_num, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, output_num)
        self.log_std_linear = nn.Linear(hidden_size, output_num)

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(0.)

    def forward(self, state, LOG_SIG_MAX = 2, LOG_SIG_MIN = -20):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)  #PyTorchでテンソルの値を指定した範囲に制限するための関数です。
        return mean, log_std   #出力は平均と（対数にした）分散

    def sample(self, state,epsilon = 1e-6):
        self.epsilon=epsilon
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)  #平均mean、標準偏std2の正規分布を定義
        x_t = normal.rsample()      #rsample() と sample() の違いは、rsample() はサンプリングの際に勾配追跡が可能になる点です（requires_grad=True のテンソルに対して有効）
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)       #与えられた値がその分布に従う確率密度関数（PDF）の対数値を計算します。
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
            #y はネットワークの出力値で、通常 tanh 関数を使用して制限されています。
            #log(1 − y²) は、tanh の特性を利用し、出力値が端（±1付近）に近いほど大きなペナルティを与える形で計算されます。
            #ε は、小さい定数であり、ゼロ除算を回避するために加えられます。

        log_prob = log_prob.sum(1, keepdim=True)    #テンソルの指定された次元で要素を加算する際に使用されるメソッドです。
                                                    #特に、keepdim=True を設定すると、計算後のテンソルの形状が元の次元を保持します（
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
