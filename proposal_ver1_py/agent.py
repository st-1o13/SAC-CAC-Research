# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from networks import ClippedCriticNet, SoftActorNet
from network_utils import soft_update, hard_update, convert_network_grad_to_false
from gradient_blend import restricted_direction
# NOTE: update_critics_and_actor() references the bare name `args` directly
# (not self.args) in its Stage1 condition check. This is a pre-existing
# inconsistency in Proposal_Ver1.ipynb that is intentionally left unchanged.
# Importing `args` here makes that bare reference resolve to the SAME dict
# object as config.args (and therefore the same object passed in as the
# constructor's `args` parameter), so behavior is unchanged.
from config import args


class SoftActorCriticModel(object):

    def __init__(self, state_num, action_num, action_scale, args, device):
        self.args = args

        self.h_buffer = []   # ★追加：安全度の移動平均用バッファ
        self.h_buffer_size = 80  # ★追加：移動平均の窓幅（推奨20）

        self.m_dir_buffer = []


        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.device = device
        self.target_update_interval = args['target_update_interval']
        self.updates = 0

                # 履歴記録（エピソード毎）
        self.history_safe_grad_norms = []      # safety 勾配ノルム（エピソード単位の平均または最終値）
        self.history_stability_grad_norms = [] # stability 勾配ノルム
        self.history_safe_grad_norms_stage1 = []   # ステージ1 用（オプション）
        self.history_stability_grad_norms_stage2 = [] # ステージ2 用（オプション）

        # ステージ２用の安全項スケールを args に追加しておく
        self.lambda_safe = args.get('lambda_safe', 4.0)

        self.actor_net = SoftActorNet(
            input_num=state_num, output_num=action_num, hidden_size=args['hidden_size'], action_scale=action_scale).to(self.device)

        self.critic_net = ClippedCriticNet(input_num=state_num + action_num, output_num=1, hidden_size=args['hidden_size']).to(device=self.device)

        self.critic_net_target = ClippedCriticNet(input_num=state_num + action_num, output_num=1, hidden_size=args['hidden_size']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=1e-4)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=args.get('critic_lr', 3e-4))

        self.critic_safe        = ClippedCriticNet(input_num=state_num + action_num, output_num=1, hidden_size=args['hidden_size']).to(device=self.device)
        self.critic_safe_target = ClippedCriticNet(input_num=state_num + action_num, output_num=1, hidden_size=args['hidden_size']).to(self.device)
        hard_update(self.critic_safe_target, self.critic_safe)
        convert_network_grad_to_false(self.critic_safe_target)
        self.critic_safe_optim  = optim.Adam(self.critic_safe.parameters(),lr=5e-5)


        #self.target_entropy = -torch.prod(torch.Tensor(action_num).to(self.device)).item()  #target entropy H_tは、行動の次元数dに対して、  H_t=-d
                    #torch.prod関数を使用して、テンソル内の全要素の積を計算し、その結果をPythonの数値型（item()）に変換する
        self.target_entropy = -float(action_num)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)  #torch.zeros() を使用して 値がすべて0のテンソル を作成しています。1要素だけのテンソルを作成します。
        self.alpha_optim = optim.Adam([self.log_alpha])   #log_alphaのパラメータを最適化するためのAdamオプティマイザーを初期化します。

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _, _ = self.actor_net.sample(state)
        else:
            _, _, action = self.actor_net.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        thdot = x[:, 2]
        h_raw = thdot + 2.0              # θ̇ = -2 → h_raw = 0（境界）
        # 正の側だけクリップ、負の側はそのまま
        h_pos_clipped = torch.clamp(h_raw, max=2.0)  # 上だけ 2 に飽和
        h_norm = h_pos_clipped / 2.0                 # 安全側は最大 1
        return h_norm

    def h_raw(self, s):
        # Pendulum の角速度は s[:,2]
        thdot = s[:, 2]
        return thdot + 2.0   # ← 生の h(s)




    def compute_safety_reward(self, s, a, s_next, alpha):
        """
        Control Barrier Function に基づく瞬時安全報酬 r_safe
        r_safe = exp(min(h(s') + (γ0-1)*h(s), 0))
        """
        h_s      = self.h(s)
        h_s_next = self.h(s_next)
        raw      = h_s_next + (alpha - 1.0) * h_s
        clipped = torch.clamp(raw, max=0.0)
        return torch.exp(clipped) - 1.0

    def compute_nav_reward(self, s, a, s_next):
        """
        既存の安定／目標達成報酬 r_nav を返すラッパー
        """
        return self.env_reward(s, a, s_next)

    def update_critics_and_actor(self, batch, episode_index):
        """
        Stage1/Stage2 の切り替えを含む
        - batch: replay buffer から取ってきたミニバッチ
        - episode_index: 現在のエピソード番号（1始まり）
        """
        self.updates += 1

        max_grad_norm = 500.0

        stage1 = False#(episode_index <= args['stage1_episodes'])

        # 1) バッチの展開
        s, a, reward_tuple, s_next, mask = batch
        s      = s.to(self.device)
        a      = a.to(self.device)
        s_next = s_next.to(self.device)
        mask   = mask.to(self.device)
        # (必要に応じて mask, logp_old を batch に含めてください)
        reward_tensor = torch.FloatTensor(reward_tuple).to(self.device)
        r_nav, r_safe = reward_tensor[:, 0].unsqueeze(1), \
                                 reward_tensor[:, 1].unsqueeze(1)

        # 2) Critic 更新（常に行う）
        with torch.no_grad():
            a_next, logp_next, _ = self.actor_net.sample(s_next)
            q1_t, q2_t = self.critic_net_target(s_next, a_next)
            q_min     = torch.min(q1_t, q2_t) - self.alpha * logp_next
            target_nav  = r_nav  + mask.unsqueeze(1) * self.gamma * q_min
            # 安全性クリティックのターゲット
            qs1_t, qs2_t = self.critic_safe_target(s_next, a_next)
            qs_min       = torch.min(qs1_t, qs2_t)
            target_safe  = r_safe + mask.unsqueeze(1) * self.gamma * qs_min

        if not stage1:
            q1, q2   = self.critic_net(s, a)
            loss_nav = F.mse_loss(q1, target_nav) + F.mse_loss(q2, target_nav)
            q_mean   = torch.min(q1, q2).mean().item()
        else:
            loss_nav = None  # Stage1 は CriticNav 更新をスキップ
            q_mean   = None

        qs1, qs2   = self.critic_safe(s, a)
        loss_safe  = F.mse_loss(qs1, target_safe) + F.mse_loss(qs2, target_safe)

        # Critic
        if loss_nav is not None:
            self.critic_optim.zero_grad()

            loss_nav.backward()
            # <<< INSERT: global grad clip for critic >>下２行
            max_grad_norm = 500.0
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_grad_norm)

            self.critic_optim.step()

        self.critic_safe_optim.zero_grad()

        loss_safe.backward()
        # <<< INSERT: global grad clip for critic_safe >>下１行
        torch.nn.utils.clip_grad_norm_(self.critic_safe.parameters(), max_grad_norm)

        self.critic_safe_optim.step()

        # --- 安全クリティック勾配ノルムを計算して保持 ---
        total_sq = 0.0
        for p in self.critic_safe.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.data.norm(2).item()) ** 2 #p.grad.data.norm(2).item() ** 2　変更したよ
        critic_safe_grad_norm = total_sq ** 0.5 #追加したよ
        self.last_safe_grad_norm = critic_safe_grad_norm #追加したよ


        # 3) Actor 更新
        # stability_loss と safety_loss を定義
        pi, logp_pi, _ = self.actor_net.sample(s)
        q1_pi, q2_pi   = self.critic_net(s, pi)
        q_min_pi       = torch.min(q1_pi, q2_pi)



        stability_loss = (self.alpha * logp_pi - q_min_pi).mean()

        # safety_loss
        qs1_pi, qs2_pi    = self.critic_safe(s, pi)
        safety_loss_unscaled = - torch.min(qs1_pi, qs2_pi).mean()
        safety_loss = self.lambda_safe * safety_loss_unscaled
        #safety_loss_stage = - self.lambda_safe * torch.min(qs1_pi, qs2_pi).mean()



        # Stage判定
        #stage1 = (episode_index <= args['stage1_episodes'])

        if episode_index <= args['stage1_episodes']:
            # ステージ1：安全性のみで Actor 更新

            self.actor_optim.zero_grad()

            safety_loss.backward()
            self.actor_optim.step()

            nav_loss_item   = None
            safe_loss_item  = safety_loss.item()

        else:
            # ステージ2：restricted_direction を用いた制限付き更新
            # a) stability勾配
            self.actor_optim.zero_grad()

            stability_loss.backward(retain_graph=True)
            grad_st = torch.cat([p.grad.view(-1) for p in self.actor_net.parameters()])

            # b) safety勾配

            self.actor_optim.zero_grad()

            safety_loss.backward(retain_graph=True)
            grad_sa = torch.cat([p.grad.view(-1) for p in self.actor_net.parameters()])

            # 生のノルムを記録（正規化前の大きさ）
            st_norm_raw = grad_st.norm().item()
            sa_norm_raw = grad_sa.norm().item()
            self.history_stability_grad_norms.append(st_norm_raw)
            self.history_safe_grad_norms.append(sa_norm_raw)


            # d) 正規化 ＆ restricted_direction

            stage1_eps = float(self.args.get('stage1_episodes', 0))
            stage2_eps = float(self.args.get('stage2_episodes', 1))
            """if episode_index <= stage1_eps:
                progress = 0.0
            else:
                progress = min(1.0, max(0.0, (episode_index - stage1_eps) / max(1.0, stage2_eps)))"""

            progress = min(1.0, (episode_index / args['stage2_episodes'])**2)

            #e = restricted_direction(grad_sa, grad_st,progress=progress, debug=False)
            # --- h(s) をバッチ単位で取得 ---
            with torch.no_grad():
                h_batch = self.h(s)  # shape: [batch_size]

            # --- 危ない方（下位15%）を代表値として採用 ---
            h_value_batch = float(torch.quantile(h_batch, 0.15).item())
            #h_s = agent.h(s).mean().item()#ここから
            # --- 移動平均バッファに追加 ---
            self.h_buffer.append(h_value_batch)
            if len(self.h_buffer) > self.h_buffer_size:
                self.h_buffer.pop(0)

            # --- 移動平均を計算 ---
            h_value = float(np.mean(self.h_buffer))
            e, m_dir = restricted_direction(grad_sa, grad_st, progress=progress, h_value=h_value)#ここまで
            self.last_m_dir = m_dir

            dot   = torch.dot(grad_st, grad_sa).item()
            norm1 = grad_st.norm().item()
            norm2 = grad_sa.norm().item()
            cos_sim = dot / (norm1 * norm2 + 1e-8)

            angle = math.degrees(math.acos(dot/(norm1*norm2+1e-8)))
            #print(f"[DEBUG] dot(W_nav,W_safe)={dot:.3f},cos={cos_sim:.3f}, angle={angle:.1f}°")


        # 展開前に p.grad を上書きするため、ここでは展開後に出力する

            # after building e and before assigning to p.grad (or right after assignment but before optimizer.step)
            max_e_norm = 102.5   # safety bound on the norm of the assembled gradient vector
            e_norm = e.norm().item()
            if e_norm > max_e_norm:
                e = e * (max_e_norm / (e_norm + 1e-12))
            # then write back to p.grad as before and run actor step上４行追加したよ

            # e) 各パラメータ勾配に展開
            idx = 0
            for p in self.actor_net.parameters():
                n = p.numel()
                p.grad = e[idx:idx+n].view_as(p).clone()
                idx += n

            max_grad_norm = 50.0#下２行
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_grad_norm)

            # f) 更新
            self.actor_optim.step()

            nav_loss_item  = stability_loss.item()
            safe_loss_item = safety_loss.item()

        # 4) Entropy α 更新（既存ロジック）
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # 5) ターゲットネットワークのソフト更新
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)
            soft_update(self.critic_safe_target, self.critic_safe, self.tau)

                # === ノルム記録 ===
        # stage1 の場合は安全クリティックの勾配ノルム（self.last_safe_grad_norm が既に保持されている）
        # stage2 の場合は stability と safety 勾配の L2 ノルム（計算済み grad_st, grad_sa を使う）
        try:
            if stage1:
                # stage1: 安全クリティック勾配ノルム（CriticSafe の勾配ノルム）
                self.history_safe_grad_norms.append(self.last_safe_grad_norm if hasattr(self, 'last_safe_grad_norm') else 0.0)
                # stability はこの段階では未更新なので 0 を格納
                self.history_stability_grad_norms.append(0.0)
            else:
                # stage2: actor の stability/safety 勾配ノルム（grad_st, grad_sa は正規化前の値を使うのが望ましい）
                # grad_st, grad_sa はスコープ内で定義されているはずなので取得する
                st_norm = grad_st.norm().item() if 'grad_st' in locals() else 0.0
                sa_norm = grad_sa.norm().item() if 'grad_sa' in locals() else 0.0
                self.history_stability_grad_norms.append(st_norm)
                self.history_safe_grad_norms.append(sa_norm)
        except Exception:
            # 記録でエラーが出ても学習を止めない
            self.history_safe_grad_norms.append(0.0)
            self.history_stability_grad_norms.append(0.0)


        # stage1 の場合 grad_st/grad_sa/dot は未定義の可能性があるため安全値を返す
        try:
            st_norm_raw = st_norm_raw if 'st_norm_raw' in locals() else 0.0
            sa_norm_raw = sa_norm_raw if 'sa_norm_raw' in locals() else self.last_safe_grad_norm if hasattr(self, 'last_safe_grad_norm') else 0.0
            dot_raw     = dot if 'dot' in locals() else 0.0
        except Exception:
            st_norm_raw, sa_norm_raw, dot_raw = 0.0, 0.0, 0.0

        return nav_loss_item, safe_loss_item, critic_safe_grad_norm, st_norm_raw, sa_norm_raw, dot_raw

