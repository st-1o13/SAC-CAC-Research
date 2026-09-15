# SAC-CAC-Research

Certified Actor-Critic（CAC）における勾配計算の簡易化に関する研究コードリポジトリです。

---

## 1. 研究概要

強化学習エージェントが安全領域内に留まり続けながら性能を改善する「安全な強化学習（Safe RL）」を対象とした研究です。従来手法である **Certified Actor-Critic（CAC）**[Xie et al., ICRA 2025] は、制御バリア関数（Control Barrier Function, CBF）に基づく安全報酬を用いて安全性を評価し、Stage1（安全学習）→Stage2（拘束付き方策更新）という二段階構造でActorを更新します。本研究では、この二段階構造と、Stage2で毎ステップ解く必要がある制約付き最適化（Restricted Policy Update）に着目し、これらを置き換える簡易な勾配混合手法を提案・検証しています。

## 2. Proposal Ver.1の目的

`Proposal_Ver1.ipynb.ipynb` は、提案手法の現時点での基準版（Ver.1）です。目的は以下の2点です。

1. CACの二段階構造を撤廃し、安全報酬と到達性能報酬を学習初期から同時に扱う統合型の方策更新を実現すること
2. CACのStage2で用いられる制約付き最適化（凸最適化）を、`restricted_direction()` という閉形式・ヒューリスティックな勾配混合関数に置き換え、計算コストを削減すること

## 3. SAC・CBF・CACとの関係

- **SAC（Soft Actor-Critic）**：本研究のActor-Critic学習の基盤アルゴリズム。最大エントロピー強化学習に基づき、Actor・Twin-Q Criticをオフポリシーで学習する。
- **CBF（Control Barrier Function）**：安全度を表す関数 `h(s)` を定義し、`h(s) ≥ 0` を満たす限り系が安全集合内に留まることを保証する制御理論上の枠組み。CACの安全報酬設計の基礎となっている。
- **CAC（Certified Actor-Critic）**：CBFに基づく安全報酬でSafety Criticを学習し、その勾配が「安全性を後退させない」という制約のもとで、到達性能を表すNavigation Criticの勾配を用いてActorを更新する階層的強化学習手法。本研究の直接の比較対象・出発点。

## 4. 現在の提案手法の概要

`Proposal_Ver1.ipynb.ipynb`（および同一内容の `proposal_ver1_py/`）における提案手法の要点は以下の通りです。

- **安全報酬 `r1`**：CAC論文の指数正規化形をそのまま踏襲（`exp(min(h(s')+(α-1)h(s), 0)) - 1`）
- **到達性能報酬 `r2`**：独自の評価関数ではなく、Gymnasium Pendulum-v1環境が返す標準報酬をそのまま使用
- **Safety Critic / Navigation Critic**：いずれもTwin-Q構成（`ClippedCriticNet`）のSAC準拠Critic
- **勾配混合（`restricted_direction()`）**：安全勾配W1・性能勾配W2の内積・ノルムから閉形式で混合係数 `m1`, `m2` を求め、学習進行度 `progress` とバッチ安全度 `h_value` に基づく安全ゲートで両者を動的に混合し、最終的な更新方向を決定する
- **多段階ノルム制御**：対数縮小・ソフト縮小・相対上限・ハードクランプ・複数段階のノルムクリップにより、性能勾配側の過大な更新を抑制する
- **Stage1構造の扱い**：CAC由来のStage1分岐（安全性のみでActor更新する処理）はコード上に残しているが、実験条件 `stage1_episodes = 0` により実質的に発火せず、episode 1から統合型の更新方向計算（W1・W2の混合）が行われる
- **バリア関数パラメータαのスケジューリング**：学習エピソードに応じて0.1→0.9へ線形に変化。なお、この `α` は同時にSACのエントロピー温度としても使用されている（実装上の事実であり、意図的な設計かどうかは未確認）

**要確認・既知の課題**：`SoftActorCriticModel.update_critics_and_actor()` 内の一部箇所は `self.args` ではなく素のグローバル `args` を直接参照しているなど、実装上の細かな不整合が存在する。これらは研究アルゴリズムの結果には影響しないことを確認済みだが、現時点では修正せず、事実として記録している。

## 5. リポジトリ構成

```
SAC_CAC_Research/
├── NoChange.ipynb                 # 初期の研究Notebook（フリーズ済み、変更禁止）
├── CAC_Original.ipynb             # CAC原論文の忠実な実装（フリーズ済み、変更禁止）
├── Cleaned.ipynb                  # NoChange.ipynbを整理した版（フリーズ済み、変更禁止）
├── Proposal_Compare.ipynb         # CACとの公平比較用に条件を揃えた提案手法版
├── CAC_Compare.ipynb              # 提案手法との公平比較用に条件を揃えたCAC版
├── Proposal_Ver1.ipynb.ipynb      # 提案手法Ver.1の基準版（本READMEの主対象）
├── proposal_ver1_py/              # Proposal_Ver1.ipynb.ipynbの.py移植版
├── runs/                          # Proposal_Compareの実験結果（10 run分）
├── runs_cac/                      # CAC_Compareの実験結果（1 run）
├── comparison_analysis/           # Proposal_Compare ⇄ CAC_Compareの比較分析結果
├── 提案手法１０回平均/              # 提案手法10 run分の平均統計・図
├── cac_compare_run.log 等         # 各Notebook実行時のログ
└── 論文 Ver.1  Proposal Ver.1.pdf # 修士論文原稿
```

**Notebookの位置づけの違いに注意**：`NoChange.ipynb` / `CAC_Original.ipynb` / `Cleaned.ipynb` は研究の来歴を保存するための**フリーズ済み参照版**であり、以後変更しない方針です。`Proposal_Compare.ipynb` / `CAC_Compare.ipynb` はCACとの**公平比較実験専用**に条件を揃えた版であり、条件の一部（Stage1/2の長さ、α等）が `Proposal_Ver1.ipynb.ipynb` 本来の実験条件とは異なります。**提案手法そのものの定義・実験条件は、常に `Proposal_Ver1.ipynb.ipynb` を基準としてください。**

## 6. proposal_ver1_py の各ファイルの役割

| ファイル | 役割 |
|---|---|
| `config.py` | `gym_name`/`device`/seedの初期化、`args`辞書、`BASE_SEED`、`total_episodes`、matplotlib backend設定 |
| `replay_memory.py` | `ReplayMemory`（経験再生バッファ） |
| `networks.py` | `ClippedCriticNet`（Twin-Q Critic）, `SoftActorNet`（Actor） |
| `network_utils.py` | `soft_update`, `hard_update`, `convert_network_grad_to_false` |
| `gradient_blend.py` | `restricted_direction()`（提案手法の核心）, `restricted_direction1()`（未使用の旧バリアント、保持のみ） |
| `dynamics.py` | `model_dynamics()`（Pendulumの解析的ダイナミクス、現在未使用、保持のみ） |
| `agent.py` | `SoftActorCriticModel`（h(s)、安全報酬計算、Critic/Actor更新処理） |
| `train.py` | `save_single_run()`, `save_figures_for_run()`, `run_experiment()`。global変数（`episode_*`配列）を共有するため、あえて同一ファイルにまとめている |
| `analyze.py` | `load_all_runs()`、統計print、mean/std図、散布図、実行時間統計・図。こちらもglobal変数共有のため1ファイルにまとめている |
| `main.py` | エントリーポイント（`NUM_RUNS`回のrun_experiment実行ループ） |

`Proposal_Ver1.ipynb.ipynb` の各セルとの対応、および分離にあたっての設計判断（global変数共有・`args`参照の不整合の扱いなど）の詳細は、このリポジトリの作業記録（本README作成時点ではチャット履歴側にのみ存在し、リポジトリ内ドキュメントとしては未整備）を参照してください。

## 7. 実行方法

### Notebook版（基準版）

`Proposal_Ver1.ipynb.ipynb` を、CUDA対応PyTorch・Gymnasiumがインストールされた環境（本研究では `.venv` の Python 3.12 環境）のJupyterカーネルで、セルを上から順に実行してください。`NUM_RUNS`（`run_experiment`呼び出しループのセル）で指定した回数だけ `runs/` に結果が保存されます。

### .py版

```bash
cd proposal_ver1_py
python main.py
```

`main.py` は `NUM_RUNS` 回、`run_experiment(run_id)` を実行します（`run_id` は1から始まる）。結果は既定で `runs/`（`train.py` の `save_single_run`/`save_figures_for_run` のデフォルト`save_dir`）に保存されます。

**現在の実験条件（`config.py` の `args`）は一切変更しないでください**（`stage1_episodes=0`、`alpha`初期値、学習率、`lambda_safe`等はいずれも検証済みの基準条件です）。

## 8. 実験結果データの保存場所

| フォルダ | 内容 |
|---|---|
| `runs/` | `Proposal_Compare.ipynb` によるCACとの公平比較実験（10 run分のnpz・PNG） |
| `runs_cac/` | `CAC_Compare.ipynb` によるCAC側の比較実験（1 run分） |
| `comparison_analysis/` | `runs/` と `runs_cac/` を用いた比較分析結果（CSV, JSON, 図） |
| `提案手法１０回平均/` | 提案手法10 run分の平均・標準偏差の集計図 |

保存されるnpzの主な配列：`episode_rewards_nav`, `episode_rewards_safe`, `episode_h`, `episode_h_value`, `episode_m_dir`, `episode_grad_dot`, `episode_violations`, `episode_safe_grad_stage2`, `episode_stability_grad`, `episode_theta_dot_mean`/`max`/`min`, `runtime_sec`。

## 9. 現在確認済みの再現性・同値性

以下は、いずれも `run_id=1`（`BASE_SEED + run_id` による同一seed）で実際に1回ずつ実行し、保存されたnpz中の対象12配列（`runtime_sec`を除く）を `np.array_equal()` で突き合わせて確認した実測結果です（静的なコード比較ではなく、実行結果そのものの比較です）。

- **Cleaned.ipynb と Proposal_Ver1.ipynb.ipynb**：コード整理（重複診断print・重複import・死んだコメント・未使用グローバル変数の削除、"100 runs"表示の動的化、レガシー動画付録の隔離、旧学習ループの削除）の前後で、12配列すべてが完全一致（`max|diff|=0`, `mean|diff|=0`）。
- **Notebook版（Proposal_Ver1.ipynb.ipynb）と .py版（proposal_ver1_py/main.py）**：Notebookから.pyへの移植前後で、同じく12配列すべてが完全一致（`max|diff|=0`, `mean|diff|=0`）。

いずれの確認も **n=1（1 runのみ）** による実測であり、CUDA演算の非決定性の観点から、別のseed・run_idでの追加確認によってさらに確証を強めることが望ましい状態です。

## 10. 今後の研究予定（※以下は予定であり、実施済みの内容ではありません）

- 別run_idでの再現性の追加確認
- `proposal_ver1_py` の残りのリファクタリング（global変数依存の整理、`self.args`/素の`args`参照の不整合の解消など）
- 計算効率・再現性のさらなる改善
- 複数run（例：10 run以上）による統計的な比較検証
- 修士論文第3章・第4章への実験結果の反映
