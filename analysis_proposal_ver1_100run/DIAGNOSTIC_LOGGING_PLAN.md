# 追加診断ログ 実装計画（未実装・計画のみ）

目的：`m_dir`の異常値（最大283,804.77等）と`grad_dot`の極端な負の外れ値の原因を、内部の中間変数（W1/W2ノルム、m1/m2、分母等）を直接観測することで特定する。

**本ドキュメントは計画のみであり、コードは一切変更・実装していない。**

---

## 0. 設計方針（全体像）

条件「既存のProposal Ver.1コード（`proposal_ver1_py/`内の全ファイル）を直接変更しない」を満たすため、**既存コードには一切手を加えず、新しい診断用ディレクトリから既存モジュールをそのままimportし、実行時パッチ（モンキーパッチ）で必要な値だけを外側から観測する**方式を採る。これはPhase 5（Notebook⇄.py版の実行結果検証）で実際に使用した「`__defaults__`の実行時上書きによりファイルを一切編集せずに保存先だけ変える」手法の延長線上にある。

新設ディレクトリ（案）：

```
proposal_ver1_diag_py/
├── diag_hooks.py     # 監視用ラッパー・パッチ本体
├── diag_main.py       # エントリーポイント（既存main.py相当、診断パッチを当ててからrun_experimentを呼ぶだけ）
```

`proposal_ver1_py/`配下のファイル（`config.py`, `train.py`, `agent.py`, `gradient_blend.py`, `replay_memory.py`, `networks.py`, `network_utils.py`, `main.py`, `analyze.py`）は**1行も変更しない**。

観測対象ごとに、リスクレベルが異なる3種類の取得方式を使い分ける。

| 方式 | リスク | 該当項目 |
|---|---|---|
| **A: 純観測**（既存コードが既に計算して返している値をそのまま横取りするだけ） | 低（数式の再実装なし、実際の値そのもの） | W1 norm, W2 norm, grad_dot（生の内積）, episode_length |
| **B: ライブラリ関数のラップ**（PyTorch自身が返す正確な値を、呼び出しを横取りして取得） | 低〜中（呼び出し順序の識別が必要） | 各種clipping発生フラグ・クリップ前ノルム |
| **C: シャドウ再計算**（既存関数の内部でしか計算されない値を、同じ数式を別途複製して計算） | **中〜高（数式の転記ミスによる乖離リスクあり。要検証）** | m1, m2, denom1, denom2, dot12（shrink後） |

---

## 1. 実装対象ファイルと挿入位置（新規ファイルのみ・既存ファイル無変更）

### 1.1 `proposal_ver1_diag_py/diag_hooks.py`（新規）

以下4つのパッチ関数を用意する。いずれも「対象モジュールをimportした後に、モジュールの名前空間にある関数/メソッドオブジェクトを差し替える」処理であり、`.py`ファイル自体は編集しない。

#### (a) W1 norm / W2 norm / grad_dot（生） — 方式A：純観測

- **対象**：`SoftActorCriticModel.update_critics_and_actor`（`agent.py`で定義）
- **挿入位置**：`agent.SoftActorCriticModel.update_critics_and_actor` というクラス属性を、診断用ラッパーで置き換える（`agent.py`は無変更）
- **仕組み**：この関数は既に`return nav_loss_item, safe_loss_item, critic_safe_grad_norm, st_norm_raw, sa_norm_raw, dot_raw`という形で**W2ノルム(`st_norm_raw`)・W1ノルム(`sa_norm_raw`)・生の内積(`dot_raw`)を返している**（`agent.py:349`）。現状`train.py`は`sa_norm_raw`を受け取ってはいるが保存していないだけであり、**新しい計算は一切不要**。ラッパーは元の関数を呼び出し、戻り値をそのまま診断ログに書き込んでから、**戻り値を一切変更せずにそのまま呼び出し元（`train.py`）へ返す**。これによりtrain.py側の挙動・学習結果は完全に不変。

#### (b) episode_length（違反率算出用） — 方式A：純観測

- **対象**：`gym.make()`が返す環境オブジェクト
- **挿入位置**：`train.gym.make`（`train.py`の名前空間内で`import gymnasium as gym`によって束縛された`gym`モジュールの`make`関数）を、診断用ラッパーで置き換える
- **仕組み**：`gym.make()`の戻り値を`gymnasium.Wrapper`で1枚ラップし、`step()`が呼ばれるたびに内部カウンタを+1、`reset()`で0に戻すだけの薄いラッパーを被せる。観測・状態・報酬・終了判定はすべて元の環境にそのまま委譲するため、学習ロジックへの影響はない（ただしRNG消費順序に影響しないことは要検証。詳細はリスク参照）。

#### (c) クリッピング発生フラグ — 方式B：ライブラリ関数のラップ

- **対象**：`torch.nn.utils.clip_grad_norm_`（`agent.py`内で3箇所呼ばれている：critic_net用500.0、critic_safe用500.0、actor_net用50.0）
- **挿入位置**：診断スクリプトの実行時、`torch.nn.utils.clip_grad_norm_`をグローバルにラップする（診断実行の間だけ有効にし、実験終了後に元へ戻す）
- **仕組み**：`torch.nn.utils.clip_grad_norm_`は**クリップ前の合計ノルムを戻り値として返す**（PyTorch公式仕様）。現状の`agent.py`はこの戻り値を受け取っていない（`torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_grad_norm)`のように戻り値を破棄）。ラッパーは元の関数を呼び出してその戻り値（クリップ前ノルム）を横取りし、渡された`parameters`引数を`agent.actor_net`／`agent.critic_net`／`agent.critic_safe`のいずれのパラメータと一致するかで識別した上で診断ログに記録し、**元の関数の戻り値・挙動をそのまま呼び出し元に返す**。
- **`e`（restricted_direction後の合成ベクトル）に対する102.5クリップ**：この処理（`agent.py:285-288`）は`torch.nn.utils.clip_grad_norm_`を使わない独自コードのため、上記のラップでは捕捉できない。これは (d) のシャドウ計算側で、`restricted_direction`の戻り値`e`のノルムを観測することで対応する（`e.norm() > 102.5`かどうかは、実際に返ってきた`e`を見て判定するだけなので方式Aに近い低リスク）。

#### (d) m1 / m2 / denom1 / denom2 / dot12（shrink後） — 方式C：シャドウ再計算（要検証）

- **対象**：`restricted_direction`関数（`gradient_blend.py`で定義）
- **挿入位置**：`agent.restricted_direction`（`agent.py`内で`from gradient_blend import restricted_direction`により束縛された名前）を、診断用ラッパーで置き換える（`gradient_blend.py`, `agent.py`とも無変更）
- **仕組み**：
  1. ラッパーはまず**元の`restricted_direction`関数をそのまま呼び出し**、実際に学習で使われる`(e, m_dir)`を得る（この呼び出し経路は一切変更しない＝学習結果への影響ゼロ）。
  2. 続けて、**ログ専用の目的だけ**で、`gradient_blend.py`の54〜68行目（`dot12`, `w1_sq`, `w2_sq`, `denom1`, `denom2`, `m1`, `m2`の計算式）を**そのまま複製した読み取り専用の計算**を行い、これらの中間値を得る。この複製計算の出力は学習には一切使われず、ログにのみ書き込む。
  3. **自己検証**：複製計算で得られる`m_dir`（クランプ前後含む）が、元の関数が実際に返した`m_dir`と一致するかを毎回アサーションで確認する。不一致が発生した場合は即座に警告ログを出し、その旨を診断結果に明記する（転記ミスの早期発見のため）。

---

## 2. 保存するデータ形式（提案）

既存の`runs_proposal_ver1/run_*.npz`（12配列、エピソード単位）とは**別ファイル・別ディレクトリ**に保存し、混在させない。

```
runs_proposal_ver1_diag/
├── run_{run_id}.npz          # 既存の12配列（save_single_run/save_figures_for_runをそのまま呼ぶ。回帰確認用）
├── run_{run_id}_update_log.csv   # update単位（最も細かい粒度）の診断ログ
└── run_{run_id}_episode_log.csv  # episode単位の集計ログ（episode_length等）
```

### 2.1 `run_{run_id}_update_log.csv`（update呼び出し1回＝1行、Stage2のみ発生。1 runあたり最大約49,000行）

| 列名 | 内容 | 取得方式 |
|---|---|---|
| episode | エピソード番号 | 観測 |
| update_index_in_episode | エピソード内でのupdate通し番号 | 観測 |
| W1_norm | 安全方向Actor勾配のノルム（`sa_norm_raw`） | A |
| W2_norm | 性能方向Actor勾配のノルム（`st_norm_raw`） | A |
| grad_dot_raw | shrink前の生の内積（`dot_raw`） | A |
| dot12_shrunk | shrink後のW1・W2内積 | C |
| w1_sq / w2_sq | 各ノルムの二乗 | C |
| denom1 / denom2 | m1, m2の分母 | C |
| m1 / m2 | 混合係数の2成分 | C |
| m_dir_shadow | シャドウ計算によるm_dir（クランプ後） | C |
| m_dir_real | 実際に`restricted_direction`が返したm_dir | A（既存の返り値そのもの） |
| m_dir_mismatch | `m_dir_shadow`と`m_dir_real`の不一致フラグ（自己検証） | 派生 |
| e_norm_pre_agentclip | `restricted_direction`が返した`e`のノルム（agent.py側102.5クリップ前） | A |
| e_clip_triggered | `e_norm_pre_agentclip > 102.5` | 派生（低リスク） |
| actor_grad_norm_pre_clip | `torch.nn.utils.clip_grad_norm_`がactor_netに対して返した実際のクリップ前ノルム | B |
| actor_clip_triggered | `actor_grad_norm_pre_clip > 50.0` | 派生（低リスク） |
| h_value_batch | そのupdateでの下位15%分位点 | A（既存内部値の観測） |
| progress | そのepisodeでのprogress値 | A |

### 2.2 `run_{run_id}_episode_log.csv`（エピソード1つ＝1行、250行/run）

| 列名 | 内容 |
|---|---|
| episode | エピソード番号 |
| episode_length | そのエピソードの環境ステップ数 |
| episode_violations | 違反回数（**既存npzと同じ値**、突合確認用に併記） |
| violation_rate | `episode_violations / episode_length`（新規算出。既存データ＋episode_lengthのみで計算可能、新規の学習中ログは不要） |

形式はCSV（人間可読・pandasで即分析可能）を第一候補とする。1 runあたりの行数増（約49,000行×十数列）を考慮し、行数が特に多くなる場合はParquet形式への切り替えも検討可とするが、まずは診断段階のためCSVで十分と判断する。

---

## 3. 診断実験の実行計画（1 → 5 → 10 run）

いずれの段階でも**新しい保存先（`runs_proposal_ver1_diag/`）にのみ書き込み、既存の`runs_proposal_ver1/`・`runs/`・`runs_cac/`は一切変更しない**。

### Stage A：n=1（run_id=1）— パイプライン正当性の検証

1. 診断パッチを当てた状態で`run_experiment(1, save_dir="runs_proposal_ver1_diag")`を実行
2. **最重要の回帰確認**：出力される`run_1.npz`（既存12配列）が、既存の`runs_proposal_ver1/run_1.npz`と`np.array_equal`で完全一致するか検証（Phase 3/5と同じ12配列比較プロトコル）。**一致しなければ診断パッチが学習に影響を与えている証拠であり、即座に中断してパッチ実装を見直す。**
3. `m_dir_mismatch`フラグが1回も立たないこと（シャドウ計算の自己検証が全て一致）を確認
4. `update_log.csv`・`episode_log.csv`が正常に出力され、行数・値の範囲が妥当であることを確認
5. 実行時間の増加幅を計測（ログ書き込みによるオーバーヘッド把握）

Stage Aで問題があれば、Stage B・Cには進まない。

### Stage B：n=5 — 複数run・複数seedでの安定性確認

1. run_id=1〜5を診断モードで実行
2. 各runについてStage Aと同じ12配列一致確認・m_dir_mismatchゼロ確認を行う
3. ディスク使用量・実行時間増加が許容範囲かを確認
4. **可能であれば、既存100-run分析で異常が確認された run_id=54, 60, 83（m_dir異常）および run_id=3, 7, 55（grad_dot異常）のうち数件を、この段階で優先的に含める**（同一seed設計のため、同じrun_idを再実行すれば同じ乱数系列となり、既知の異常エピソードを再現できる可能性が高い。これにより実際にその瞬間のm1/m2/denomの値を直接確認できる）

### Stage C：n=10 — 実運用規模での最終確認

1. run_id=1〜10（またはStage Bの結果次第で異常run_idを含む10件）を実行
2. 全runで12配列一致・mismatchゼロを再確認
3. ログファイル総サイズ・総実行時間を計測し、100-run全体への外挿見積もりを行う
4. ここまで全て問題なければ、初めて「本実装として`proposal_ver1_diag_py`を採用してよいか」を判断する材料が揃う

**100-run全体への診断ログ付き再実行は、本計画のスコープ外とし、Stage Cの結果を見てから別途判断する。**

---

## 4. リスクと留意点

| リスク | 内容 | 対応・軽減策 |
|---|---|---|
| **シャドウ計算の転記ミス（中〜高）** | `restricted_direction`内部の数式を手動で複製するため、実装ミスがあれば診断値そのものが誤る | 毎update呼び出しで`m_dir_shadow`と`m_dir_real`の一致を自動検証し、不一致を検知したら即座に警告 |
| **学習結果への影響（最重要）** | パッチの実装ミスにより、意図せず学習経路（実際に使われる`e`, `p.grad`等）を変えてしまう可能性 | Stage Aで既存npz結果との`np.array_equal`完全一致を必須条件とする。一致しない限り先に進まない |
| **`torch.nn.utils.clip_grad_norm_`のグローバルパッチ** | この関数はプロセス全体でパッチされるため、critic/critic_safe/actorの3箇所の呼び出しを引数のパラメータ集合から正しく識別する必要がある | `agent.actor_net.parameters()`等との同一性判定ロジックを実装し、識別できなかった呼び出しは「unknown」として明示的に記録（黙って誤分類しない） |
| **環境ラッパーによるRNG消費順序への影響** | `gym.Wrapper`でstep数を数えるだけの薄いラッパーでも、理論上は挙動に影響しうる | Stage Aの12配列完全一致確認により、影響がないことを実測で確認する（影響があれば即座に判明する） |
| **ログ量・実行時間の増大** | update単位ログは1 runあたり最大約49,000行。100-run規模に拡大すると数百万行規模になりうる | 本計画では10 runまでに留め、100-run全体への適用は別途コスト評価の上で判断する |
| **診断コードとProposal Ver.1本体のバージョン乖離** | `proposal_ver1_py/`側が将来変更された場合、シャドウ計算（複製コード）が古いままになるリスク | シャドウ計算部分に「`gradient_blend.py`の何行目を複製したものか」を明記するコメントを残し、Ver.1側の変更時に追随が必要な旨をREADME等に明記する |
| **既存ファイルへの誤上書き** | 保存先指定ミスで`runs_proposal_ver1/`や`runs/`を上書きする可能性 | Stage Aの前に保存先パスを`runs_proposal_ver1_diag/`に固定し、既存ディレクトリとファイル名・パスが重複しないことを事前確認する |

---

## 5. 未実施事項の確認

- `proposal_ver1_diag_py/`ディレクトリ・ファイルはまだ作成していない
- 上記いずれのパッチ・診断実験も実行していない
- `runs_proposal_ver1_diag/`は存在しない（まだ何も生成していない）
- `proposal_ver1_py/`配下の既存ファイルは無変更

この計画についてご確認・修正指示をいただいた上で、実装（Stage Aの`n=1`から）に着手します。
