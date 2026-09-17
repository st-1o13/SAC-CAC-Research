# Stage B拡張診断 設計書（計画のみ・未実装）

Stage B（n=5：run_id 54, 60, 83, 3, 7）で、m_dir異常が`denom1`/`denom2`のゼロ（近傍）割りに起因すること、および`e_norm_pre_agentclip`が最大でも11.85程度と小さいまま保たれていたことを確認した。本書は、この「m1/m2は数百万〜数千万まで発散するのに、最終的な`e`はなぜ小さいままなのか」を`restricted_direction()`の後段パイプラインに沿って追跡するための追加ログ設計である。

**コード変更・実験実行・commitはまだ行っていない。**

---

## 1. 追加すべきログ項目と 2. 取得する具体的な位置

`gradient_blend.py`の`restricted_direction()`本体（`agent.py`から見て`restricted_direction`呼び出し1回の内部）を、既存のシャドウ計算（`diag_shadow.py`の`compute_shadow_values()`、m_dirクランプまでをカバー）から**さらに後段まで延長**する。以下は該当関数の実際のコード（`gradient_blend.py` 92-126行目）に対応する。

| # | 項目 | 該当コード（`restricted_direction()`内） | 取得方式 |
|---|---|---|---|
| 1 | m_dir | `m_dir = safety_level*m1 + (1-safety_level)*m_progress`（クランプ後） | 方式C（既存・延長不要、Stage Bで実装済み） |
| — | w_dir | `w_dir = m_dir*W1 + (1-m_dir)*W2_use` | 方式C（新規、シャドウ延長） |
| 2 | m_soft | `m0 = dot(W2_use,w_dir)/‖w_dir‖²` → `m_shrunk`（`m_thresh=100`によるソフト閾値）→ `m_soft = sign(m0)*m_shrunk` | 方式C（新規、シャドウ延長） |
| 3 | rel_cap | `rel_cap = k_rel*(‖W2_use‖+1e-12)`（`k_rel=4.5`） | 方式C（新規、シャドウ延長。ただし`W2_use`のノルムのみに依存する単純な式で、複雑な分岐がないため相対的に低リスク） |
| 4 | m_scale（クランプ前後） | `m_rel = sign(m_soft)*min(|m_soft|, rel_cap)`（クランプ前に相当する中間値）→ `m_scale = clamp(m_rel, -21, 21)`（クランプ後） | 方式C（新規、シャドウ延長） |
| 5 | e_norm_pre_agentclip | `w_scaled = m_scale*w_dir` → `if ‖w_scaled‖>400: w_scaled *= 400/‖w_scaled‖` → これが関数の戻り値`e` | **方式A（既存）**：Stage A/Bで既に`e.norm()`として取得済み。追加ログ不要 |
| 6 | e_norm_post_agentclip | `agent.py`側：`if e_norm>102.5: e = e*(102.5/e_norm)` | **新規だが方式A′（低リスク・派生値）**：102.5は固定定数であり、`e_norm_post_agentclip = min(e_norm_pre_agentclip, 102.5)`として**新たなフックなしに事後計算で導出可能**（下記「実際のActor更新ベクトル」参照） |
| 7 | eとW1/W2の内積 | `dot(e, W1)`, `dot(e, W2)` | 方式A（新規だが低リスク）：`restricted_direction`ラッパー内で、既に保持している`e`（agent.py側102.5クリップ**前**の値）とラッパー引数の`W1`,`W2`から`torch.dot()`するだけ。agent.py側クリップ**後**の内積は、102.5クリップが方向を変えないスカラー倍であることを利用し、`dot_post = dot_pre * min(1, 102.5/e_norm_pre_agentclip)`として事後計算で導出（新規フック不要） |
| 8 | 実際のActor更新ベクトルのノルム | `torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 50.0)`直前の`p.grad`（= agent.py側102.5クリップ後の`e`を展開したもの）、および`self.actor_optim.step()`前後のパラメータ差分 | 下記「3. 実際のActor更新ベクトルを取得できるか」参照 |
| 9 | 異常発生前後の変化 | 上記全項目の時系列（update_index_in_episode順） | 新規フック不要。既存のupdate単位CSVの粒度で十分（Stage Bのデータで既に「前後2件」等の分析実績あり）。**分析手法の問題であり、ログ項目の追加ではない** |

### シャドウ延長部分の自己検証（重要）

既存のStage A/Bでは「shadow計算したm_dir」と「本物のrestricted_direction()が返したm_dir」の一致のみを検証していた。今回m_soft以降まで延長するにあたり、**検証範囲を最終出力ベクトルまで広げる**：

- `torch.allclose(shadow_e, real_e)`（要素ごとの一致）
- `shadow_e.norm()` と `real_e.norm()`（= 既存の`e_norm_pre_agentclip`）の一致
- 上記のいずれかが閾値を超えて不一致の場合、既存の`m_dir_mismatch`と同様に`chain_mismatch`フラグを記録し、以降の`m_soft`/`rel_cap`/`m_scale`の値は「検証済みではない」ものとして扱う

---

## 3. 実際のActor更新ベクトルを取得できるか

**取得可能。2段階の粒度で提案する。**

### (a) クリップ後の勾配ノルム（低リスク・優先実装）

Stage Bで既に`torch.nn.utils.clip_grad_norm_`をラップしており（Method B）、`actor_net`向け呼び出しの**クリップ前**ノルムは取得済み。これを拡張し、**実関数呼び出し後に`params`の`.grad`を再度`.norm()`で観測**するだけで、クリップ後（＝実際に`self.actor_optim.step()`に渡される）ノルムが得られる。`clip_grad_norm_`は対象パラメータの`.grad`を直接書き換える（in-place）仕様のため、呼び出し直後に読み直せば正確な値が取れる。**新しい数式の複製は一切不要**（既存テンソルを読むだけ）。

### (b) Adamステップ後の実際のパラメータ変化量（中リスク・任意実装）

「勾配のノルム」と「Adamが実際にパラメータへ加える変化量」は、Adamの適応的スケーリング（一次・二次モーメント推定）により**必ずしも比例しない**。真に「Actorがどれだけ動いたか」を見るには、`self.actor_optim.step()`の前後で`actor_net`の全パラメータをスナップショットし、差分ノルムを取る必要がある。

- **フック位置**：`update_critics_and_actor`ラッパー（既存、`self`＝agentインスタンスを取得済み）内で、そのrunの最初の呼び出し時に一度だけ`self.actor_optim.step`をインスタンス単位でラップする（`self.actor_optim.step = wrapped_step`）。ラッパーは「`actor_net`の全パラメータの`.data.clone()`を取る → 本物の`step()`を呼ぶ → 再度全パラメータを読んで差分の全体ノルムを計算する」という**観測のみ**の処理。
- 学習ロジック自体（Adamの計算過程）には一切介入しない。

---

## 4. モンキーパッチで取得する場合のリスク

| 対象 | リスク分類 | 内容と軽減策 |
|---|---|---|
| シャドウ計算の延長（w_dir, m0, m_soft, rel_cap, m_scale） | **中〜高**（既存のm1/m2/m_dirと同種） | 数式転記ミスのリスク。軽減策：最終`e`ベクトル全体を`torch.allclose`で比較する`chain_mismatch`検証を必須化（スカラーm_dirだけでなく、ベクトル全体・ノルムまで一致させる、より厳格な検証） |
| e_norm_post_agentclip / eとW1・W2の内積（agent.py側クリップ後） | **低**（フック追加ではなく既知の定数102.5を使った事後計算） | agent.py側の`max_e_norm`が将来変更された場合、この事後計算がサイレントに誤った値を出す可能性。軽減策：`e_clip_triggered`フラグ（既存）との整合性を突き合わせるアサーションを追加し、矛盾があれば警告 |
| `clip_grad_norm_`ラッパーの拡張（クリップ後ノルム観測） | **極めて低** | 既に稼働実績のあるMethod Bの単純な拡張。実測値をもう一度読むだけで、アルゴリズムには一切触れない |
| `self.actor_optim.step`のインスタンス単位パッチ（8b） | **中**（新しい種類のパッチ） | ①パラメータの`.clone()`をエピソード当たり最大200回×全パラメータ分行うため、n=1検証時点でのメモリ・実行時間オーバーヘッドの計測が必須。②ラッパー実装ミスで本物の`step()`呼び出しを省略/二重実行してしまうと学習そのものが壊れる重大リスクのため、**Stage Aと同じ「12配列完全一致」を必須のゲートとして継続**する。③Adamの内部状態（`state`辞書のモーメント推定等）を誤って読み書きしないよう、`.step()`の引数・戻り値には一切手を加えない設計とする |
| 既存Stage B資産の上書き | 低（運用上の注意） | 新しい保存先ディレクトリ（例：`runs_proposal_ver1_diag_stageB_ext/`）を新設し、既存の`runs_proposal_ver1_diag_stageB/`は一切上書きしない |

---

## 5. 追加診断で判明する可能性のあること

1. **m1/m2の発散がどの段階で吸収されているか**：`m_soft`（`m_thresh=100`のソフト閾値）と`rel_cap`（`k_rel=4.5×‖W2_use‖`）のどちらが実質的な抑制の主因かを切り分けられる。現状は「最終的に小さい」ことしか分かっておらず、どのステップで桁が落ちているかは未確認。
2. **`e`が実際に安全方向W1・性能方向W2のどちらを向いているか**：`dot(e,W1)`, `dot(e,W2)`の符号・大きさから、m1/m2が数百万に発散していた瞬間に、最終的な更新方向が実際に安全側（W1）を向いていたのか、性能側（W2）を向いていたのか、あるいは両方からほぼ無関係な方向になっていたのかを直接確認できる。restricted_directionの設計意図（安全性を損なわない範囲での更新）が実際に守られているかの直接的な検証材料になる。
3. **勾配ノルムと実パラメータ変化量の乖離**：クリップ後の勾配ノルムは小さくても、Adamの適応的スケーリング（特に二次モーメント推定が小さいパラメータ）によって、実際のパラメータ変化が相対的に大きくなっている可能性がある。これが確認されれば、「勾配は制御されているが実質的な学習への影響は無視できない」という、現在の推測とは異なる結論になり得る。
4. **異常発生の時間的パターン**：異常update前後で`m_soft`/`rel_cap`/`m_scale`が滑らかに変化しているか、それとも単一updateだけの孤立したスパイクかを確認できる（Stage Bのepisode平均データでは既に「孤立スパイク」との整合が見えているが、m_soft以降の段階でも同様かは未確認）。

---

## 6. Stage C（n=10）を先に実行する必要性

**先に実行する必要はないと判断する。**

理由：
- Stage C（n=10）は元の計画（`DIAGNOSTIC_LOGGING_PLAN.md`）において「パイプライン全体のスケール確認」を目的としており、run_idは任意（1〜10連番、または既知異常runを含める案）だった。
- 一方、今回の拡張診断の目的は「Stage Bで**既に異常発生位置が判明している**5つのrun_id・episodeにおいて、その内部メカニズムをさらに深掘りすること」であり、**任意のrunを増やしても新しい知見は得られにくい**。既知の異常が起きる正確なrun_id・episodeが分かっている以上、そこを再現・深掘りする方が診断効率が高い。
- 推奨する順序：①拡張フックをn=1（run_id=1、Stage Aと同一）で実装・検証（12配列完全一致＋拡張シャドウ検証`chain_mismatch`ゼロを確認）→②検証済みの拡張フックで、Stage Bと**同じ5つのrun_id**（54, 60, 83, 3, 7）を再実行し、今回の重点確認項目を取得する（便宜上「Stage B-extended」と呼ぶ）。
- Stage C（任意n=10への拡大）は、この拡張診断の結果を見て「100-run全体への本格適用を検討する段階」になった時点で、別途判断すればよく、**現時点で先行させる技術的必然性はない**。

---

## 7. 未実施事項の確認

- `diag_shadow.py`・`diag_hooks.py`・`diag_main.py`・`diag_main_stageB.py`はいずれも無変更
- 新しいフック・拡張シャドウ計算はまだ実装していない
- `runs_proposal_ver1_diag_stageB_ext/`（案）は存在しない
- `proposal_ver1_py/`・`runs_proposal_ver1/`・既存の`runs_proposal_ver1_diag/`・`runs_proposal_ver1_diag_stageB/`はいずれも無変更

この設計についてご確認・修正指示をいただいた上で、実装（まずn=1検証から）に着手します。
