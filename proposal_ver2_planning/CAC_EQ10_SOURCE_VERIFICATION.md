# CAC原論文 Eq.10 の原文画像による確認

**本レポートは分析・原文確認のみを目的とする。研究コードは一切変更せず、実験も実行していない。Git操作も行っていない。既存の調査レポートは一切上書きしていない。**

---

## 0. 確認方法

これまでの調査では、CAC原論文PDF（プロジェクト内`Certificated Actor-Critic_...pdf`、arXiv:2501.17424）を`pdftotext`でテキスト抽出しており、数式中のノルム記号・不等号の一部が失われるという限界があった。今回、PyMuPDF（`pymupdf`、pip経由でインストール。研究コード・実験環境には影響しない、本レポート作成のための補助ツール）を用いてPDFページを高解像度画像としてレンダリングし、**該当ページ（3ページ目、Section IV-B "Restricted Policy Update"）を画像として直接目視確認した**。

---

## 1. Eq.10の全文（画像から直接読み取り、確認済み）

```
∇θ = argmax_e  e · ∇θJ2(θ)

     s.t.      e · ∇θJ1(θ) ≥ 0

               ‖e‖ ≤ ‖∇θJ2(θ)‖                          (10)
```

**これは画像を直接目視して確認した内容であり、推測ではない。**

---

## 2. 第3行のノルム制約・不等号・変数（正確な記録）

| 項目 | 内容 |
|---|---|
| 制約式 | `‖e‖ ≤ ‖∇θJ2(θ)‖` |
| 不等号 | `≤`（以下） |
| 左辺 | `‖e‖`——更新方向`e`自身のノルム |
| 右辺 | `‖∇θJ2(θ)‖`——性能（ナビゲーション）勾配のノルム |
| 制約の意味 | 更新方向`e`の大きさを、性能勾配`∇θJ2(θ)`自身のノルム以下に制限する |

**前回レポート（`CAC_VER1_MATHEMATICAL_COMPARISON.md`）で「`‖e‖≤‖∇J2(θ)‖`である可能性が高いが未確認」として推定復元していた内容は、今回の画像確認により正確に一致することが確認できた。** 前回レポートの推定は正しかったが、前回時点では確定した事実として扱っていなかった点を、本レポートで確定させる。

---

## 3. 前後の本文との照合

- Eq.10の直前（本文）："we derive the gradient `∇θ` using **restricted policy update** as" —Eq.10が`∇θ`（次の更新に使う勾配、`e`と同一視される）の導出式であることを確認。
- Eq.10の直後（本文、原文引用）："where `J1(θ), J2(θ)` are actor loss functions with reward `r1, r2` respectively, and `∇` is gradient operator. Under **Assumption 1**, the policy `π*_safe` will update and converge gradually to the final optimal policy `π*` along `∇θ`." —変数の定義および、収束の主張が**Assumption 1**（Parameter Continuity of Safe Policies：安全方策が十分小さい摂動`‖θ'-θ‖≤δ`の範囲でも安全であり続けるという仮定）に基づくものであることを確認。
- 第IV.C.3節「Gradient enhancement」（画像で確認、原文引用）：

  > "Although `∇θ·∇θJ1(θ)≥0` works theoretically, there exist two reasons for possible failure in algorithm implementation. One is there always needs a step length to update the parameters, and local gradient does not guarantee global convergence. Another is the true gradient `∇θJ1(θ)` is unknown, and is replaced by the estimation `∇̃θJ1(θ)` derived from data. Hence, it is essential to enhance the constraint as `∇θ·∇̃θJ1(θ)≥δ` or `cos{∇θ,∇̃θJ1(θ)}≥δ, δ>0`"

  ここで実際に強化されているのは**`∇θ`（最終的に採用される更新方向、Eq.10の解`e`に相当）と、真の勾配`∇θJ1(θ)`ではなく推定勾配`∇̃θJ1(θ)`（データから推定されたもの、チルダ記号で明示）との内積**である。強化されるのは制約の下限（`0→δ`）のみであり、**ノルム制約（第3行）を変更・追加するものではない**ことを画像で確認した。

---

## 4. Eq.10の解法手続きがAlgorithm 1・本文に記載されているかの再確認

**Algorithm 1（画像で全文確認）**：

```
9:  %Stage 2: Restricted Policy Update
10: Set reward r2, initialize the critic network Qφ2 or Vφ2 with parameters φ2 for navigation
11: Define learning rate λφ2, the loss function J2(θ) for actor and L2(φ2) for critic with r2
12: for each step in training do
13:     ∇θ ← (10)
14:     θ ← θ - λθ∇θ
15:     φ1 ← φ1 - λφ1∇φ1L1(φ1)
16:     φ2 ← φ2 - λφ2∇φ2L2(φ2)
17: end for
18: Output: θ, φ1 and φ2
```

**確認できた事実**：13行目は単に「`∇θ ← (10)`」（Eq.10の解を`∇θ`に代入する）と記すのみであり、**Eq.10という制約付き線形計画問題をどのような計算手続き（数値ソルバー名、反復回数、閉形式の公式等）で解くかについて、Algorithm 1にも、これまでに画像確認した本文（Section IV-B、IV-C）にも、具体的な記述は見当たらなかった**。

第IV.C節「Practical Improvements」は3項目（Policy improvement／Exponential reward normalization／Gradient enhancement）から成るが、いずれもEq.10の解法アルゴリズムそのものには言及しておらず、①SACのKL最小化への読み替え、②報酬の指数正規化、③制約の下限強化、という別の実装上の工夫を述べているのみであることを画像で確認した。

**したがって、`CAC_VER1_MATHEMATICAL_COMPARISON.md`第1.4節で述べた「CAC原論文はEq.10の具体的な解法手続きを本文中で明示していない」という記述は、今回の画像確認によって改めて裏付けられた。**

---

## 5. 既存レポートの記述との相違の整理

| 既存レポートの記述 | 今回の画像確認結果 | 相違・扱いの変更 |
|---|---|---|
| `CAC_VER1_MATHEMATICAL_COMPARISON.md`第1.2節：「（推定）ノルム制約」として`‖e‖≤‖∇J2(θ)‖`を推測復元し、「断定できない」と明記 | `‖e‖≤‖∇θJ2(θ)‖`であることを画像で直接確認 | **推測から確認済みの事実へ格上げ**。以後の議論で「推定」の留保を外してよい |
| `LITERATURE_AND_MATHEMATICAL_COMPARISON.md`等、それ以前のレポート群でも同様に`e·W1≥0, ‖e‖≤‖W2‖`という形でCACのEq.10を記述していた（本レポート作成の会話より前の段階の記述） | 今回の画像確認と完全に一致 | 相違なし。これらの記述は結果として正確だった |
| `CAC_VER1_MATHEMATICAL_COMPARISON.md`第1.4節：「Eq.10の解法手続きが本文に明記されていない」 | Algorithm 1・第IV.C節を画像で確認したが、解法手続きの記述は見当たらなかった | 相違なし。従来の結論を維持・再確認 |
| `CAC_VER1_MATHEMATICAL_COMPARISON.md`第1.3節：制約強化`e·∇J1(θ)≥ε`という表記 | 画像確認により、正確には「`∇θ·∇̃θJ1(θ)≥δ`」（`e`ではなく`∇θ`、真の勾配ではなく推定勾配`∇̃θJ1(θ)`とのチルダ付き表記）であることを確認 | **表記の精緻化**。「`e`」ではなく採用済み更新方向`∇θ`との内積であること、対象が真の勾配ではなく推定勾配であることを明示すべき |

**`CAC_VER1_MATHEMATICAL_COMPARISON.md`第5.1節・第6節で述べた「`m1`がCACの制約境界解に対応する可能性がある」という推測、および第5.3節で導出した「`m2`優勢時に`e・W1<0`となりうる」という事実は、いずれもノルム制約の内容が確定したことによって直接的な影響は受けない**（`m1`/`m2`の導出は`e·W1=0`という等式条件のみに基づいており、ノルム制約`‖e‖≤‖∇J2(θ)‖`とは独立した計算であるため）。ただし、CACの真の閉形式解（第7節「不足している数学的情報」参照）を今後導出する際には、このノルム制約を確定した内容として組み込む必要がある。

---

## 6. 未確認のまま残る事項

1. **Eq.10（制約付き線形計画）の完全な閉形式解**：ノルム制約の内容が確定したことで、KKT条件に基づく場合分け解の導出は可能になったが、**本レポートではまだ導出していない**。
2. **CAC実装（コードが公開されていれば）における実際の解法**：論文本文には記述がないため、実装が別途公開されていればその確認が必要だが、本レポートでは未実施。
3. **Assumption 1・Theorem 1とEq.10の解の一意性・存在性の関係**：画像で内容は確認したが、Eq.10の解が常に存在する（実行可能領域が非空である）ことの保証についての議論は、本文中に明示的には見当たらなかった。

---

## まとめ

CAC原論文Eq.10の完全な定式化を、PDF画像の直接確認により確定した：

```
∇θ = argmax_e  e·∇θJ2(θ)
     s.t.      e·∇θJ1(θ) ≥ 0
               ‖e‖ ≤ ‖∇θJ2(θ)‖
```

これは前回レポートの推定と一致しており、以後のProposal Ver.1との数学的比較において、この定式化を**確認済みの事実**として用いてよい。Eq.10の解法手続きが原論文本文に明記されていないという結論も、画像確認により再確認された。
