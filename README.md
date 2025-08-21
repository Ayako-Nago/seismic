# README

## 概要

本リポジトリでは、Radon変換を用いた地震データの雑音除去アルゴリズムを実装しています。
特に **Primal-Dual Splitting (PDS)** 法に基づき、観測データに対する **L1正則化 + Radon領域制約 + L2 fidelity制約** を組み合わせた新しい定式化を扱います。

## 数学的定式化

問題設定は次の最適化問題として与えられます：

$$
\min_{\mathbf{d}, \mathbf{a}} \|\mathbf{d}\|_1 + \lambda_1\|\mathbf{a}\|_1 + \frac{\lambda_2}{2}\|\mathbf{d}-\mathbf{La}\|_2^2
\quad \text{s.t. } \|\mathbf{d}-\tilde{\mathbf{d}}\|_2 \leq \varepsilon
$$

* $\tilde{\mathbf{d}}$：観測データ
* $\mathbf{d}$：所望のデータ
* $\mathbf{a}$：Radonドメインの係数
* $\mathbf{L}$：Radon変換演算子

制約付き最小化は指示関数を用いて

$$
\min_{\mathbf{d}, \mathbf{a}} \|\mathbf{d}\|_1 + \lambda_1\|\mathbf{a}\|_1
+ \frac{\lambda_2}{2}\|\mathbf{d}-\mathbf{La}\|_2^2 + \iota_{B_{2,\varepsilon}}(\mathbf{d})
$$

と表されます。ここで

$$
B_{2,\varepsilon} := \{ \mathbf{d} \mid \|\mathbf{d}-\tilde{\mathbf{d}}\|_2 \leq \varepsilon \}
$$

です。

## PDS形式への変換

変数を

$$
\mathbf{x} = \begin{bmatrix}\mathbf{d}\\ \mathbf{a}\end{bmatrix},
\quad \mathbf{M} = \begin{bmatrix}\mathbf{I} & -\mathbf{L}\end{bmatrix}
$$

とすると、
$$
f(\mathbf{x}) = \frac{\lambda_2}{2}\|\mathbf{Mx}\|_2^2,
\quad g(\mathbf{x}) = \iota_{B_{2,\varepsilon}}(\mathbf{d}) + \lambda_1\|\mathbf{a}\|_1,
\quad h(\mathbf{Ax}) = \|\mathbf{d}\|_1
$$
の形に書き換えられます。ここで $\mathbf{A} = [\mathbf{I}, 0]$ です。

勾配は

$$
\nabla f\!\begin{bmatrix}\mathbf{d}\\ \mathbf{a}\end{bmatrix}
= \lambda_2 \begin{bmatrix}\mathbf{d}-\mathbf{La} \\ -\mathbf{L}^\top(\mathbf{d}-\mathbf{La})\end{bmatrix}.
$$

## 実装ファイル

* `CRST.py`: Radon2Dを用いた雑音データに対する PDS 法の基本実装
* `radon_proposed.py`: 提案手法（$\ell_1$-$\ell_1$-$\ell_2$ 定式化＋制約）の完全版。PSNR, SNR, CC, RMSE を計算
* `SRT.py`: ISTA に基づく Radon Sparse Reconstruction（Semblance weightingあり）
* `README.md`: 本ファイル（理論と実装の対応関係を記載）

## 依存関係

* Python 3.x
* numpy, scipy
* matplotlib
* pylops

## 実行例

```bash
python CRST.py
python radon_proposed.py
python SRT.py
```

各スクリプトの実行により、雑音データ・復元データ・Radonドメインの可視化および定量評価が出力されます。
