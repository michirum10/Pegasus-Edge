"""価値最大化カスタム目的関数 (設計書 §3-§4 の実装)。

レース内 softmax 確率 p = softmax(log q + F) に対し、

  L = -(1/R) Σ_r log(max(W_r, ε)) + λ CE + (露出はステーキング関数で制御)
  W_r = 1 + Σ_i f_i c_i,  c_i = y_i O_i - 1 (実払戻。payout=100*O_i は検証済み)
  f_i = κ σ(β(e_i-τ)) softplus(e_i)/(O_i-1),  e_i = p_i O_i - 1

勾配・ヘシアン (LightGBM へ渡す対角 Gauss-Newton 近似):

  a_i = c_i f'_i O_i p_i,  A = Σ a_i
  g_k = -(1/W)(a_k - p_k A) + λ(p_k - y_k)
  h_k = (1/W²)(a_k - p_k A)² + λ p_k(1-p_k) + ε_h

市場アンカーは LightGBM の init_score を使わず、クロージャが保持する
log q をスコアに加算して実現する (predict 時の挙動差異を避けるため)。

self-test:
  py -m src.models.value_objective   # 有限差分との勾配一致を検証
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS_HESS = 1e-6


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def race_groups(race_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """連続配置済み race_id 列から (各レース先頭 index, 頭数) を返す。

    呼び出し側はレース単位で行が連続するようソート済みであること。
    """
    change = np.flatnonzero(race_ids[1:] != race_ids[:-1]) + 1
    starts = np.concatenate(([0], change))
    sizes = np.diff(np.concatenate((starts, [len(race_ids)])))
    return starts, sizes


def softmax_by_race(scores: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    smax = np.repeat(np.maximum.reduceat(scores, starts), sizes)
    z = np.exp(scores - smax)
    denom = np.repeat(np.add.reduceat(z, starts), sizes)
    return z / denom


def kelly_stakes(
    p: np.ndarray,
    odds: np.ndarray,
    *,
    kappa: float,
    beta: float,
    tau: float,
    e_max: float = 3.0,
) -> np.ndarray:
    """Soft-Kelly ステーキング (バックテストと学習で同一の式を使う)。"""
    e = np.minimum(p * odds - 1.0, e_max)
    gate = sigmoid(beta * (e - tau))
    return kappa * gate * softplus(e) / np.maximum(odds - 1.0, 0.1)


@dataclass
class ValueObjective:
    """LightGBM 4.x 用 callable objective。params["objective"] に渡す。

    アニーリング: warmup_rounds は純CE (λ=1, κ=0)。その後 total_rounds まで
    λ: 1→lambda_end, β: beta_start→beta_end を線形に動かし価値項を有効化する。
    """

    logq: np.ndarray
    odds: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    kappa: float = 0.10
    tau: float = 0.08
    beta_start: float = 2.0
    beta_end: float = 8.0
    lambda_end: float = 0.2
    warmup_rounds: int = 150
    total_rounds: int = 500
    w_floor: float = 0.05
    e_max: float = 3.0
    iteration: int = field(default=0, init=False)

    def _schedule(self, it: int) -> tuple[float, float, float]:
        if it < self.warmup_rounds:
            return 1.0, 0.0, self.beta_start
        span = max(1, self.total_rounds - self.warmup_rounds)
        t = min(1.0, (it - self.warmup_rounds) / span)
        lam = 1.0 + (self.lambda_end - 1.0) * t
        beta = self.beta_start + (self.beta_end - self.beta_start) * t
        return lam, self.kappa, beta

    def __call__(self, preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        lam, kappa, beta = self._schedule(self.iteration)
        self.iteration += 1
        grad, hess = self.grad_hess(preds, lam=lam, kappa=kappa, beta=beta)
        return grad, hess

    def grad_hess(
        self, preds: np.ndarray, *, lam: float, kappa: float, beta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        p = softmax_by_race(self.logq + preds, self.starts, self.sizes)
        grad_ce = p - self.y
        hess_ce = p * (1.0 - p)

        if kappa == 0.0:
            return lam * grad_ce, lam * hess_ce + EPS_HESS

        e_raw = p * self.odds - 1.0
        e = np.minimum(e_raw, self.e_max)
        gate = sigmoid(beta * (e - self.tau))
        sp = softplus(e)
        denom = np.maximum(self.odds - 1.0, 0.1)
        f = kappa * gate * sp / denom
        fprime = kappa / denom * (beta * gate * (1.0 - gate) * sp + gate * sigmoid(e))
        fprime = np.where(e_raw > self.e_max, 0.0, fprime)

        c = self.y * self.odds - 1.0
        w = 1.0 + np.add.reduceat(f * c, self.starts)
        # フロア以下では -log(max(W,ε)) の勾配は厳密に 0
        inv_w = np.repeat(np.where(w > self.w_floor, 1.0 / np.maximum(w, self.w_floor), 0.0),
                          self.sizes)

        a = c * fprime * self.odds * p
        a_sum = np.repeat(np.add.reduceat(a, self.starts), self.sizes)
        core = a - p * a_sum

        grad = -inv_w * core + lam * grad_ce
        hess = (inv_w * core) ** 2 + lam * hess_ce + EPS_HESS
        return grad, hess

    def loss(self, preds: np.ndarray, *, lam: float, kappa: float, beta: float) -> float:
        """有限差分検証・評価用のスカラー損失 (合計, 平均化しない)。"""
        p = softmax_by_race(self.logq + preds, self.starts, self.sizes)
        ce = -np.sum(self.y * np.log(np.maximum(p, 1e-300)))
        if kappa == 0.0:
            return lam * ce
        f = kelly_stakes(p, self.odds, kappa=kappa, beta=beta, tau=self.tau, e_max=self.e_max)
        c = self.y * self.odds - 1.0
        w = 1.0 + np.add.reduceat(f * c, self.starts)
        kelly = -np.sum(np.log(np.maximum(w, self.w_floor)))
        return kelly + lam * ce


def _self_test(seed: int = 7, n_races: int = 5, tol: float = 1e-6) -> None:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(6, 18, size=n_races)
    race_ids = np.repeat(np.arange(n_races), sizes)
    starts, sizes_arr = race_groups(race_ids)
    n = race_ids.size

    odds = np.exp(rng.normal(1.5, 1.0, size=n)).clip(1.1, 200.0)
    logq = np.log(softmax_by_race(-np.log(odds), starts, sizes_arr))
    y = np.zeros(n)
    for s, k in zip(starts, sizes_arr):
        y[s + rng.integers(0, k)] = 1.0

    obj = ValueObjective(logq=logq, odds=odds, y=y, starts=starts, sizes=sizes_arr)
    preds = rng.normal(0.0, 0.5, size=n)

    for lam, kappa, beta, label in [
        (1.0, 0.0, 2.0, "warmup(純CE)"),
        (0.5, 0.1, 4.0, "価値項あり"),
        (0.2, 0.2, 8.0, "価値項支配"),
    ]:
        grad, hess = obj.grad_hess(preds, lam=lam, kappa=kappa, beta=beta)
        num = np.empty(n)
        h = 1e-6
        for i in range(n):
            up, dn = preds.copy(), preds.copy()
            up[i] += h
            dn[i] -= h
            num[i] = (obj.loss(up, lam=lam, kappa=kappa, beta=beta)
                      - obj.loss(dn, lam=lam, kappa=kappa, beta=beta)) / (2 * h)
        err = np.max(np.abs(grad - num) / (np.abs(num) + 1e-8))
        ok = err < tol or np.max(np.abs(grad - num)) < 1e-7
        print(f"  {label}: max相対誤差={err:.2e} -> {'OK' if ok else 'NG'}")
        assert ok, f"勾配検証失敗 ({label}): {err}"
        assert np.all(hess > 0), "ヘシアンに非正値"

    print("勾配の有限差分検証: 全ケース合格")


if __name__ == "__main__":
    _self_test()
