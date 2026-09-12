"""Transparent tracers over per-turn (KC set, correctness) labels, optionally with a second observation
channel: an LLM's per-skill judgment y in [0,1] made from the dialogue text. Pure numpy.

Sequence format: list over labeled turns i of (kc_ids, label).
Judgment format (optional): {i: {kc_id: y}} where y is the LLM's P(skill) after seeing turn i, available for
the skills practiced at turn i and the skills of turn i+1 (computed from context through the teacher's next
question, so never from the label being predicted).

Both tracers expose:
    fit(sequences, judgments=None)
    trajectories(sequence, prior=None, judgment=None) -> (states L x K, states_cf L x K)
    where states[i][k] = P(correct on skill k) after everything observed up to and including step i.
"""
import numpy as np
from scipy.stats import norm

from scipy.stats import beta as beta_dist


def _events(sequence, judgment):
    """Per-skill event lists in time order. Event = (a or None, y or None, transition: bool)."""
    per_skill = {}
    L = len(sequence)
    for i, (kcs, a) in enumerate(sequence):
        j = judgment.get(i, {}) if judgment else {}
        for k in kcs:
            per_skill.setdefault(k, []).append((int(a), j.get(k), True))
        if i + 1 < L:
            for k in sequence[i + 1][0]:
                if k not in kcs and k in j:
                    per_skill.setdefault(k, []).append((None, j[k], False))
    return per_skill


# ---------------------------------------------------------------- constrained two-channel BKT
class ConstrainedBKT:
    """Per-skill two-state HMM (no forgetting), MAP-EM, guess/slip bounded, Beta shrinkage toward global
    parameters. Optional judgment channel: categorical emissions theta[state][bin], fitted globally.
    """

    def __init__(self, num_kcs, max_guess=0.3, max_slip=0.3, shrink=10.0, iters=50, prior_weight=0.0, judgment_weight=1.0,
                 n_bins=4, judgment_model="bins"):
        self.K, self.max_guess, self.max_slip, self.shrink, self.iters = num_kcs, max_guess, max_slip, shrink, iters
        self.prior_weight = prior_weight
        self.judgment_weight = judgment_weight   # tempering exponent on the judgment likelihood
        self.n_bins, self.judgment_model = n_bins, judgment_model   # "bins": categorical over n_bins; "beta": Beta density on y
        self.edges = np.linspace(0, 1, n_bins + 1)[1:-1]
        self.theta = None                        # bins: 2 x n_bins P(bin | state); beta: 2 x 2 (alpha, beta) per state

    def _bin(self, y):
        return int(np.searchsorted(self.edges, y, side="right"))

    def _jlik(self, y):
        """[P(y | unlearned), P(y | learned)] for a judgment y."""
        if self.judgment_model == "beta":
            yc = float(np.clip(y, 0.01, 0.99))
            return np.array([beta_dist.pdf(yc, *self.theta[0]), beta_dist.pdf(yc, *self.theta[1])])
        return self.theta[:, self._bin(y)]

    # --- emissions for one event
    def _emis(self, a, y, g, s):
        e = np.ones(2)
        if a is not None:
            e *= np.array([g, 1 - s]) if a == 1 else np.array([1 - g, s])
        if y is not None and self.theta is not None:
            e *= self._jlik(y) ** self.judgment_weight
        return e

    def _expected_counts(self, seqs, p0, T, g, s):
        c = dict(n_init=0.0, n_seq=0, n_learn=0.0, n_unl=0.0, g_num=0.0, g_den=0.0, s_num=0.0, s_den=0.0, ll=0.0,
                 th=np.zeros((2, self.n_bins)), ym=np.zeros(2), yv=np.zeros(2), yn=np.zeros(2))
        A = np.array([[1 - T, T], [0.0, 1.0]]); I = np.eye(2)
        for seq in seqs:
            L = len(seq)
            E = np.array([self._emis(a, y, g, s) for a, y, _ in seq])
            As = [A if tr else I for _, _, tr in seq]
            alpha = np.zeros((L, 2)); cn = np.zeros(L)
            alpha[0] = np.array([1 - p0, p0]) * E[0]; cn[0] = alpha[0].sum(); alpha[0] /= cn[0]
            for t in range(1, L):
                alpha[t] = (alpha[t - 1] @ As[t - 1]) * E[t]; cn[t] = alpha[t].sum(); alpha[t] /= cn[t]
            beta = np.ones((L, 2))
            for t in range(L - 2, -1, -1):
                beta[t] = (As[t] * (E[t + 1] * beta[t + 1])).sum(1) / cn[t + 1]
            gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
            c["ll"] += np.log(cn).sum(); c["n_init"] += gamma[0, 1]; c["n_seq"] += 1
            for t in range(L - 1):
                if seq[t][2]:
                    xi = (alpha[t][:, None] * A * (E[t + 1] * beta[t + 1])[None, :]) / cn[t + 1]
                    c["n_learn"] += xi[0, 1]; c["n_unl"] += xi[0].sum()
            for t, (a, y, _) in enumerate(seq):
                if a is not None:
                    if a == 1: c["g_num"] += gamma[t, 0]
                    else: c["s_num"] += gamma[t, 1]
                    c["g_den"] += gamma[t, 0]; c["s_den"] += gamma[t, 1]
                if y is not None:
                    c["th"][:, self._bin(y)] += gamma[t]
                    yc = float(np.clip(y, 0.01, 0.99))
                    c["ym"] += gamma[t] * yc; c["yv"] += gamma[t] * yc * yc; c["yn"] += gamma[t]
        return c

    def _em(self, seqs, prior_means, shrink, init, fit_theta=False):
        p0, T, g, s = init
        for _ in range(self.iters):
            c = self._expected_counts(seqs, p0, T, g, s)
            def mstep(num, den, mean):
                return (num + mean * shrink) / (den + shrink) if den + shrink > 0 else mean
            p0 = mstep(c["n_init"], c["n_seq"], prior_means[0])
            T = mstep(c["n_learn"], c["n_unl"], prior_means[1])
            g = min(mstep(c["g_num"], c["g_den"], prior_means[2]), self.max_guess)
            s = min(mstep(c["s_num"], c["s_den"], prior_means[3]), self.max_slip)
            p0, T, g, s = (float(np.clip(v, 1e-4, 1 - 1e-4)) for v in (p0, T, g, s))
            if fit_theta and c["th"].sum() > 0:
                if self.judgment_model == "beta":       # method of moments per state
                    params = []
                    for st in range(2):
                        n = max(c["yn"][st], 1e-6); m = c["ym"][st] / n; v = max(c["yv"][st] / n - m * m, 1e-4)
                        m = float(np.clip(m, 0.02, 0.98)); common = max(m * (1 - m) / v - 1, 0.2)
                        params.append((max(m * common, 0.3), max((1 - m) * common, 0.3)))
                    self.theta = np.array(params)
                else:
                    th = c["th"] + 1.0                   # Dirichlet(2) smoothing
                    self.theta = th / th.sum(1, keepdims=True)
        return p0, T, g, s

    def fit(self, sequences, judgments=None):
        judgments = judgments or [None] * len(sequences)
        per_skill = {k: [] for k in range(self.K)}
        for seq, jd in zip(sequences, judgments):
            for k, ev in _events(seq, jd).items():
                per_skill[k].append(ev)
        pooled = [ev for evs in per_skill.values() for ev in evs]
        use_j = any(jd for jd in judgments)
        if use_j:
            self.theta = np.array([(1.0, 1.0), (1.0, 1.0)]) if self.judgment_model == "beta" else np.full((2, self.n_bins), 1.0 / self.n_bins)
        flat_means = (0.5, 0.2, min(0.2, self.max_guess), min(0.1, self.max_slip))
        self.global_params = self._em(pooled, flat_means, shrink=0.0, init=flat_means, fit_theta=use_j)
        self.params = np.tile(self.global_params, (self.K, 1)).astype(float)
        self.seen = np.zeros(self.K, dtype=bool)
        for k, evs in per_skill.items():
            if evs:
                self.params[k] = self._em(evs, self.global_params, self.shrink, self.global_params); self.seen[k] = True
        return self

    # --- forward pass
    def _init_belief(self, prior=None):
        b = self.params[:, 0].copy()
        if prior and self.prior_weight > 0:
            g, s = self.params[:, 2], self.params[:, 3]
            for k, p in prior.items():
                p = float(np.clip(p, 0.02, 0.98)); span = 1 - s[k] - g[k]
                Lk = (p - g[k]) / span if span > 0.05 else p
                b[k] = (1 - self.prior_weight) * b[k] + self.prior_weight * float(np.clip(Lk, 0.02, 0.98))
        return b

    def _pcorrect(self, belief):
        g, s = self.params[:, 2], self.params[:, 3]
        return belief * (1 - s) + (1 - belief) * g

    def _update(self, belief, k, a, y=None, transition=True):
        p0, T, g, s = self.params[k]; Lk = belief[k]
        e = self._emis(a, y, g, s)
        post = Lk * e[1] / (Lk * e[1] + (1 - Lk) * e[0])
        belief[k] = post + (1 - post) * T if transition else post

    def trajectories(self, sequence, prior=None, judgment=None):
        L = len(sequence); states = np.zeros((L, self.K)); states_cf = np.zeros((L, self.K))
        b = self._init_belief(prior)
        for i, (kcs, a) in enumerate(sequence):
            j = judgment.get(i, {}) if judgment else {}
            b_cf = b.copy()
            for k in kcs:
                y = j.get(k) if self.theta is not None else None
                self._update(b, k, int(a), y); self._update(b_cf, k, 1 - int(a), y)
            if i + 1 < L and self.theta is not None:
                for k in sequence[i + 1][0]:
                    if k not in kcs and k in j:
                        self._update(b, k, None, j[k], transition=False); self._update(b_cf, k, None, j[k], transition=False)
            states[i] = self._pcorrect(b); states_cf[i] = self._pcorrect(b_cf)
        return states, states_cf


# ---------------------------------------------------------------- Gaussian (TrueLearn-style) two-channel tracer
class GaussianTracer:
    """theta_k ~ N(mu_k, var_k); a turn over KC set C has performance s = mean_k theta_k and
    P(correct) = Phi((s - d) / beta). Judgment channel: a soft single-skill probit observation with
    outcome +1 weighted y and -1 weighted 1-y (moment-matched mixture), noise beta_j.
    """

    def __init__(self, num_kcs, beta=1.0, drift=0.0, difficulty=0.0, prior_var=1.0, prior_weight=0.0, beta_j=1.0, judgment_weight=1.0):
        self.K, self.beta, self.drift, self.d, self.prior_var = num_kcs, beta, drift, difficulty, prior_var
        self.prior_weight, self.beta_j, self.judgment_weight = prior_weight, beta_j, judgment_weight

    def fit(self, sequences, judgments=None):
        return self

    def _init(self, prior=None):
        mu, var = np.zeros(self.K), np.full(self.K, self.prior_var)
        if prior and self.prior_weight > 0:
            for k, p in prior.items():
                p = float(np.clip(p, 0.02, 0.98))
                mu[k] = self.prior_weight * (self.d + norm.ppf(p) * np.sqrt(var[k] + self.beta ** 2))
        return mu, var

    def _pcorrect(self, mu, var):
        return norm.cdf((mu - self.d) / np.sqrt(var + self.beta ** 2))

    @staticmethod
    def _probit_moments(m, v, d, beta, sign):
        c = np.sqrt(v + beta ** 2); t = sign * (m - d) / c
        V = norm.pdf(t) / max(norm.cdf(t), 1e-12); W = V * (V + t)
        return m + sign * (v / c) * V, v * (1 - (v / c ** 2) * W)

    def _update(self, mu, var, kcs, a):
        idx = np.array(kcs); n = len(idx)
        m = mu[idx].mean(); v = var[idx].sum() / n ** 2
        m_new, v_new = self._probit_moments(m, v, self.d, self.beta, 1.0 if a == 1 else -1.0)
        cov = var[idx] / n
        mu[idx] = mu[idx] + cov / v * (m_new - m)
        var[idx] = var[idx] - cov ** 2 / v ** 2 * (v - v_new)
        var[idx] = np.clip(var[idx] + self.drift ** 2, 1e-6, None)

    def _update_soft(self, mu, var, k, y):
        y = float(np.clip(y, 0.0, 1.0)); w = self.judgment_weight
        m, v = mu[k], var[k]
        m1, v1 = self._probit_moments(m, v, self.d, self.beta_j, 1.0)
        m0, v0 = self._probit_moments(m, v, self.d, self.beta_j, -1.0)
        mm = y * m1 + (1 - y) * m0
        vv = y * (v1 + (m1 - mm) ** 2) + (1 - y) * (v0 + (m0 - mm) ** 2)
        mu[k] = (1 - w) * m + w * mm
        var[k] = max((1 - w) * v + w * vv, 1e-6)

    def trajectories(self, sequence, prior=None, judgment=None):
        L = len(sequence); states = np.zeros((L, self.K)); states_cf = np.zeros((L, self.K))
        mu, var = self._init(prior)
        for i, (kcs, a) in enumerate(sequence):
            j = judgment.get(i, {}) if judgment else {}
            mu_cf, var_cf = mu.copy(), var.copy()
            self._update(mu, var, kcs, int(a)); self._update(mu_cf, var_cf, kcs, 1 - int(a))
            if j:
                targets = list(kcs) + ([k for k in sequence[i + 1][0] if k not in kcs] if i + 1 < L else [])
                for k in targets:
                    if k in j:
                        self._update_soft(mu, var, k, j[k]); self._update_soft(mu_cf, var_cf, k, j[k])
            states[i] = self._pcorrect(mu, var); states_cf[i] = self._pcorrect(mu_cf, var_cf)
        return states, states_cf


def predict_sequence(model, sequence, prior=None, judgment=None):
    """Per-turn next-turn predictions (t >= 1) with mean-ar aggregation, as in dialogue-kt."""
    states, _ = model.trajectories(sequence, prior, judgment)
    out = []
    for t in range(1, len(sequence)):
        kcs = sequence[t][0]
        if kcs:
            out.append((int(sequence[t][1]), float(states[t - 1][kcs].mean())))
    return out
