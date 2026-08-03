# ml 44

# **ARIMA and Classical Time Series Forecasting**

Regression assumes i.i.d. observations; LSTM (Ch. 038) handles temporal dependence with a learned nonlinear recurrence but needs lots of data and less interpretability. ARIMA: linear, statistically well-understood, cheap, interpretable baseline for univariate forecasting.

---

# **1. Stationarity**

Mean, variance, autocovariance don't depend on t:

E[y_t] = μ,  Cov(y_t, y_{t+k}) = γ_k (depends only on lag k)

Why it matters: AR/MA models have time-invariant coefficients — only makes sense if the process's behavior is itself time-invariant. Non-stationary series (trends, seasonality) must be transformed (differencing) first.

Test: **ADF test** (Augmented Dickey-Fuller) — tests for unit root / non-stationarity.

---

# **2. AR(p) — Autoregressive**

y_t = c + Σ φ_i y_{t-i} + ε_t

Same structure as linear regression, but features = the series' own lagged values.

## **Stationarity Condition**

Roots of characteristic equation 1 - φ_1 z - ... - φ_p z^p = 0 must lie outside unit circle. AR(1): reduces to |φ| < 1 (else random walk / non-stationary).

## **Yule-Walker Equations**

Multiply model by y_{t-k}, take expectation, ε_t term vanishes (uncorrelated with past) for k>0:

γ_k = Σ φ_i γ_{k-i}

Closed-form linear system — solve for φ from sample autocovariances. No iterative optimization needed (unlike MA).

## **ACF / PACF**

* ACF(k) = correlation at lag k (includes indirect paths through intermediate lags)
* PACF(k) = correlation at lag k after removing effect of lags 1..k-1 (direct effect only)
* True AR(p): PACF cuts off to ~0 after lag p, ACF decays gradually → use PACF plot to identify p

---

# **3. MA(q) — Moving Average**

y_t = μ + ε_t + Σ θ_i ε_{t-i}

Regresses on past **shocks** (noise), not past values. Always stationary for any finite θ (no unit-root condition).

* No closed-form estimator (ε_{t-i} unobserved) → fit via maximum likelihood, iteratively
* ACF cuts off after lag q, PACF decays gradually — mirror image of AR's signature

---

# **4. ARMA(p,q) and ARIMA(p,d,q)**

ARMA combines both: often needs fewer total parameters than pure AR or MA alone.

## **Differencing for Non-Stationarity**

Δy_t = y_t - y_{t-1} = (1-L)y_t removes a linear trend. Apply d times: Δ^d y_t removes polynomial trend of degree d-1.

ARIMA(p,d,q): fit ARMA(p,q) to the d-times-differenced series. d = minimum differences needed to pass ADF test (usually 0, 1, or 2).

## **Choosing (p,d,q)**

Beyond ACF/PACF heuristics, use penalized likelihood (same principle as GMM's K selection):

AIC = -2·log-likelihood + 2(p+q+1)  
BIC = -2·log-likelihood + (p+q+1)·log(n)

`auto_arima` automates the search over (p,d,q) via AIC/BIC.

---

# **5. Forecasting**

Recursive: forecast t+1, plug it in as if known to forecast t+2, etc.

AR(1): ŷ_{T+h} = c·Σφ^i + φ^h·y_T → converges to c/(1-φ) (unconditional mean) as h→∞.

**Key property**: forecasts mean-revert — influence of last known value decays geometrically (φ^h). Contrast with random walk (φ=1): forecast stays flat forever.

---

# **Complexity**

| Step | Time |
| ----- | ----- |
| Yule-Walker AR(p) | O(n + p³) |
| Full MLE (ARMA/ARIMA, Kalman filter) | O(n·(p+q)²) per likelihood eval × optimizer iterations |
| h-step forecast | O(h) |

Much cheaper than LSTM for comparable ambition — but only captures **linear** dependence.

---

# **Key Pitfalls**

* Fitting AR/ARMA to non-stationary series → unstable φ estimates near/at 1, violates model assumptions — always ADF-test first
* Over-differencing → spurious negative autocorrelation, inflated forecast variance — use minimum d needed
* Plain ARIMA doesn't model seasonality → use **SARIMA** for periodic patterns
* Plain ARIMA is univariate → use **ARIMAX/SARIMAX** for external regressors (holidays, price, weather)
* Overfitting via excessive (p,q) → always select via AIC/BIC, not in-sample fit
* Point forecasts alone understate uncertainty — confidence intervals widen with horizon, report them
* Structural breaks (regime changes) silently break the fixed-coefficient assumption — no (p,d,q) tuning fixes this

---

# **Connections**

* Linear regression — AR(p) is literally regression on the series' own lagged values
* Markov chains — AR(1) is a continuous-state, Gaussian-noise first-order Markov process
* Kalman filters / state-space models — ARMA has an equivalent state-space form; this is how statsmodels actually fits it via exact MLE
* LSTM/GRU (Ch. 038) — nonlinear, data-hungry alternative; ARIMA preferred with limited data / need for interpretability / near-linear dependence
* Exponential smoothing (Holt-Winters) — sibling classical forecasting family, models trend/seasonality differently
* AIC/BIC model selection — same penalized-likelihood principle as GMM's K, Ridge/Lasso's λ

---

# **Interview Summary**

ARIMA models a time series' own temporal dependence linearly: **AR(p)** regresses on past values (fit via the closed-form Yule-Walker equations), **MA(q)** regresses on past unobserved shocks (fit via maximum likelihood, since residuals aren't directly observed), and **ARIMA(p,d,q)** first differences the series d times to remove non-stationarity (trends) before fitting an ARMA(p,q) model to what remains. Order selection combines ACF/PACF cutoff heuristics (AR cuts off in PACF, MA cuts off in ACF) with AIC/BIC-based search. Forecasts are computed recursively and mean-revert geometrically toward the series' long-run mean, with confidence intervals that widen with the forecast horizon. ARIMA remains a strong, cheap, interpretable baseline whenever dependence is close to linear and data is limited — LSTM/GRU or Transformer-based models are the natural alternative once dependence is genuinely nonlinear or long-range and enough data is available.
