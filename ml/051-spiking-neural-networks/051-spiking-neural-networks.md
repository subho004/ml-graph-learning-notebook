# ml 51

# **Spiking Neural Networks (SNNs)**

Every other network here uses continuous activations, synchronous forward passes. SNNs model neurons communicating via discrete, TIMED spikes — closer to biology, and enables specialized neuromorphic hardware (Intel Loihi, IBM TrueNorth) at far lower energy cost than GPUs, at the cost of a fundamentally harder training story.

---

# **1. Leaky Integrate-and-Fire (LIF) Neuron**

Membrane potential = leaky capacitor driven by input current:

τ_m dV/dt = -(V(t) - V_rest) + R·I(t)

* -(V-V_rest): leak — decays exponentially to rest with no input
* R·I(t): driving term — input pushes potential up

## **Spiking Rule**

if V(t) ≥ V_th: emit spike, V(t) ← V_reset

Discrete threshold-crossing-then-reset → output is a SPARSE SPIKE TRAIN (timestamps), not a continuous value. Info encoded in WHEN/HOW MANY spikes occur.

## **Discretized (simulation) form**

V[t+1] = V[t] + (Δt/τ_m)·(-(V[t]-V_rest) + R·I[t])
s[t] = 1[V[t] ≥ V_th]; if s[t]=1: V[t]←V_reset

---

# **2. Encoding Real Values as Spike Trains**

* **Rate coding**: value x → Poisson spike train, firing rate ∝ x. Robust, but needs long window for precision (latency/precision trade-off)
* **Temporal/latency coding**: value x → timing of a SINGLE spike (larger x fires earlier). Much more info per spike, but sensitive to timing jitter

Real design decision with real trade-offs — unlike ANNs which consume real values directly.

---

# **3. Why SNNs Are Hard to Train**

## **Non-Differentiability**

s[t] = 1[V≥V_th] is a hard step function — derivative is exactly ZERO everywhere except at V_th (undefined delta spike). Vanilla backprop gives essentially zero gradient everywhere — fundamentally broken, not just difficult.

## **Surrogate Gradients (the fix)**

Forward: keep the TRUE hard threshold (preserves real spiking/sparsity/energy benefits)
Backward: substitute a SMOOTH function's derivative (e.g. sigmoid derivative centered at V_th)

forward: s = 1[V≥V_th]
backward: ∂s/∂V ≈ σ'(β(V-V_th))

Principled, not a random hack: near threshold, small ΔV genuinely affects nearby firing — surrogate encodes "how close to firing" as real gradient signal the true (zero) derivative can't express. Same family as straight-through estimators for quantized networks.

## **BPTT for SNNs**

LIF dynamics are a discrete-time recurrence → unroll like an RNN (Ch.038), apply BPTT with surrogate gradients at spikes. Same vanishing/exploding gradient challenges as vanilla RNN, PLUS surrogate approximation error at every spike.

---

# **Energy Efficiency — Why Hardware Matters**

On GPU/CPU: simulating SNN's discrete-time recurrence is typically MORE expensive than an equivalent ANN forward pass — no efficiency win here.

On neuromorphic hardware (event-driven): energy/compute spent ONLY when a spike occurs. Since spike trains are typically very sparse, cost scales with NUMBER OF SPIKES, not network size × timesteps. This is the actual source of SNNs' efficiency claim — conditional on the right hardware.

| Platform | Cost driver |
| ----- | ----- |
| GPU/CPU (dense sim) | O(T·network size) — every neuron every timestep, usually slower than ANN |
| Neuromorphic (event-driven) | O(#spikes) — genuinely sub-linear when activity is sparse |

---

# **Key Pitfalls**

* Non-differentiability is THE central obstacle — without surrogate gradients or ANN-to-SNN conversion, gradient training doesn't work at all
* Running on GPU defeats the purpose — captures none of the efficiency benefit, often slower than the ANN it replaces
* Encoding/decoding overhead — added latency (esp. rate coding) and info loss, absent in ordinary ANNs
* Heavy hyperparameter sensitivity (surrogate steepness β, τ_m, V_th) — less mature tuning guidance than ANNs
* Accuracy gap vs ANNs persists on standard benchmarks — case for SNNs is energy efficiency + biological plausibility, not (yet) accuracy

---

# **Connections**

* RNN/LSTM (Ch.38) — membrane recurrence = RNN unrolled over time, trained via BPTT; same gradient-flow analysis, plus surrogate-gradient complication
* Straight-through estimators/quantized NNs — surrogate gradients = same "different function's derivative than what ran forward" family of tricks
* Neuromorphic computing — distinct hardware/systems research area vs the GPU/CPU/TPU assumption everywhere else in these notes
* Biological neuroscience — LIF model comes directly from computational neuroscience; most literally biologically-plausible architecture in these notes

---

# **Interview Summary**

Spiking Neural Networks model neurons via a Leaky Integrate-and-Fire dynamic: membrane potential integrates input current, leaks continuously toward rest, and emits a discrete spike (resetting the potential) once it crosses a threshold — information is carried by spike timing/rate, not a continuous activation. The central obstacle is that the spike function's derivative is zero almost everywhere, breaking standard backpropagation outright; the practical fix, **surrogate gradients**, keeps the true hard threshold on the forward pass but substitutes a smooth function's derivative (e.g., a sigmoid centered at the threshold) on the backward pass, giving a real, usable gradient signal near the firing boundary. Training then proceeds via backpropagation-through-time, exactly as with an RNN, since the LIF dynamics are a discrete-time recurrence. SNNs' practical appeal is not speed or accuracy on conventional GPUs (where they're typically slower than an equivalent ANN) but energy efficiency on **neuromorphic hardware**, which spends compute only when a spike actually fires — exploiting the natural sparsity of spike trains for a genuinely different, event-driven cost model.
