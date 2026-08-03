# ml 38

# **Sequence Models: RNN, LSTM, GRU**

Feedforward and convolutional networks assume fixed-size, order-independent inputs. Sequence problems (language, time series, speech, sensor streams) need models that carry history forward through a **hidden state**.

---

# **Why Sequence Models?**

Input:

x_1, x_2, ..., x_T

Hidden state carries memory:

h_t = f(h_{t-1}, x_t)

Same weights reused at every time step.

---

# **1. Vanilla RNN**

Update:

h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)

Output:

y_t = W_hy h_t + b_y

---

## **Backpropagation Through Time (BPTT)**

Gradient w.r.t. shared weights sums contributions across all time steps.

Repeated multiplication by W_hh (and tanh derivative) across steps causes:

* **Vanishing gradients** — long-range dependencies never learned
* **Exploding gradients** — unstable, overflowing updates

---

# **2. LSTM (Long Short-Term Memory)**

Adds a separate **cell state** c_t updated additively, plus three gates.

## **Gates**

* Forget gate f_t — how much old memory to keep
* Input gate i_t — how much new info to write
* Output gate o_t — how much of cell state to expose

## **Equations**

f_t = σ(W_f[h_{t-1}, x_t] + b_f)  
i_t = σ(W_i[h_{t-1}, x_t] + b_i)  
o_t = σ(W_o[h_{t-1}, x_t] + b_o)  
c̃_t = tanh(W_c[h_{t-1}, x_t] + b_c)

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  
h_t = o_t ⊙ tanh(c_t)

## **Why it Works**

Cell state update is additive → gradient path ≈ product of forget gates, not forget-gate × weight-matrix every step. Lets the model learn to preserve gradients over long spans.

---

# **3. GRU (Gated Recurrent Unit)**

Simpler: two gates, no separate cell state.

z_t = σ(W_z[h_{t-1}, x_t] + b_z)   — update gate  
r_t = σ(W_r[h_{t-1}, x_t] + b_r)   — reset gate  
h̃_t = tanh(W_h[r_t ⊙ h_{t-1}, x_t] + b_h)

h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

---

# **LSTM vs GRU**

| Aspect | LSTM | GRU |
| ----- | ----- | ----- |
| States | h_t, c_t | h_t only |
| Gates | 3 | 2 |
| Params | more | fewer |
| Typical use | long sequences, large data | faster, fewer params, similar accuracy |

---

# **4. Deep / Bidirectional RNNs**

* **Stacked RNNs** — layer output feeds next layer as input
* **Bidirectional RNNs** — forward + backward pass concatenated; needs full sequence upfront, not usable for autoregressive generation

---

# **Common Pitfalls**

* Forgetting to clip gradients (exploding gradients)
* Forget-gate bias initialized to 0 instead of 1 (kills memory early in training)
* Feeding padded sequences without masking / packing
* Using bidirectional RNN for autoregressive prediction (leaks future info)
* Very long sequences still degrade even with gating — motivates attention/Transformers

---

# **Interview Summary**

RNNs process sequences by carrying a hidden state through time, but vanilla RNNs suffer from vanishing/exploding gradients because the same weight matrix and squashing nonlinearity get multiplied at every step (BPTT). **LSTM** fixes this with a separate cell state updated additively and three gates (forget, input, output) that let the model learn what to keep, write, and expose — the additive path keeps the gradient from being forced through a shrinking multiplication every step. **GRU** simplifies this to two gates (update, reset) and a single hidden state, with comparable accuracy and fewer parameters. Both dominated sequence modeling until attention-based Transformers, which give direct O(1)-hop connections between any two positions instead of O(T) sequential hops.
