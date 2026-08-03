# reinforcement-learning 1

# **Q-Learning and Deep Q-Networks (DQN)**

Bandit notes (UCB, Thompson Sampling) = stateless: one situation forever. Full RL adds **state** — optimal action depends on where you are, and actions change the state. Q-learning is the canonical algorithm; DQN scales it to huge/continuous state spaces with a neural net instead of a table.

---

# **1. MDP Setup**

Tuple (S, A, P, R, γ). Policy π(a|s).

Q-function: expected discounted return from (s,a) following π after:

Q^π(s,a) = E[Σ γ^t R_t | s_0=s, a_0=a]

Optimal Q* → optimal policy is just π*(s) = argmax_a Q*(s,a). Whole control problem reduces to learning one function.

---

# **2. Bellman Optimality Equation**

Q*(s,a) = E_s'[ R(s,a,s') + γ max_a' Q*(s',a') ]

Self-consistency / fixed-point equation. The Bellman operator T is a **contraction** in sup-norm (γ<1) → by Banach fixed-point theorem, repeated application converges to the unique fixed point Q*. This is why Q-learning converges.

---

# **3. Tabular Q-Learning**

## **Update Rule**

Q(s,a) ← Q(s,a) + α[ r + γ max_a' Q(s',a') − Q(s,a) ]

* Bracketed term = TD error δ
* This is a *sampled, stochastic* version of the Bellman contraction iteration

## **Off-Policy**

Update always uses max_a' Q(s',a') (greedy), regardless of what action was actually taken next → data can be collected under any exploration policy (e.g. ε-greedy) while still learning Q* (not the behavior policy's own Q^π).

## **Exploration: ε-greedy**

* With prob 1-ε: greedy argmax_a Q(s,a)
* With prob ε: random action
* Anneal ε from ~1 down to a small floor over training

## **Convergence Conditions**

1. Every (s,a) visited infinitely often
2. Learning rate: Σα_t = ∞, Σα_t² < ∞ (Robbins-Monro)

---

# **4. From Tables to DQN**

Replace table with Q_θ(s,a) neural net, minimize squared TD error via gradient descent. Two new problems vs tabular case:

## **Problem 1: Correlated data → Experience Replay**

Consecutive transitions are correlated, breaks i.i.d. assumption, causes forgetting. Fix: store transitions in a replay buffer, train on random minibatches.

## **Problem 2: Moving target → Target Network**

TD target uses the same θ being updated → chasing your own tail, unstable. Fix: separate target network Q_θ⁻ (periodic hard copy every C steps, or Polyak/soft update), compute target using frozen θ⁻.

## **Overestimation Bias → Double DQN**

max_a' Q_θ(s',a') uses the same noisy network to both select and evaluate the best action → systematically overestimates (Jensen's inequality: E[max] ≥ max[E]).

Fix — Double DQN:

target = r + γ · Q_θ⁻( s', argmax_a' Q_θ(s',a') )

Online net selects the action, target net evaluates it — decouples selection from evaluation, mostly removes the bias since the two networks' errors are largely independent.

---

# **Training Loop**

1. ε-greedy action from Q_θ
2. Execute, observe (s,a,r,s'), push to replay buffer
3. Sample random minibatch
4. Compute TD target using target net (vanilla) or online-selects/target-evaluates (Double DQN)
5. Squared/Huber TD-error loss → backprop → update θ
6. Every C steps: sync θ⁻ ← θ (or continuous Polyak average)
7. Anneal ε

---

# **Key Pitfalls**

* Forgetting to zero bootstrap at terminal states → `(1-done)` factor is essential
* Target update frequency: too fast → moving-target instability again; too slow → stale targets
* No Double DQN → overestimation bias, especially early training
* Reward scale: unnormalized/sparse rewards destabilize gradients → clip/normalize rewards
* Replay buffer size: too small → correlation/forgetting; too large → stale outdated-policy data
* ε annealed too fast → locks into suboptimal policy before exploring enough
* No general convergence guarantee for DQN (unlike tabular case) — function approximation breaks the contraction argument
* Continuous action spaces: max_a Q(s,a) requires enumerating actions → infeasible; need actor-critic methods (DDPG/TD3/SAC) instead

---

# **Connections**

* Bandits — Q-learning is the stateful generalization
* SARSA / TD(λ) — same TD-learning family, on-policy vs off-policy, n-step interpolation
* Actor-critic / policy gradients — explicit policy + value function, needed for continuous actions
* GANs — online/target network dynamic mirrors generator/discriminator instability class

---

# **Interview Summary**

Q-learning learns the optimal action-value function Q* by exploiting the **Bellman optimality equation**, a fixed-point self-consistency condition; because the underlying Bellman operator is a contraction in sup-norm, repeated (sampled) application provably converges to Q*, and because the update always bootstraps off the greedy max action regardless of the actual exploration policy, Q-learning is **off-policy** — you can explore however you like while still learning the optimal value function. **DQN** scales this to large state spaces by replacing the table with a neural network, which introduces two new instabilities absent from the tabular case — correlated training data and a constantly moving regression target — fixed respectively by **experience replay** and a **target network**. A further refinement, **Double DQN**, fixes a systematic overestimation bias caused by using the same noisy network to both select and evaluate the best next action, by decoupling those two roles across the online and target networks.
