# Reinforcement learning

---

<details>
<summary>Table of content</summary>

# Table of content


#### 1. [Introduction to Reinforcement Learning](#1)

### I. [Tabular Solution methods](#I)

#### 2. [Multi-armed Bandits](#2)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.1 [A k-armed Bandit Problem](#2.1)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.2 [Action-value Methods](#2.2)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.3 [andit Algorithm Examples](#2.3)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.4 [Tracking a Nonstationary Problem](#2.4)

#### 3. [Finite Markov Decision Processes](#3)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.1 [The Agent–Environment Interface](#3.1)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.2 [Goals and Rewards](#3.2)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.3 [Returns and Episodes](#3.3)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.4 [Unified Notation for Episodic and Continuing Tasks](#3.4)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.5 [Policies and Value Functions](#3.5)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.6 [Optimal Policies and Value Functions](#3.6)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.7 [Optimality and Approximation](#3.7)

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.8 [Summary](#3.8)

#### 5. [Monte Carlo Methods](#5)


</details>

---

## Summary of Notation

$\dot{=}$ is used for **"is defined as"**

$S =$ set of **nonterminal states**
  * $s, s' \rightarrow$ some states

$S^+ =$ set of **all states**, including the terminal state

$A =$ set of **actions**

$R =$ set of **rewards**

$T \rightarrow$ transition function

We'll assume that each of these sets will have a *finite number of elements*.

<br>


**<font color=blue>Value functions</font>:**

* **<font color=blue>State-Value functions</font>:** <u>expected cumulative rewards</u> when **starting in $s$** and following policy $\pi$ thereafter.

  * <font color=blue>$v_\pi(s)$</font> $= E_\pi[G_t|S_t=s] \rightarrow$ value of state $s$ under policy $\pi$ (expected return)
      * <font color=blue>$V, V_t$</font> $\rightarrow$ array **estimates** of <font color=blue>$v_\pi(s)$</font> or <font color=blue>$v_*(s)$</font>
  * <font color=blue>$v_*(s)$</font> $= E_*[G_t|S_t=s] \rightarrow$ value of state s under the optimal policy
      * $\;\;\; = \max_\pi v_\pi(s)$

<br>

* **<font color=blue>Action-Value functions</font>:** <u>expected cumulative rewards</u> when **taking action $a$** in state $s$ and following policy $\pi$ thereafter.
  * <font color=blue>$q_\pi(s,a)$</font> $= E[R_t | S_t=s,  A_t = a]$
      * <font color=blue>$Q, Q_t$</font> $\rightarrow$ array **estimates** of <font color=blue>$q_\pi(s,a)$</font> or <font color=blue>$q_*(s,a)$</font>
  * <font color=blue>$q_*(s, a)$</font> $= E_*[G_t|S_t=s,  A_t = a] \rightarrow$ value of taking action $a$ in state $s$ under the optimal policy
      * $\;\;\; = \max_\pi q_\pi(s)$


<br>


**<font color=blue>Policy $\pi$</font>:** mapping from state to action (decision-making rule)
* **<font color=blue>Optimal Policy $\pi_*$</font>**
    * for $q_*(s,a) \rightarrow$ **<font color=blue>$\pi_*(s)$</font>** $= \argmax_a q_*(s,a)$
    * for $v_*(s) \rightarrow$ **<font color=blue>$\pi_*(s)$</font>** $= \argmax_a E_{s'}[r(s,a,s') + \gamma v_*(s')]$
        * $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; = \argmax_a \sum_{s'} T(s,a,s') (r(s,a,s') + \gamma v_*(s'))$

<br>

* **<font color=blue>Value iteration</font>:**, for estimating $\pi \approx \pi_* \rightarrow \texttt{converges to } v_*(s)$. Only one iteration of iterative policy evaluation is performed between each step of policy improvement.

    * Starting from $V_0^*(s)=0$ for all ($\forall$) $s$ $\rightarrow$ iterate until convergence (usually change being smaller than some threshold we choose):
        * $V_{i+1}^*(s) = \max_a \sum_{s'} T(s,a,s') (r(s,a,s') + \gamma V_i^*(s'))$
        * $\;\;\;\;\;\;\;\;\;\;\;\; = \max_a ImmediateReward + Discount*FutureRewards$

* **<font color=blue>Policy iteration (iterative policy evaluation)</font>:**, for estimating $\pi \approx \pi_*$.



<br>
---

## 1. Introduction to Reinforcement Learning <a id='1'></a>

**<font color=blue>Reinforcement learning</font>** an area of machine learning concerned with how intelligent **agents** ought to take **actions** in an **environment** in order to maximize the notion of cumulative **reward**.

The **environment** is typically stated in the form of a <u>Markov decision process (MDP)</u>, because many reinforcement learning algorithms for this context use dynamic programming techniques.

<br>

**<font color=blue>Exploration & Exploitation trade-off:</font>** **Dilemma:** choosing when to explore & when to exploit?
* **Exploration:** improve knowledge for long-term benefit.
* **Exploitation:** exploit knowledge for short-term benefit.

<br>

**<font color=blue>Four main subelements of a reinforcement learning system:</font>** a **policy**, a **reward signal**, a **value function**, and, optionally, a **model** of the environment.
* **<font color=blue>Policy</font>** defines the learning agent’s <u>way of behaving</u> at a given time.
* **<font color=blue>Reward signal</font>** indicates <u>what is good in an immediate sense</u>.
* **<font color=blue>Value function</font>** specifies <u>what is good in the long run</u>. Roughly speaking, the value of a state is the *total amount of reward* an agent can expect to accumulate over the future, starting from that state.
* **<font color=blue>Model</font>** <u>mimics the behaviour of the environment</u>, or
more generally, that allows inferences to be made about how the environment will behave.
  * **Used for planning**, by which we mean any way of <u>deciding on a course of action by considering possible future situations</u> before they are actually experienced.

<br>

**<font color=blue>Model-based vs Model-free RL:</font>**
* If model $\rightarrow$ **model-based** methods
* If no model $\rightarrow$ simpler **model-free** methods
  * explicitly <u>trial-and-error learners</u>—viewed as almost the opposite of planning.
* $\rightarrow$ **Modern reinforcement learning** spans the spectrum **from** <u>low-level, trial-and-error learning</u> **to** <u>high-level, deliberative planning</u>.

**<font color=blue>Evolutionary methods:</font>** (not focused on this course)
* Instead of estimating value functions, these methods apply multiple static policies each interacting over an extended period of time with a separate instance of the environment. The policies that obtain the most reward, and random variations of them, are carried over to the next generation of policies, and the process repeats.
* Although evolution and learning share many features and naturally work together, we do not consider evolutionary methods by themselves to be especially well suited to reinforcement learning problems and, accordingly, we do not cover them in this book.

<br>

---

# I. Tabular Solution methods <a id='I'></a>

In this part we describe almost all the <u>core ideas of reinforcement learning algorithms</u> in their **simplest forms:** $\rightarrow$ State & action spaces are small enough for the <u>**approximate value functions** to be represented as arrays, or tables</u>.
* <u>Often find exact solutions</u>, that is, they can often find exactly the optimal value function and the optimal policy


This **<font color=red>contrasts with</font>** the <b>*approximate methods*</b> described in the next part, which only <u>find approximate solutions</u>, but which in return can be <u>applied effectively to much larger problems</u>.

<br>

**Chapters in this section:**

* 2. **Multi-armed Bandits:** special case of the reinforcement learning problem in which there is only a <u>single state</u>.
* 3. **Finite Markov Decision Processes:** the <u>general problem formulation</u> that we treat throughout the rest of the notes.
      * Its main ideas including **Bellman equations** and **value functions**.
* The next three chapters (4., 5. & 6.) describe **three <u>fundamental classes of methods</u> for solving finite Markov decision problems:**

  4. **Dynamic Programming**
      * **<font color=green>+</font>** Well developed mathematically
      * **<font color=red>-</font>** Require a complete and accurate model of the environment
  5. **Monte Carlo Methods**
      * **<font color=green>+</font>** Don’t require a model
      * **<font color=green>+</font>** Conceptually simple
      * **<font color=red>-</font>** Not well suited for step-by-step incremental computation
  6. **Temporal-Difference Learning**
      * **<font color=green>+</font>** Don’t require a model
      * **<font color=green>+</font>** Fully incremental
      * **<font color=red>-</font>** More complex to analyze

&nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; The methods also differ in several ways with respect to their efficiency and speed of convergence.

* The remaining two chapters (7. & 8.) **describe how these three classes of methods can be combined** to <u>obtain the best features of each of them</u>.

  7. <b>*n*-step Bootstrapping:</b> Strengths of <u>Monte Carlo methods</u> can be **combined** with the strengths of <u>temporal-difference methods</u> via multi-step bootstrapping methods.
  8. **Planning and Learning with Tabular Methods:** <u>temporal-difference learning</u> methods can be **combined with** <u>model learning and planning methods</u> (such as dynamic programming) for a **<font color=green>complete and unified solution to the tabular reinforcement learning problem</font>**.


<br><br>

---

## 2. Multi-armed Bandits <a id='2'></a>

The most **important feature distinguishing reinforcement learning** from other types of learning is that it uses training information that <u>*evaluates* the actions taken</u> rather than instructs by giving correct actions. $\rightarrow$ need for active exploration, for an explicit search for good behaviour.

**In this chapter** we <u>**study** the evaluative aspect</u> of reinforcement learning <u>**in** a simplified setting</u>, one that does not involve learning to act in more than one situation.

Studying this case enables us to **see** most clearly **how** <u>evaluative feedback</u> **differs from**, and yet **can be combined with**, <u>instructive feedback</u>.

### 2.1 A $k$-armed Bandit Problem <a id='2.1'></a>

**Problem statement:**

1. Being faced repeatedly with a <u>choice among $k$ different options/actions</u>
2. <u>Receive numerical reward</u> chosen from a stationary probability distribution that depends on the action you selected
    * Each of the $k$ actions has an <u>expected (mean) reward</u> $\rightarrow$ **value** of that action:
        * $q_*(a) = E[R_t | A_t = a]$
    * We denote the **<font color=purple>estimated value</font>** as:
        * $Q_t(a)$

* We assume that <u>you don't know the **action values** with certainty</u>, although you may have estimates. <font color=orange>Otherwise it would be trivial to solve the k-armed bandit problem: you would always select the action with highest value.</font>

**Objective:** maximize the expected total reward over some time period.

<br><br>

**Exploiting vs Exploring**
* **Exploiting** your current knowledge:
  * Selecting  action whose <font color=purple>estimated value</font> is greatest (*greedy* action).
  * **Goal:** maximize the expected reward on the one step

* **Exploring**  to improve your estimate of the nongreedy action’s value:
  * Not selecting *greedy* action
  * **Goal:** to produce the greater total reward in the long run

*  There are <u>many sophisticated methods for balancing **exploration** and **exploitation**</u> for particular mathematical formulations of the $k$-armed bandit and related problems.
    * <font color=red>most of these methods make strong assumptions about stationarity and prior knowledge</font> that are **either** <u>violated</u> or <u>impossible to verify</u> in applications.


### 2.2 Action-value Methods <a id='2.2'></a>

**What:**  methods for <u>(1) estimating the **values**</u> of actions and for using the estimates to <u>(2) make **action** selection decisions</u>.

(1) <u>estimating the **values**</u>

* **<font color=orange>ONE natural WAY</font>** for **<font color=purple>estimating the true value of an action</font>** is by <u>averaging the rewards actually received</u> **(*sample-average* method)**:

$$Q_t(a) \dot{=} \frac{\textrm{sum of rewards when } a \textrm{ taken prior to } t}{\textrm{number of times } a \textrm{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1_{predicate, A_t = a}}}{\sum_{i=1}^{t-1} \mathbb{1_{predicate, A_t = a}}} \;\;\;\;\;(2.1)$$

<p align="center">
  Where:
</p>

  * $1_{predicate}$ denotes the random variable that is <u>**1** if predicate is true</u> and <u>**0** if it is not</u>.

  * If the denominator:
    * $= 0$, then we instead define <u>$Q_t(a)$ as some default value</u> (e.g. zero).
    * $\rightarrow \infty$, then by the **law of large numbers, <font color=green>$Q_t(a)$ converges to $q_*(a)$</font>**.

  * **<font color=orange>NOTE:</font>** **$Q_n$** can be **computed in** a <u>computationally efficient manner</u>, in particular, with constant memory and constant per-time-step computation as:

  $$Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n] \;\;\;\;\;(2.3)$$
  $$NewEstimate = OldEstimate + StepSize[Target - OldEstimate] \;\;\;\;\;(2.4)$$

(2) <u>make **action** selection decisions</u>

* The **<font color=orange>simplest action selection rule</font>**, is <u>always selecting one of the actions with the highest estimated value</u> **(*greedy* action selection method)**:

$$A_t = \argmax_{a}Q_t(a) \;\;\;\;\;(2.2)$$

* A **<font color=orange>simplest alternative action selection rule</font>**, is to behave greedily most of the time, but every once in a while, say <u>with small probability $\epsilon$, instead select action randomly</u> **($\epsilon$-greedy methods)**.


### 2.3 Bandit Algorithm Examples <a id='2.3'></a>

**Example:** 10-bandit problems

* Given:

<p align="center">
  <img src="/images/10-armed-eg1.png" alt="drawing" width="500">
</p>

  * **Figure 2.1:** An example bandit problem from the 10-armed testbed. The true value $q_*(a)$ of each of the ten actions was selected according to a normal distribution with mean zero and unit variance, and then the actual rewards were selected according to a mean $q_*(a)$ unit variance normal distribution, as suggested by these gray distributions.

<br>

* Then compares a greedy method with two $\epsilon$-greedy methods ($\epsilon = 0.01$ and $\epsilon = 0.1$):

<p align="center">
  <img src="/images/10-armed-eg2.png" alt="drawing" width="500">
</p>

  * **Figure 2.2:** Average performance of $\epsilon$-greedy action-value methods on the 10-armed testbed. These data are averages over 2000 runs with different bandit problems. All methods used sample averages as their action-value estimates.

<br>

* NOTE:
    * The $\epsilon = 0.01$ method <font color=red><u>improved more slowly</u></font>, but <font color=green><u>eventually would perform better</u></font> than the $\epsilon = 0.1$ method on both performance measures shown in the figure.
    * It is **also possible** to <font color=green><u>reduce $\epsilon$ over time to try to get the best of both high and low $\epsilon$ values</u></font>.
    * With **noisier rewards** it <u>takes more exploration</u> to find the optimal action.


**Exploration** is <font color=green>beneficial</font> even in the deterministic worlds **if:**
  * **<font color=blue>nonstationary</font>** task, that is, the **true values of the actions changed over time** $\rightarrow$ **agent’s decision-making policy changes**.
    *  **<font color=blue>nonstationary</font>** is the case <u>most commonly encountered in reinforcement learning</u>.


<br><br>

**Example:** A simple bandit algorithm

Pseudocode for a complete bandit algorithm using incrementally computed sample averages and $\epsilon$-greedy action selection is shown in the box below.

> **Initialize, for $a = 1 \; to \; k$:**
> $\;\;\;$ $Q(a) \leftarrow 0$
> $\;\;\;$ $N(a) \leftarrow 0$
>
> **Loop forever:**
> $\;\;\; A \leftarrow$ $\begin{cases} \argmax_{a}Q(a) &\texttt{with probability } 1- \epsilon \;\;\; \texttt{(breaking ties randomly)}  \\ \texttt{random action} &\texttt{with probability } \epsilon \end{cases}$
> $\;\;\; R \leftarrow bandit(A)$
> $\;\;\; N(a) \leftarrow N(a)+1$
> $\;\;\; Q(A) \leftarrow Q(A) + \frac{1}{N(A)}[R-Q(A)]$


* Where:
    * **function $bandit(a)$** is assumed to **take** an <u>action</u> and **return** a <u>corresponding reward</u>



<br>
### 2.4 Tracking a Nonstationary Problem <a id='2.4'></a>

**What:** <u>true values of the actions changed over time</u> $\rightarrow$ agent’s decision-making policy changes.

**Adjustments:** it makes sense to give <u>more weight to recent rewards</u> than to long-past rewards.

* A **<font color=orange>POPULAR WAY</font>** it to use a **constant step-size parameter $\alpha$** (2.3 modified to be):

$$Q_{n+1} = Q_n + \alpha [R_n - Q_n] \;\;\;\;\;(2.5)$$

<p align=center>
Where:
</p>

* $\alpha \in (0,1]$ is constant

**$\rightarrow$ resulting in $Q_{n+1}$ being a weighted average of past rewards and the initial estimate $Q_1$** (sometimes called an *exponential recency-weighted average*):

$$Q_{n+1} = (1 - \alpha )^n Q_1 + \sum_{i=1}^n \alpha (1 - \alpha)^{n-1} R_i \;\;\;\;\;(2.6)$$



<br><br>

---

## 3. Finite Markov Decision Processes <a id='3'></a>

**<font color=blue>Markov decision process (MDP)</font>** provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.

A **MDP is a 4-tuple $(S,A,p,R)$**, where:

* $S$ is a set of states called the state space,
* $A$ is a set of actions called the action space (alternatively, $A(s)$ is the set of actions available from state $s$),
* $p(s', r|s, a) = Pr\{S_t = s', R_t =r | S_{t-1} = s, A_{t-1} = a\}$ is the probability that action $a$ in state $s$ at time $t$ will lead to state $s'$ at time $t+1$,
* $R_{a}(s,s')$ is the immediate reward (or expected immediate reward) received after transitioning from state $s$ to state $s'$, due to action $a$

The state and action spaces may be **finite** or **infinite**:
* e.g. the set of real numbers is infinite.
* Some processes with countably infinite state and action spaces can be reduced to ones with finite state and action spaces.

A **policy function $\pi$**  is a (potentially probabilistic) mapping from state space to action space.

**Optimization objective** $\rightarrow$ find a good "policy" for the decision maker.
* **EXTRA:** Once a <u>MDP</u> is **combined with** a <u>policy</u> in this way, this fixes the action for each state and the resulting combination **behaves like a <u>Markov chain</u>** (since the action chosen in state $s$ is completely determined by $\pi(s)$ and $\Pr(s_{t+1}=s'\mid s_{t}=s,a_{t}=a)$ reduces to $\Pr(s_{t+1}=s'\mid s_{t}=s)$, a **Markov transition matrix**).

### 3.1 The Agent–Environment Interface <a id='3.1'></a>

**Agent:** <u>learner and decision maker</u>.

**Environment:** <u>thing agent interacts with</u>, comprising everything outside the agent.

The environment also gives rise to **rewards**, special numerical values that the <u>agent seeks to maximize</u> over time through its choice of **actions**.

<p align="center">
  <img src="/images/MDP-agent-environment-interaction.png" alt="drawing" width="400">
</p>

The MDP and agent together give rise to a *sequence* or *trajectory* that begins like this:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3,... $$

*In general*, **actions** can be <u>any decisions we want to learn how to make</u>, and the **states** can be <u>anything we can know that might be useful in making them</u>.

<br><br>

In a **finite MDP**, the sets of states, actions, and rewards ($S$, $A$, and $R$) all have a finite number of elements.

In this case, the **random variables $R_t$ and $S_t$** have well defined <u>discrete probability distributions</u> **dependent only on** the <u>preceding state and action</u>.

There is a **probability of those values occurring at time t**, given particular values of the preceding state and action:

$$p(s', r|s, a) = Pr\{S_t = s', R_t =r | S_{t-1} = s, A_{t-1} = a\},$$
<p align="center">
  Where:
</p>

* $s' =$ particular values of the random variable $S$ ($s' \in S$)
* $r =$ particular values of the random variable $R$ ($r \in R$)

The **function $p$** defines the <u>dynamics of the MDP</u>. $p$ specifies a <u>probability distribution for each choice of $s$ and $a$</u>.

Total probability is thus:

$$\sum_{s' \in S} \sum_{r \in R} p(s', r|s, a) = 1, for\;all\;s \in S, a \in A(s)$$


**<font color=blue>Markov property:</font>** The state must include information about all aspects of the past agent–environment interaction that make a difference for the future.
* (only present matters)
* (things/rules/transition model are stationary)
* <font color=blue>We will assume the Markov property throughout this book.</font>


**Calculations from the four-argument dynamics function:**

* **state-transition probabilities $p : S \times S \times A \rightarrow [0, 1]$**
  * $p(s'|s, a) = Pr\{S_t = s'| S_{t-1} = s, A_{t-1} = a\} = \sum_{r \in R} p(s', r|s, a)$

* **expected rewards for state–action pairs $r : S \times A \rightarrow \R$**
  * $r(s,a) = E[R_t | S_{t-1}=s, A_{t-1}=a] = \sum_{r \in R} \sum_{s' \in S} p(s', r|s, a)$

* **expected rewards for state–action–next-state triples $r : S \times A \times S \rightarrow \R$**
    * $r(s,a, s') = E[R_t | S_{t-1}=s, A_{t-1}=a, S_t = s'] = \sum_{r \in R} r \frac{p(s', r | s, a)}{p(s' | s, a)}$





### 3.2 Goals and Rewards <a id='3.2'></a>

**Reward Hypothesis:**

* That all of what we mean by goals and purposes can be well thought of asthe maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

It is critical that the rewards we set up truly indicate what we want accomplished.
* **For example**, a chess-playing agent should
be rewarded only for actually winning, not for achieving subgoals such as taking its
opponent’s pieces or gaining control of the center of the board.  If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal.

**Reward signal** is your way of communicating to
the robot <u>what you want it to achieve</u>, not how you want it achieved.

### 3.3 Returns and Episodes <a id='3.3'></a>

In general, we seek to maximize the expected **return**, where the return, denoted $G_t$.

#### Episodic Tasks

$G_t$ is in the simplest case the **return** of the sum of the rewards:

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ··· + R_T \;\;\;\;\;(3.7)$$


Tasks with *episodes* of this kind are called **<font color=blue>episodic tasks</font>**. In episodic tasks we **sometimes need to distinguish** the set of all <u>nonterminal states</u>, denoted $S$, from the set of all states plus the <u>terminal state</u>, denoted $S^+$. The time of termination, $T$, is a random variable that normally varies from episode to episode.

#### Continuing Tasks

On the other hand, in many cases the <u>agent–environment interaction does not break naturally into identifiable episodes</u>, but goes on **continually without limit**. We call these **<font color=blue>continuing tasks</font>**.

In this book we usually use a definition of return that is slightly more complex conceptually but much simpler mathematically. The additional concept that we need is that of **discounting**. According to this approach, the agent tries to select actions so that the <u>sum of the discounted rewards</u> it receives over the future is maximized. In particular, it chooses $A_t$ to maximize the expected discounted return:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ··· = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\;\;\;\;(3.8)$$
<p align="center">
  Where:
</p>

* $\gamma =$ is a parameter, $0 \leq \gamma \leq 1$, called the **discount rate**.
  * If $\gamma < 1$, the infinite sum has a finite value as long as the reward sequence $\{R_k\}$ is bounded.
  *  If $\gamma = 0$, the agent is “myopic” in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose $A_t$ so as to maximize only $R_{t+1}$. But in general, <font color=red>acting to maximize immediate reward can reduce access to future rewards</font> so that the return is reduced.
  * As $\gamma$ approaches $1$, the <font color=green>return objective takes future rewards into account more strongly</font>; the agent becomes more farsighted.

**Returns at successive time steps** are related to each other in a way that is important for the theory and algorithms of reinforcement learning:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + ··· \\
\;\;\;\;\;\;\;= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + ···)  \\
= R_{t+1} + \gamma G_{t+1} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(3.9)$$

* Note that this works for all time steps $t<T$, even if termination occurs at $t + 1$, if we define $G_T = 0$

* Note that although the return is a sum of an infinite number of terms, it is still finite if the reward is nonzero and constant, if $\gamma < 1$.


### 3.4 Unified Notation for Episodic and Continuing Tasks <a id='3.4'></a>

In the preceding section we described two kinds of reinforcement learning tasks:

* **episodic tasks:** agent–environment interaction naturally breaks down into a sequence of separate episodes.

* **continuing tasks:** agent–environment interaction don't breaks down into a sequence of separate episodes.

We have defined the return as a sum over a finite number of terms in one case (3.7) and as a sum over an infinite number of terms in the other (3.8). These two can be unified by considering **episode termination** to be the entering of a <u>special absorbing state that transitions only to itself and that generates only rewards of zero</u>. For example, consider the state transition diagram:

<p align="center">
  <img src="/images/Unified-Notation-Episodic-and-Continuing-Tasks.png" alt="drawing" width="400">
</p>

Starting from $S_0$, we get the reward sequence $+1, +1, +1, 0, 0, 0,...$. Summing these, we get the same return whether we sum over the first $T$ rewards (here $T = 3$) or over the full infinite sequence. This remains true even if we introduce discounting.

Thus, we can define the **return, in general**, according to (3.8), using the convention of omitting episode numbers when they are not needed, and including the possibility that  = 1 if the sum remains defined (e.g., because all episodes terminate). Alternatively, we can write:

$$G_t = \sum_{k=t+1}^T \gamma^{k-t-1} R_k$$

* including the possibility that $T = \infty$ or $\gamma = 1$ (but not both).


### 3.5 Policies and Value Functions <a id='3.5'></a>

Almost all reinforcement learning algorithms involve estimating **<font color=blue>value functions</font>** — functions of states (or of state–action pairs) that **estimate** <u>how good it is for the agent to be in a given state</u> (or <u>how good it is to perform a given action in a given state</u>).

**<font color=blue>Policy $\pi$</font>** is a **mapping from** <u>states</u> **to** <u>probabilities of selecting each possible action</u>. If the agent is following policy $\pi$ at time $t$, then $\pi(a|s)$ is the probability that $A_t = a$ if $S_t = s$.
* Accordingly, value functions are defined with respect to <u>particular ways of acting</u>, called **<font color=blue>policies</font>**.

**Reinforcement learning methods specify** <u>how the agent’s policy is changed as a result of its experience</u>.

The **<font color=blue>value functions $v_\pi(s)$</font>** of a state $s$ under a **policy $\pi$**, is the expected return when starting in $s$ and following $\pi$ thereafter. For MDPs, we can define $v_\pi$ formally by

$$v_\pi(s) = E_\pi[G_t|S_t=s] = E_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s], for\;all\; s \in S \;\;\;\;\;(3.12)$$

<p align="center">
  Where:
</p>

* $E_\pi[]$ denotes the expected value of a random variable given that the agent follows policy $\pi$, and $t$ is any time step.

The **<font color=blue>action-value function $q_\pi(s, a)$</font>** for policy $\pi$ is defined as the **value of taking <u>action $a$</u> in <u>state $s$</u> under a <u>policy $\pi$</u>**, as the <b><u>expected return</u></b> starting from $s$, taking the action $a$, and thereafter following policy $\pi$:

$$q_\pi(s, a) = E_\pi[G_t|S_t=s, A_t=a] = E_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a] \;\;\;\;\;(3.13)$$

A **fundamental property of value functions** used throughout reinforcement learning and dynamic programming is that they <u>satisfy recursive relationships</u> similar to that which we have already established for the return (3.9).

### 3.6 Optimal Policies and Value Functions <a id='3.6'></a>

**<font color=green>Solving a reinforcement learning task</font>** means, roughly, <u>finding a policy that achieves a lot of reward over the long run</u>.

For finite MDPs, we can precisely define an **<font color=blue>optimal policy $\pi_*$</font>** as follows:
* $\pi \geq \pi'$ if and only if $v_\pi(s) \geq v_{\pi'}(s)$ for all $s \in S$
* **Always at least one policy** that is better than or equal to all other policies.

All optimal policies share the same **<font color=blue>optimal state-value function $v_*$</font>** defined as:

$$v_*(s) = \max_\pi v_\pi(s) \;for\;all\; s \in S \;\;\;\;\;\;(3.15)$$

Optimal policies also share the same **<font color=blue>optimal action-value function $q_*$</font>** defined as:

$$q_*(s, a) = \max_\pi q_\pi(s, a) \;for\;all\; s \in S \; \& \; a \in A(s) \;\;\;\;\;\;(3.16)$$


For the state–action pair $(s, a)$, this function gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy. Thus, we can write **$q_*$ in terms of $v_*$** as follows **(Bellman equation)**:

$$q_*(s, a) = E[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a] \;\;\;\;\;\;(3.17)$$

### 3.7 Optimality and Approximation <a id='3.7'></a>

For the kinds of tasks in which we are interested, <font color=red>optimal policies can be generated only with extreme computational cost</font>. In particular, the amount of computation it can perform in a single time step.


### 3.8 Summary <a id='3.8'></a>

**<font color=green>Solving a reinforcement learning task</font>** means, roughly, <u>finding a policy that achieves a lot of rewards over the long run</u>.

<br>

**Markov property:**
* Only present (current state $s$) matters.
  * Thus current state $s$ must hold <u>everything you need to know from the past</u>.
* Rules are stationary.
* <font color=blue>We will assume the Markov property throughout this book.</font>

<br>

**<font color=blue>Policy $\pi$, $\pi(a|s)$</font>:** action to take for any given state
* Any policy: $\pi(s) \rightarrow a$
* Optimal policy: $\pi^*(s) \rightarrow a$
  * maximizes long-term expected reward

<br>

**<font color=blue>State-Value functions $v_\pi(s)$</font>:** <u>expected cumulative rewards</u> when starting in $s$ and following policy $\pi$ thereafter.
* For MDPs, we can define $v_\pi$ formally by $v_\pi(s) = E_\pi[G_t|S_t=s]$

**<font color=blue>State-Value functions $v_\pi(s)$</font>:** can be <u>decomposed into **immediate** and **future** components</u> using **<font color=blue>Bellman equation</font>**. Bellman equation forms the basis of a number of ways to compute, approximate, and learn $v_\pi$.
* $V(s) = E[G_t | s_t = s]$
* $V(s) = E[r_{t+1} + \gamma V(s_{t+1} | s_t = s)]$
  * Where:
      * $\gamma =$ is a parameter, $0 \leq \gamma \leq 1$, called the **discount rate**.
          * When $0$ $\rightarrow$ consider only immediate rewards
          * When approaches $1 \rightarrow$ forward looking

**<font color=blue>Action-Value functions $q_\pi(s, a), Q_\pi$</font>:** <u>expected cumulative reward of taking action $a$</u> when starting in $s$ and following policy $\pi$ thereafter.
* $q_\pi(a) = E[R_t | A_t = a]$

**<font color=blue>Action-Value functions $v_\pi(s)$</font>:** can also be <u>decomposed into **immediate** and **future** components</u> using **<font color=blue>Bellman equation</font>**.
* $Q_\pi(s,a) = E_\pi[r_t + \gamma Q_\pi (S_{t+1}, a_{t+1}) | s_t=s, a_t=a]$
* $Q_\pi(s,a) = \sum_{s'} T(s,a,s') r(s,a,s') + \gamma \sum_{s'}T(s,a,s')Q_\pi(s',\pi(s'))$

<br>

**<font color=blue>Return $G$ & Rewards $R$</font>:** return is the total of rewards
  * $R(s) =$ reward of **entering state $s$**
  * $R(s, a) =$ reward of **entering state $s$** & **taking action $a$**
  * $R(s, a, s') =$ reward of **being in state $s$** & **taking action $a$** & **entering state $s'$**

<br>

**<font color=blue>State $S$</font>:** every state agent could be in
* $S =$ set of **nonterminal states**
  * $s' =$ particular values of the random variable $S$ ($s' \in S$)
* $S^+ =$ set of **terminal states**

<br>

**<font color=blue>Actions $A, A(s)$</font>:** every action agent could take

<br>

**<font color=blue>Function $p$ (a.k.a Model / Transition function)</font>:** defines the <u>dynamics of the environment</u>. $p$ specifies a <u>probability distribution for each choice of $s$ and $a$</u>.
* **state-transition probabilities $p : S \times S \times A \rightarrow [0, 1]$**
  * $p(s'|s, a) = Pr\{S_t = s'| S_{t-1} = s, A_{t-1} = a\} = \sum_{r \in R} p(s', r|s, a)$



<br><br>

---

## 5. Monte Carlo Methods <a id='5'></a>

> To ensure that well-defined returns are available, here we define Monte Carlo methods only for episodic tasks.
> * That is, we assume experience is divided into episodes, and that all episodes eventually terminate no matter what actions are selected.
> * Only on the completion of an episode are value estimates and policies changed.

**What:** ways of solving the reinforcement learning problem.
  * **estimating** value functions $\rightarrow$ **discovering** optimal policies.


**How:** based on **averaging** <u>sample returns</u> **for each** <u>state–action pair</u>.

**Monte Carlo Methods**
  * **<font color=green>+</font>** Don’t require a model
  * **<font color=green>+</font>** Conceptually simple
  * **<font color=red>-</font>** Not well suited for step-by-step incremental computation


**Base:**

* Unlike previously, we **don't assume complete knowledge of the environment**.

* Require only <b>*experience*</b> — <u>**sample** sequences of **states, actions, and rewards**</u> from actual or simulated interaction with an environment.

* There are m**ultiple states**, *each acting like a different bandit problem* (like an associative-search or contextual bandit) and the different bandit problems are interrelated.


**Terms:**

* The term **“Monte Carlo”** is often used more broadly for <u>any estimation method whose operation involves a significant random component</u>.
  * **HERE:** Here we use it specifically for <u>methods based on averaging complete returns</u>.



<br>
### 5.1 Monte Carlo Prediction <a id='5.1'></a>

Monte Carlo methods for learning the state-value function for a given policy.

The **first-visit MC method** (focus of this chapter) estimates $v_\pi(s)$ as the average of the returns following first visits to s, whereas the **every-visit MC method** averages the returns following all visits to s.
