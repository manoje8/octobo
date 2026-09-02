# Probability & Statistics for Machine Learning

> [!NOTE]
> This guide builds each concept on the previous one. By the end, you'll see how **probability → distributions → Bayes' theorem → expectation/variance → MLE** form the statistical backbone of machine learning — from Naive Bayes to neural network training.

---

## 1. Probability — "How Likely Is This Event?"

### The Core Intuition

Probability quantifies **uncertainty**. Every ML model lives in a world of uncertainty: uncertain inputs, uncertain predictions, uncertain parameters. Probability gives us a language to reason about all of it.

> **Probability = (favorable outcomes) / (total possible outcomes)**

But this "counting" definition only works for simple cases. The deeper intuition is:

> **Probability is a number between 0 and 1 that represents your degree of belief that something will happen.**

### The Three Rules You Actually Need

**Rule 1: Probabilities sum to 1**

If you list *all* possible outcomes, their probabilities must add up to 1. A die has six faces — each with probability 1/6, and 6 × (1/6) = 1.

$$\sum_{i} P(x_i) = 1$$

**Rule 2: OR → Add (for mutually exclusive events)**

The probability of rolling a 1 OR a 2:

$$P(1 \text{ or } 2) = P(1) + P(2) = \frac{1}{6} + \frac{1}{6} = \frac{1}{3}$$

For events that *can* overlap:

$$P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$$

**Rule 3: AND → Multiply (for independent events)**

The probability of flipping heads AND rolling a 6:

$$P(\text{heads and } 6) = P(\text{heads}) \times P(6) = \frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$$

### Conditional Probability — The Gateway to ML

> **$P(A | B)$ = "The probability of A, given that B has already happened"**

$$P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$$

**Real example:** What's the probability that an email is spam, *given* that it contains the word "lottery"?

This is the question every spam filter answers — and it's a conditional probability.

### Joint vs. Marginal vs. Conditional

| Concept | Notation | Meaning | Example |
|---------|----------|---------|---------|
| **Joint** | $P(A, B)$ | Probability of A *and* B together | P(spam, contains "lottery") |
| **Marginal** | $P(A)$ | Probability of A regardless of B | P(spam) across all emails |
| **Conditional** | $P(A \mid B)$ | Probability of A *given* B | P(spam \| contains "lottery") |

**The chain rule connects them:**

$$P(A, B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

### Why This Matters for ML

- **Classification:** What's $P(\text{cat} | \text{image})$? Every classifier estimates conditional probabilities.
- **Generative models:** What's $P(\text{image} | \text{cat})$? GANs and VAEs learn to generate data from these distributions.
- **Training:** We maximize $P(\text{data} | \text{model parameters})$ — this is literally MLE (Section 5).

---

## 2. Probability Distributions — "What Shape Does Uncertainty Take?"

### The Core Intuition

A probability distribution tells you **how probability is spread across possible outcomes**. It's a complete description of randomness.

> Think of it like a histogram with infinite data. The shape of that histogram *is* the distribution.

### Discrete vs. Continuous

| | Discrete | Continuous |
|---|---|---|
| **Outcomes** | Countable (heads/tails, 1-6) | Uncountable (height, temperature) |
| **Described by** | Probability Mass Function (PMF) | Probability Density Function (PDF) |
| **P(exact value)** | Can be > 0 | Always = 0 |
| **P(range)** | Sum the PMF values | Integrate the PDF |

---

### 2.1 Bernoulli Distribution — "Yes or No?"

The simplest distribution. A single trial with two outcomes.

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

**ML connection:** A single neuron's output in binary classification (spam or not spam).

**Parameters:** $p$ (probability of success)

---

### 2.2 Binomial Distribution — "How Many Successes in n Trials?"

If you repeat a Bernoulli experiment $n$ times:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Intuition:** You show an ad to 1000 people. Each person clicks with probability $p = 0.02$. How many clicks do you expect? What's the probability of getting more than 30 clicks?

**Parameters:** $n$ (number of trials), $p$ (probability per trial)

---

### 2.3 Gaussian (Normal) Distribution — "The Bell Curve"

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Why is it everywhere?**

The **Central Limit Theorem** says: if you add up many independent random variables, the sum tends toward a Gaussian — *regardless of the original distributions*. Since many real-world quantities are the result of many small, independent effects added together (height = genetics + nutrition + environment + ...), they end up approximately Gaussian.

**Parameters:** $\mu$ (mean — where the peak is), $\sigma^2$ (variance — how spread out it is)

**The 68-95-99.7 rule:**
- 68% of data falls within $\mu \pm 1\sigma$
- 95% within $\mu \pm 2\sigma$
- 99.7% within $\mu \pm 3\sigma$

**ML connections:**
- Weight initialization in neural networks: usually $\mathcal{N}(0, \sigma^2)$
- Gaussian Naive Bayes: assumes features are normally distributed within each class
- The noise in linear regression is assumed Gaussian → minimizing squared error = MLE under Gaussian noise (we'll prove this in Section 5)

---

### 2.4 Uniform Distribution — "Everything Is Equally Likely"

$$f(x) = \frac{1}{b - a} \quad \text{for } a \le x \le b$$

**ML connection:** Random weight initialization (sometimes), random search for hyperparameters, dropout masks.

---

### 2.5 Poisson Distribution — "How Many Events in a Fixed Interval?"

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Intuition:** How many customers arrive per hour? How many server errors per day? How many typos per page?

**Parameter:** $\lambda$ (average rate of events)

---

### 2.6 Exponential Distribution — "How Long Until the Next Event?"

$$f(x) = \lambda e^{-\lambda x} \quad \text{for } x \ge 0$$

**Intuition:** If events arrive at rate $\lambda$ (Poisson), the time *between* events follows an exponential distribution. How long until the next customer? How long until the next server crash?

**The "memoryless" property:** The probability of waiting another 5 minutes doesn't depend on how long you've already waited. This is the *only* continuous distribution with this property.

---

### 2.7 Beta Distribution — "A Distribution Over Probabilities"

$$f(p; \alpha, \beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha, \beta)}$$

**Intuition:** You don't know the true click-through rate of an ad. After observing $\alpha - 1$ clicks and $\beta - 1$ non-clicks, the Beta distribution represents your *uncertainty about the true probability*.

**ML connection:** Bayesian priors, A/B testing, Thompson sampling in reinforcement learning.

---

### Distribution Cheat Sheet

| Distribution | Type | Use When | Key Parameter(s) | Mean | Variance |
|---|---|---|---|---|---|
| Bernoulli | Discrete | Single yes/no trial | $p$ | $p$ | $p(1-p)$ |
| Binomial | Discrete | Count of successes in $n$ trials | $n, p$ | $np$ | $np(1-p)$ |
| Poisson | Discrete | Count of events in fixed time/space | $\lambda$ | $\lambda$ | $\lambda$ |
| Uniform | Continuous | All outcomes equally likely | $a, b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Gaussian | Continuous | Sum of many small effects | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| Exponential | Continuous | Time between events | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ |
| Beta | Continuous | Uncertainty about a probability | $\alpha, \beta$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

---

## 3. Bayes' Theorem — "How to Update Beliefs With Evidence"

### The Core Intuition

Bayes' theorem answers the most practical question in all of ML:

> **I have a belief about the world. I just observed new data. How should I update my belief?**

$$\boxed{P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}}$$

| Term | Name | Meaning |
|---|---|---|
| $P(H \mid D)$ | **Posterior** | Your updated belief after seeing data |
| $P(D \mid H)$ | **Likelihood** | How probable is the data if the hypothesis is true? |
| $P(H)$ | **Prior** | Your belief before seeing data |
| $P(D)$ | **Evidence** | How probable is the data under *all* hypotheses? |

### The Medical Test Example (The Classic)

- A disease affects **1 in 1000** people → $P(\text{disease}) = 0.001$
- A test is **99% accurate**: if you have the disease, it says positive 99% of the time → $P(+|\text{disease}) = 0.99$
- The test has a **5% false positive rate** → $P(+|\text{healthy}) = 0.05$

**You test positive. What's the probability you actually have the disease?**

Your gut says ~95%. Let's see what Bayes says:

$$P(\text{disease}|+) = \frac{P(+|\text{disease}) \cdot P(\text{disease})}{P(+)}$$

First, compute $P(+)$ (the total probability of testing positive):

$$P(+) = P(+|\text{disease}) \cdot P(\text{disease}) + P(+|\text{healthy}) \cdot P(\text{healthy})$$
$$= 0.99 \times 0.001 + 0.05 \times 0.999 = 0.00099 + 0.04995 = 0.05094$$

Now apply Bayes:

$$P(\text{disease}|+) = \frac{0.99 \times 0.001}{0.05094} \approx 0.019 = 1.9\%$$

**Only 1.9%!** Your gut was wildly wrong. The reason: the disease is so rare (the prior is so small) that most positive tests come from the 5% false positive rate among the vast majority of healthy people.

> [!IMPORTANT]
> **The lesson:** Prior probabilities matter enormously. This is why ML models that ignore class imbalance perform poorly — they're ignoring the prior.

### Thinking About It Visually (The Counting Way)

Imagine 100,000 people:
- **100** have the disease (1 in 1000)
  - 99 test positive (99% sensitivity)
- **99,900** are healthy
  - 4,995 test positive (5% false positive)

Total positive tests: 99 + 4,995 = **5,094**

Of those, only 99 actually have the disease: 99 / 5,094 ≈ **1.9%**

### Bayes in ML: Three Major Applications

**1. Naive Bayes Classifier**

To classify an email as spam or not:

$$P(\text{spam} | \text{words}) \propto P(\text{words} | \text{spam}) \cdot P(\text{spam})$$

The "naive" assumption: words are conditionally independent given the class. This is almost always wrong — but the classifier works surprisingly well anyway!

**2. Bayesian Inference for Model Parameters**

Instead of finding a single "best" parameter (MLE), treat the parameter itself as a random variable:

$$P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})}$$

This gives you a *distribution* over parameters, not just a point estimate. You know not just what the best guess is, but *how confident you should be*.

**3. MAP Estimation (Maximum A Posteriori)**

A middle ground between MLE and full Bayesian inference:

$$\theta_{\text{MAP}} = \arg\max_\theta \, P(\theta | \text{data}) = \arg\max_\theta \, P(\text{data}|\theta) \cdot P(\theta)$$

This is just MLE with a prior. When the prior is Gaussian → this is **L2 regularization (Ridge)**. When the prior is Laplace → this is **L1 regularization (Lasso)**.

> [!TIP]
> **Regularization is just Bayesian inference with a prior on the weights.** L2 regularization says "I believe weights should be small (Gaussian prior centered at 0)." L1 says the same but with a Laplace prior, which encourages weights to be *exactly* zero (sparsity).

---

## 4. Expectation and Variance — "What's the Average? How Spread Out Is It?"

### 4.1 Expected Value (Mean) — "The Balancing Point"

#### The Core Intuition

The expected value is the **long-run average** — if you could repeat an experiment infinitely many times, what would the average outcome be?

> Think of it as the center of gravity of a probability distribution.

#### Formally

**Discrete:**
$$E[X] = \sum_{x} x \cdot P(X = x)$$

**Continuous:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

#### Example: Expected Value of a Die Roll

$$E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} = 3.5$$

You'll never actually roll a 3.5, but over many rolls, your average will converge to 3.5.

#### Key Properties (These Save You Constantly)

| Property | Formula | Why It Matters |
|---|---|---|
| **Linearity** | $E[aX + bY] = aE[X] + bE[Y]$ | Always true, even if X and Y are dependent! |
| **Constant** | $E[c] = c$ | Expected value of a constant is itself |
| **Product (independent)** | $E[XY] = E[X] \cdot E[Y]$ | Only when X and Y are independent |

> [!NOTE]
> **Linearity of expectation** is arguably the most useful property in probability. It doesn't require independence! This is why we can compute the expected loss of a model as a sum of expected losses over individual data points.

#### Expected Value of a Function

If you have a random variable $X$ and a function $g$:

$$E[g(X)] = \sum_x g(x) \cdot P(X = x)$$

This is called the **Law of the Unconscious Statistician** (LOTUS) — because people use it unconsciously.

**ML connection:** The loss function $L(\theta)$ that we minimize is actually $E[\ell(y, \hat{y})]$ — the expected loss over the data distribution. In practice, we approximate this expectation with the average over our training set.

---

### 4.2 Variance and Standard Deviation — "How Unpredictable Is It?"

#### The Core Intuition

Variance measures **how spread out** the values are from the mean. A small variance means values cluster tightly around the mean; a large variance means they're scattered.

> **Variance = the average of squared deviations from the mean**

#### Formally

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

The second form ($E[X^2] - (E[X])^2$) is the **computational formula** — often easier to calculate.

**Standard deviation** is just the square root of variance, bringing it back to the original units:

$$\sigma = \sqrt{\text{Var}(X)}$$

#### Example: Comparing Two Classifiers

| Metric | Classifier A (Accuracy per run) | Classifier B (Accuracy per run) |
|---|---|---|
| Run 1 | 90% | 85% |
| Run 2 | 88% | 95% |
| Run 3 | 91% | 80% |
| Run 4 | 89% | 96% |
| Run 5 | 92% | 94% |
| **Mean** | **90%** | **90%** |
| **Variance** | **2.0** | **42.0** |

Same mean accuracy, but Classifier A is much more **reliable**. Variance captures this.

#### Key Properties

| Property | Formula | Note |
|---|---|---|
| **Scaling** | $\text{Var}(aX) = a^2 \cdot \text{Var}(X)$ | Scaling by $a$ scales variance by $a^2$ |
| **Shift** | $\text{Var}(X + c) = \text{Var}(X)$ | Adding a constant doesn't change spread |
| **Sum (independent)** | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ | Only when X, Y are independent |
| **Sum (general)** | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | When they might be dependent |

---

### 4.3 Covariance and Correlation — "Do These Move Together?"

#### Covariance

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

| Cov(X,Y) | Meaning |
|---|---|
| > 0 | X and Y tend to increase together |
| < 0 | When X increases, Y tends to decrease |
| = 0 | No *linear* relationship (but could still be dependent!) |

#### Correlation (Normalized Covariance)

$$\rho(X, Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y} \in [-1, 1]$$

Correlation is covariance scaled to $[-1, 1]$, making it **unit-free** and comparable across different variables.

**ML connections:**
- **Feature selection:** Highly correlated features → multicollinearity → unstable linear regression
- **PCA:** Finds directions of maximum variance by eigendecomposing the covariance matrix
- **Batch normalization:** Keeps activations' means near 0 and variances near 1 throughout training

---

### 4.4 The Bias-Variance Tradeoff (Connecting It All)

For any model's prediction, the expected test error decomposes as:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Term | Meaning | Caused By |
|---|---|---|
| **Bias²** | How far off the average prediction is from the truth | Model too simple (underfitting) |
| **Variance** | How much predictions vary across different training sets | Model too complex (overfitting) |
| **Noise** | Inherent randomness in the data | Cannot be reduced |

> [!IMPORTANT]
> This is where expectation and variance become *directly operational* in ML. The bias-variance tradeoff is the reason we need regularization, cross-validation, and ensemble methods (bagging reduces variance; boosting reduces bias).

---

## 5. Maximum Likelihood Estimation (MLE) — "Finding the Best Parameters"

### The Core Intuition

MLE answers the question:

> **What parameter values make my observed data most probable?**

You've already collected data. Now you're asking: "Of all the possible parameter values, which ones would have made this data most likely to appear?"

### The Setup

You have:
- Data: $x_1, x_2, \ldots, x_n$ (assumed to be i.i.d. — independent and identically distributed)
- A model with parameter(s) $\theta$
- Each data point has probability $P(x_i | \theta)$

### Step 1: Write the Likelihood Function

The **likelihood** is the probability of seeing *all* the data, given the parameter:

$$L(\theta) = P(x_1, x_2, \ldots, x_n | \theta) = \prod_{i=1}^{n} P(x_i | \theta)$$

The product comes from the independence assumption.

### Step 2: Take the Log (Always!)

Products are numerically nasty (underflow!) and hard to differentiate. Taking the log turns products into sums:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta)$$

Since $\log$ is monotonically increasing, maximizing $\ell(\theta)$ gives the same $\theta$ as maximizing $L(\theta)$.

> [!TIP]
> This is why ML loss functions are sums (or averages) rather than products — we're always working in log-space.

### Step 3: Maximize

Take the derivative, set it to zero, solve for $\theta$:

$$\frac{d\ell}{d\theta} = 0 \implies \theta_{\text{MLE}}$$

---

### Worked Example 1: MLE for a Coin (Bernoulli)

**Data:** You flip a coin 10 times and get 7 heads and 3 tails.

**Model:** Each flip follows $\text{Bernoulli}(p)$, so $P(x_i = 1 | p) = p$ and $P(x_i = 0 | p) = 1 - p$.

**Likelihood:**

$$L(p) = p^7 \cdot (1-p)^3$$

**Log-likelihood:**

$$\ell(p) = 7\log(p) + 3\log(1-p)$$

**Maximize:** Take the derivative and set to zero:

$$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0$$

$$7(1-p) = 3p \implies 7 - 7p = 3p \implies p = 0.7$$

**Result:** $p_{\text{MLE}} = 7/10 = 0.7$

That's exactly the fraction of heads — which is exactly what your intuition says!

> In general, for a Bernoulli, $p_{\text{MLE}} = \frac{\text{number of successes}}{n}$

---

### Worked Example 2: MLE for Gaussian (The Big One)

**Data:** $x_1, x_2, \ldots, x_n$ drawn from $\mathcal{N}(\mu, \sigma^2)$.

**Log-likelihood:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Maximize w.r.t. $\mu$:**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\mu_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x} \quad \text{(the sample mean!)}$$

**Maximize w.r.t. $\sigma^2$:**

$$\sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2 \quad \text{(the sample variance!)}$$

The MLE parameters are just the sample mean and variance. MLE is formalizing what you'd do intuitively.

---

### MLE = Minimizing Cross-Entropy = Minimizing NLL

Here's the key insight connecting MLE to neural network training:

$$\theta_{\text{MLE}} = \arg\max_\theta \sum_{i} \log P(x_i | \theta) = \arg\min_\theta \left(-\sum_{i} \log P(x_i | \theta)\right)$$

That last term — $-\sum \log P(x_i | \theta)$ — is the **Negative Log-Likelihood (NLL)**.

- For **classification** (categorical output), NLL becomes **cross-entropy loss**
- For **regression** (Gaussian output), NLL becomes **mean squared error**

> [!IMPORTANT]
> **When you train a neural network by minimizing cross-entropy loss, you are doing MLE.** You're finding the parameters that make the training data most probable under the model.

### The Connections Summarized

| ML Loss Function | Distribution Assumption | Derived From |
|---|---|---|
| Mean Squared Error (MSE) | Gaussian noise | MLE of $\mathcal{N}(\mu, \sigma^2)$ |
| Cross-Entropy Loss | Categorical distribution | MLE of categorical/Bernoulli |
| MSE + L2 penalty | Gaussian noise + Gaussian prior on weights | MAP estimation |
| Cross-Entropy + L1 penalty | Categorical + Laplace prior on weights | MAP estimation |

---

## 6. Putting It All Together — The Full Picture

Here's how every concept in this guide connects to form the statistical backbone of ML:

```
PROBABILITY (Foundation)
    │
    ├── Conditional Probability P(Y|X)
    │       → Every classifier estimates this
    │
    ├── DISTRIBUTIONS (Section 2)
    │       → Define the "shape" of our assumptions
    │       → Gaussian → MSE loss
    │       → Bernoulli/Categorical → Cross-entropy loss
    │
    ├── BAYES' THEOREM (Section 3)
    │       → Naive Bayes classifier
    │       → MAP estimation = MLE + prior = regularization
    │       → Full Bayesian inference → uncertainty quantification
    │
    ├── EXPECTATION & VARIANCE (Section 4)
    │       → Expected loss = objective function
    │       → Bias-Variance tradeoff → model selection
    │       → Covariance → PCA, feature analysis
    │
    └── MLE (Section 5)
            → Training = minimizing NLL
            → Cross-entropy, MSE are special cases
            → Gradient descent finds θ_MLE when no closed form exists
```

> [!TIP]
> **The grand unification:** Training a neural network is finding $\theta_{\text{MLE}}$ (or $\theta_{\text{MAP}}$ with regularization) by minimizing the negative log-likelihood via gradient descent (calculus guide), where the likelihood is defined by a probability distribution (this guide), and the model capacity is controlled by the bias-variance tradeoff (expectation and variance).

---

## 7. Quick Reference — Formulas You'll Use Most

| Concept | Formula |
|---|---|
| Conditional probability | $P(A\|B) = P(A,B) / P(B)$ |
| Bayes' theorem | $P(H\|D) = P(D\|H) \cdot P(H) / P(D)$ |
| Expected value | $E[X] = \sum x \cdot P(x)$ |
| Variance | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| Covariance | $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$ |
| Gaussian PDF | $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2 / 2\sigma^2}$ |
| Log-likelihood | $\ell(\theta) = \sum_i \log P(x_i \| \theta)$ |
| MLE | $\theta_{\text{MLE}} = \arg\max_\theta \ell(\theta)$ |
| Bias-Variance | $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$ |
