# On loss functions - a probabilistic perspective

## What is a loss function and what does it have to do with distributions?

In a practical sense, loss functions quantify the discrepancy of a model, a function $\hat f : X \to Y$, compared to the true data, which lies in $X \times Y$. It quantifies the difference between the data point $y \in Y$ in the codomain and the model prediction $\hat y \in Y$ as a real number that can be differentiated with respect to the model parameters.

However, the data usually comes from a non-determinsitic process. This means that the codomain is a space of probability distributions given some input data in $x \in X$, rather than a space of eg. real or natural numbers. Thus, a suitable model would also yield probability distributions given the input data $x \in X$.

In a regression task . This can be thought of as a reasonable output value – perhaps the mean or the median of the output distribution.

On the other hand, this is equivalent to minimizing the KL divergence between the estimated probability distribution and the empirical distribution we obtain from the data.

## Normally distributed codomain

Oftentimes, the codomain is best representated as an unrestricted real space. Be it heights of people, salaries, velocities, time or countless other things, real numbers are the best way to capture that quantity. Moreover, a common way to approach such a codomain is to model it via the Normal distribution.

The Normal distribution has the following point probability

$$
p(x \mid \mu, \sigma) =\frac{1}{\sqrt{2 \pi \sigma^2}}e^{-\frac{1}{2 \sigma^2}(x-\mu)^2}
\\
\log p(x \mid \mu, \sigma) = -\frac{1}{2 \sigma^2}(x-\mu)^2 - \log \sigma + C
$$

We can regress this from data with any function $[\mu(x), \sigma(x)] = f(x; \theta)$.

It is common to assume homoscedasticity when training a regression neural network. This yields then 

$$
p(x \mid \mu, \sigma) = \sum_{i=1}^N -\frac{1}{2 \sigma(x_i)^2}(y_i -\mu(x_i))^2 + C
\\ = \sum_{i=1}^N -\frac{1}{2 \sigma_0^2}(y_i-\mu(x_i))^2 + C
\\ = -\frac{1}{2 \sigma_0^2} \sum_{i=1}^N (y_i -\mu(x_i))^2 + C
$$

which, in terms of maximization with respect to $\mu$, is equivalent to

$$
- \sum_{i=1}^N (y_i-f(x_i; \theta))^2
$$

i.e. the mean squared error.

Indeed, mean squared error is the most commonly used loss function in neural networks for regression tasks. Implicitly, we are performing the mean regression on a Normal distribution in the codomain, assuming homoscedasticity. Usually, we just don't estimate the variance when training neural networks–but we could and can if it's useful. We can also relax the assumption of homoscedasticity if we want a non-static codomain variance wrt. the domain.

## Bernoulli distributed codomain

The Bernoulli distribution is univariate discrete, and has the support $k \in \{0, 1\}.$ Such a distribution can possibly only have one degree of freedom. We can use the success probability parametrization, though others are more convenient when actually implementing the distribution

$$
P(K=1 \mid p) = p, \\ P(K=0 \mid p) = 1 - p
$$

The log probability can be nicely written out

$$
K \log p + (1-K) \log (1-p)
$$

## Categorically distributed codomain

## Poisson distributed codomain

The Poisson distribution has the support $k \in \{0, 1, 2, \dots\}$. 

$$
\log(P(k \mid \lambda(x)) = \log(P(k \mid e^{f(x)}) \\
= \log[\frac{\lambda(x)^k e^{-\lambda(x)}}{k!}] \\
= k \log[\lambda(x)] - \lambda(x) - \log(k!) \\
= k f(x) - e^{f(x)} - \log(k!)
$$

The last term is invariant with respect to the distribution parameters, and can be thus omitted during parameter regression. We get a loss function

$$
\mathcal L(y, \hat y(x)) = y \hat y(x) - e^{\hat y(x)}
$$

Differentiating this in the plain codomain, we get

$$
\frac{\partial \mathcal L(y, \hat y)}{\partial \hat y} = 0
\\
\implies y -  e^{\hat y(x)} = 0 \\
\implies y = e^{\hat y(x)}
$$

In other words, the loss has a stationary point when the prediction equals the true value. On the other hand, both as $\hat y (x) \to \infty$ and $\hat y (x) \to -\infty$, the loss approaches infinity ($y$ is always positive). Thus, the loss is minimized when the prediction equals the observed value.

## Binomially distributed codomain

The Binomial distribution has the following probability mass function

$$
P(k \mid p) = {n \choose k} p^{k}(1-p)^{n-k}
\\
\log P(k \mid p) = k \log p + (n-k) \log (1-p) + C
$$

### Negative binomially distributed codomain

The Negative binomial distribution has the following probability mass function

$$
P(k \mid r, p) = {k + r - 1 \choose k} \cdot (1-p)^r p^k
$$

Its logarithm is

$$
\log P(k \mid r, p) = \log \left({k + r - 1 \choose k} \cdot (1-p)^r p^k \right)
\\
= \log{k + r - 1 \choose k} + r \log (1-p) + k \log p
$$

Per definition, $r$ is a natural number. The definition can however be extended so that $r$ is a positive real number instead. This can be achieved using the Gamma function instead of the factorial when calculating the combinatorial coefficient

$$
\log{k + r - 1 \choose k} + r \log (1-p) + k \log p
\\
= \log\frac{\Gamma(k+r-1)}{\Gamma(r)\Gamma(k-1)} + r \log (1-p) + k \log p
\\
= \log \Gamma(k+r-1) - \log \Gamma(r) + r \log (1-p) + k \log p + C
\\
= \sum_{i=0}^k \log (r+i) + r \log (1-p) + k \log p + C
$$

### Example: NHL finals

$$
\mathcal L(\hat y, y) = \sum_{i=1}^N y_i \log \hat y(x_i)
+ (n-y_i)\log (1-\hat y(x_i))
$$

## Bi- and multivariate codomains
