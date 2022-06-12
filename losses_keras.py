import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

# Heteroscedastic Normal distribution loss
def normal_heteroscedastic(y_true, y_pred):
    if len(y_pred.shape) == 2:
        mu_pred = y_pred[:,0]
        sigma_pred = tf.math.exp(y_pred[:, 1])
    elif len(y_pred.shape) == 1:
        mu_pred = y_pred[0]
        sigma_pred = tf.math.exp(y_pred[1])

    dist = tfd.Normal(loc=mu_pred, scale=sigma_pred)
    loss = dist.log_prob(y_true)
    return - tf.reduce_mean(loss)

# Gamma loss
def gamma_loss(y_true, y_pred):
    if len(y_pred.shape) == 2:
        alpha_pred = tf.math.exp(y_pred[:,0])
        beta_pred = tf.math.exp(y_pred[:, 1])
    elif len(y_pred.shape) == 1:
        alpha_pred = tf.math.exp(y_pred[0])
        beta_pred = tf.math.exp(y_pred[1])

    dist = tfd.Gamma(concentration=alpha_pred, rate=beta_pred)
    loss = dist.log_prob(y_true)
    return - tf.reduce_mean(loss)

# Poisson loss
def poisson_loss(y_true, y_pred):
    mu_pred = tf.math.exp(y_pred)
    dist = tfd.Poisson(rate=mu_pred)
    loss = dist.log_prob(y_true)
    return - tf.reduce_mean(loss)

#  Negative Binomial loss
def negbin_loss(y_true, y_pred):
    # Batches
    if len(y_pred.shape) == 2:
        log_r = y_pred[:,0]
        logit_p = y_pred[:,1]

        r = tf.math.exp(log_r)
        p = tf.math.sigmoid(logit_p)
        dist = tfd.NegativeBinomial(r, p)
        loss = dist.log_prob(y_true)
        return - tf.reduce_mean(loss)

    # Single observations
    elif len(y_pred.shape) == 1:
        log_r = y_pred[0]
        logit_p = y_pred[1]

        r = tf.math.exp(log_r)
        p = tf.math.sigmoid(logit_p)
        dist = tfd.NegativeBinomial(r, p)
        loss = dist.log_prob(y_true)
        return - tf.reduce_mean(loss)

def main():
    # Test Normal heteroscedastic loss
    y_pred = np.random.randn(2)
    y_true = np.random.randn(1)
    loss = normal_heteroscedastic(y_true, y_pred)
    print(loss)

    y_pred = np.random.randn(3,2)
    y_true = np.random.randn(3)
    loss = normal_heteroscedastic(y_true, y_pred)
    print(loss)

    # Test Gamma loss
    y_pred = np.random.randn(2)
    y_true = tf.math.exp(np.random.randn(1)-1)
    loss = gamma_loss(y_true, y_pred)
    print(loss)

    y_pred = np.random.randn(3,2)
    y_true = tf.math.exp(np.random.randn(3)-1)
    loss = gamma_loss(y_true, y_pred)
    print(loss)

    # Test Poisson loss
    y_pred = np.random.randn(1)
    y_true = 2

    loss = poisson_loss(y_true, y_pred)
    print(loss)

    y_pred = np.random.randn(3)
    y_true = np.array([0,3,1])

    loss = poisson_loss(y_true, y_pred)
    print(loss)

    # Test Negative Binomial loss
    y_pred = np.random.randn(2)
    y_true = 2

    loss = negbin_loss(y_true, y_pred)
    print(loss)

    y_pred = np.random.randn(3,2)
    y_true = np.array([0,3,1])

    loss = negbin_loss(y_true, y_pred)
    print(loss)

if __name__ == '__main__':
    main()