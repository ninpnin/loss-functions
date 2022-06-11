import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

def normal_heteroscedastic(y_true, y_pred):
    mu_pred = y_pred[:,0]
    sigma_pred = tf.math.exp(y_pred[:, 1])
    loss = 0.0
    no_of_examples = y_pred.shape[0]
    for ix in range(no_of_examples):
        dist = tfd.Normal(loc=mu_pred[ix], scale=sigma_pred[ix])
        loss += dist.log_prob(y_true[ix])
    return - loss / no_of_examples

def poisson_loss(y_true, y_pred):
    mu_pred = tf.math.exp(y_pred)
    dist = tfd.Poisson(rate=mu_pred)
    loss = dist.log_prob(y_true)
    return - tf.reduce_mean(loss)

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
    y_pred = np.random.randn(2)
    y_true = 2

    loss = negbin_loss(y_true, y_pred)
    print(loss)

    y_pred = np.random.randn(3,10)
    y_true = np.array([0,3,1])

    loss = negbin_loss(y_true, y_pred)
    print(loss)

if __name__ == '__main__':
    main()