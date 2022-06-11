import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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
