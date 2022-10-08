from probabilistic_loss_functions.losses import *
import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Test(unittest.TestCase):
    def test_shapes(self):
        y_pred = np.random.randn(2)
        y_true = np.random.randn(1)
        loss = normal_heteroscedastic(y_true, y_pred)

        y_pred = np.random.randn(3,2)
        y_true = np.random.randn(3)
        loss = normal_heteroscedastic(y_true, y_pred)

        # Test Gamma loss
        y_pred = np.random.randn(2)
        y_true = tf.math.exp(np.random.randn(1)-1)
        loss = gamma_loss(y_true, y_pred)

        y_pred = np.random.randn(3,2)
        y_true = tf.math.exp(np.random.randn(3)-1)
        loss = gamma_loss(y_true, y_pred)

        # Test Poisson loss
        y_pred = np.random.randn(1)
        y_true = 2

        loss = poisson_loss(y_true, y_pred)

        y_pred = np.random.randn(3)
        y_true = np.array([0,3,1])

        loss = poisson_loss(y_true, y_pred)

        # Test Negative Binomial loss
        y_pred = np.random.randn(2)
        y_true = 2

        loss = negbin_loss(y_true, y_pred)

        y_pred = np.random.randn(3,2)
        y_true = np.array([0,3,1])

        loss = negbin_loss(y_true, y_pred)

        # Test Beta loss
        y_pred = np.random.randn(2)
        y_true = tf.math.sigmoid(np.random.randn(1))
        loss = beta_loss(y_true, y_pred)

        y_pred = np.random.randn(3,2)
        y_true = tf.math.sigmoid(np.random.randn(3))
        loss = beta_loss(y_true, y_pred)
