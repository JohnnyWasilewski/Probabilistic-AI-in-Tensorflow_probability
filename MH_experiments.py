import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import time
from scipy import stats

hmc = tfp.mcmc.MetropolisHastings(
        tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn = lambda x:  - x**2 ,
        step_size=0.5,
        num_leapfrog_steps=40))

# Time comparison
t_0_mcmc = time.time()

samples_1000, is_accepted = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=0,
        current_state = tf.zeros(1),
        kernel=hmc)

t_1_mcmc = time.time() - t_0_mcmc

t_0 = time.time()
s = np.random.normal(0, 1, 1000)
t_1 = time.time() - t_0 

# Distribution evaluation
samples_100, is_accepted = tfp.mcmc.sample_chain(
        num_results=100,
        num_burnin_steps=0,
        current_state = tf.zeros(1),
        kernel=hmc)


samples_200, is_accepted = tfp.mcmc.sample_chain(
        num_results=200,
        num_burnin_steps=0,
        current_state = tf.zeros(1),
        kernel=hmc)

samples_100_burn, is_accepted = tfp.mcmc.sample_chain(
        num_results = 100,
        num_burnin_steps = 500,
        current_state = tf.zeros(1),
        kernel=hmc)

stats.kstest(samples_100.numpy().reshape(1,-1)[0],'norm')
stats.kstest(samples_200.numpy().reshape(1,-1)[0],'norm')
stats.kstest(samples_100_burn.numpy().reshape(1,-1)[0],'norm')
stats.kstest(samples_1000.numpy().reshape(1,-1)[0],'norm')
