import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

toy_data = tfd.Normal(loc = 4, scale = 4).sample(5000)

# Define join probability
def joint_log_prob(data, mu):
    prior = tfd.Uniform(low = 0, high = 5)
    likelihood = tfd.Normal(loc = mu, scale = 4)
    return prior.log_prob(mu) + sum(likelihood.log_prob(data))

# Sample from posterior distributions
num_results = int(1e4)
num_burnin_steps = int(1e3)

data_length = [1, 5,  10,  20, 50]
samples_all = []
for i in range(5):
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob(toy_data[:data_length[i]], *args)

    hmc = tfp.mcmc.MetropolisHastings(
        tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size=0.5,
            num_leapfrog_steps=3))

    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state = tf.zeros(1),
        kernel=hmc)

    samples_all.append(samples.numpy())

# Plot prior and posterior distributions
plt.figure(figsize(12.5, 12.5))

ax = plt.subplot(321)
plt.hist(tfd.Uniform(low = 0, high = 5).sample(1e4), normed = True, bins = 50)
plt.title("Prior distribution")
plt.ylim(0, 0.8)

for i in range(5):    
    ax = plt.subplot(3, 2, i + 2)
    plt.hist(samples_all[i], histtype = 'stepfilled', bins = 50, alpha = 0.85,
             label = None, normed = True)
    plt.title("Posterior distributions of $\mu$ with " + str(data_length[i]) + " data points")
    plt.vlines(np.median(samples_all[i]), 0, 0.8, linestyle="--", label = "MAP estimator")
    plt.legend(loc = "upper left")
    plt.xlim(0, 5)
