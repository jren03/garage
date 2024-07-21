# algorithms

This directory contains the following three files, which we discuss in further detail below:
* [`model_free_irl.py`](#model-free-inverse-reinforcement-learning)
* [`model_based_irl.py`](#model-based-inverse-reinforcement-learning) 
* [`pretrain_antmaze.py`](#antmaze-model-pretraining)


## Model-Free Inverse Reinforcement Learning
At a high level, standard inverse reinforcement learning involves an actor and discriminator. The objective of the discriminator is to differentiate between learner trajectories and expert trajectories. In other words, for each state-action pair it sees, the discriminator essentially has to classify whether it is comes from the expert or learner policy. 

In practice, this is done by training the discriminator to assign higher reward (or lower cost) to the expert state-action pairs and lower reward (or higher cost) to the learner state-action pairs. The actor is trained by running any typical reinforcement learning algorithm to maximize reward under the current reward function. This process is repeated in an iterative fashion until the actor performance converges to the expert. In theory, this means that at convergence, the actor is able to "fool" the discriminator and exactly imitate the expert.

`model_free_irl.py` implements a number of variants slightly different from the standard inverse reinforcement learning approach, which we enumerate below. 

1. All model-free IRL implementations relabel each state-action pair sampled from the buffer with the discriminator function when training the actor, as opposed to the reward assigned to that state-action pair when it was added into the buffer, see lines [here](https://github.com/jren03/garage/blob/main/garage/models/sac.py#L96).
2. For `FILTER` specifically, we reset the learner to expert states via the `ResetWrapper`. For `antmaze` environments, we always reset to the expert states. However, for mujoco locomotion environments, we follow the example from [FastIRL](https://github.com/gkswamy98/fast_irl/tree/master) and rest to expert states 50% of the time, which is empirically more performant. Note that in practice this is done by setting `cfg.algorithm.reset_prob=0.5`. *Only `FILTER` sets this variable to a non-zero value*. See lines [here](https://github.com/jren03/garage/blob/main/garage/algorithms/model_free_irl.py#L56).
3. For `HyPE`, we employ *hybrid training*. This means that in practice we sample both expert and learner states from the replay buffer to update our actor. This is done through the `HybridReplayBuffer`. See lines [here](https://github.com/jren03/garage/blob/main/garage/utils/replay_buffer.py#L178).


## Model-Based Inverse Reinforcement Learning

Model-based IRL learns to model the environment dynamics while training the actor and discriminator. Notably, model-based IRL trains the actor purely within the learned model, thus incurring no environment interaction penalty when doing policy improvement. `model_based_irl.py` is implemented as follows:

1. We first initialize the model, actor, and discriminator. Following the implementation from `lamps.py` in [LAMPS-MBRL](https://github.com/vvanirudh/LAMPS-MBRL/blob/master/MujocoSysID/mbrl/algorithms/lamps.py), we train the model on a mixture of learner and expert data. We note that this differs from theory, which requires resets solely to expert states. However, in practice reseting to a mixture leads to better empirical performance.
2. The actor then interacts with the learned model to populate it's replay buffer. For every interaction, it is updated `overrides.num_policy_updates_per_step` number of times. 
3. The model is updated every `overrides.freq_train_model` number of steps that the actor takes.
4. The discriminator is updated every `overrides.discriminator.train_every` number of steps that the actor takes. 

A total of four buffers are used in `model_based_irl.py`, which we detail below for clarity:

1. `hybrid_buffer`: Contains both offline demonstrations and transition tuples of the actor interacting with the *real* environment. This buffer is sampled from to train the model. 
2. `policy_buffer`: Contains *only* transition tuples of the actor interacting with the real environment. When updating the actor, this buffer is sometimes used to reset the model to states the actor actually visited in the real environment.
3. `expert_buffer`: Contains *only* offline demonstrations. When updating the actor, this buffer is sometimes used to reset the model to states from the offline dataset.
4. `learner_buffer`: Contains rollouts of the actor purely within the learned model. This buffer is sampled from directly to update the actor. 

Furthermore, we make the following modifications to standard model-based inverse reinforcement learning:

1. Following [LAMPS-MBRL](https://github.com/vvanirudh/LAMPS-MBRL/blob/master/MujocoSysID/mbrl/algorithms/lamps.py), we train the model using a mixture of offline data from the expert and online data collected by the learner. While in theory the model should only be reset to expert states, a mixed sampling leads to better performance in practice. See lines [here](https://github.com/jren03/garage/blob/main/garage/algorithms/model_based_irl.py#L258).
2. For mujoco environments, we train the learner within the learned model by resetting to a mixture of learner or expert states. For `antmaze` environments, we reset to expert states backwards in time in a sliding window fashion. For example, after 20% of the training is complete, we reset to the last 15-20% of expert states, assuming our sliding window is 5% of the task horizon. See lines [here](https://github.com/jren03/garage/blob/main/garage/algorithms/model_based_irl.py#L269).
3. Similar to [above](#model-free-inverse-reinforcement-learning), we relabel all state-action pairs sampled for updating the actor with the current discriminator function. See lines [here](https://github.com/jren03/garage/blob/main/garage/mbrl/third_party/pytorch_sac_pranz24/sac.py#L329).


## Antmaze Model Pretraining

We find that the offline dataset for `antmaze` has an uneven distribution over the state space. Thus, we perform pretraining by partitioning the state-space into bins, then sampling transition tuples from each bin proportional to that bin's density. This proportional sampling is specifically enforced in the pre-training phase, thus allowing for a learned model that much more accurately models transition dynamics in all regions of the environment.
