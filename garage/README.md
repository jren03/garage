# garage

This document provides an overview of the `garage` codebase for users interested in trying out the code or extending it for their own research. Specifically:
* To see a high level overview of the code strucutre, see the [Code Structure](#code-structure) section. 
* To see recommendations for adapting this repository to new environment and experiments, see the [Extensions](#extensions) section.
* To see recommendations for hyperparameter tuning, see the [Hyperparameter Search](#hyperparameter-search) section.

## Code Structure
The critical components of the codebase are discussed below. They are listed in the order which we believe is most intuitive for first-time viewers. 

> [!NOTE]
> The respository is structured such that all experiments can be run through `main.py` by merely changing the config files that are passed in. 

### `algorithms`
This directory contains the main training loop for `HyPE` and `HyPER` in `model_free_irl.py` and `model_based_irl.py`, respectively. However, both files are written with the intention to be as general as possible to any model-free and model-based inverse RL algorithm. In both files, we take inspiration from the tricks in [FastIRL](https://github.com/gkswamy98/fast_irl/tree/master) to stabilize training; namely, we add gradient penalty to the discriminator, learning rate decay, and Optimistic Adam as our optimizer. For more information, please see [here](../garage/algorithms/README.md).


### `utils`
We provide a detailed breakdown of each of the files in this directory.
* `common.py`: contains logging and path constants, as well as helper functions for seed setting, agent rollouts, and plotting.
* `ema_wrapper.py`: adapted from [ema-pytorch](https://github.com/lucidrains/ema-pytorch/tree/main), a wrapper around any `nn.Module` to allow for inference with the exponential moving average of the model's weights.
* `fetch_demos.py`: fetch demonstrations from `experts/<env_name>`.
* `gym_wrappers.py`: adapted from [FastIRL](https://github.com/gkswamy98/fast_irl/tree/master), containing all `gym` wrappers used in experiments. Please reference the section [here](#a-note-on-deterministic-resets) for more information regarding our `ResetWrapper`.
* `logger.py`: for logging training results to the terminal and `garage/experiment_logs`.
* `nn_utils.py`: gradient penalties, learning rate decay, etc.
* `oadam.py`: Optimistic Adam extended from [optimistic-adam](https://github.com/georgepar/optimistic-adam/blob/master/optim.py).
* `replay_buffer.py`: for replay buffers that can sample from both policy and expert data.


### `mbrl`
`model_based_irl.py` builds off the `mbpo.py` file in [mbrl-lib](https://github.com/facebookresearch/mbrl-lib/tree/main), and thus calls on helper functions stored in `garage/mbrl`. There are two main directories of interest: `garage/mbrl/utils` and `garage/mbrl/third_party`. General utility functions such as initializing and training the model, as well as filling the replay buffer can be found in `garage/mbrl/util/common.py`. We use the same SAC optimizer as mbrl-lib, which is different from the StableBaselines3 optimizer used in model-free experiments, and it's implementation can be found in `garage/mbrl/third_party/pytorch_sac_pranz24`.


### `models`
The three model architectures in this directory are as follows:
1. `discrminator.py`: implements both the a single discriminator and ensemble of discriminators
2. `sac.py`: used in all Mujoco experiments
3. `td3_bc.py`: used in all `antmaze` experiments


## Extensions

We detail how and where one can make modifications to this codebase. In general, for all new experiments in simulation, we recommend starting first with `model_free_irl.py` since it is quicker to train, then using those hyperparameters as a starting point for `model_based_irl.py`.


### Custom Policy Optimizers
In practice, we find certain actor networks work better for others for certain environments (e.g. TD3-BC was more performant than SAC on `antmaze` environments for our set of hyperparameters.) To experiment with new models or to modify existing ones, simply add a new file under `garage/models` and update the config files under `garage/config` with a set of instantiation keywords accordingly. 

> [!NOTE]
> To ensure compatability with existing function calls, it is important to add functions for `reset()`, `act()`, `predict()`, and `learn()`. An example can be seen in `garage/models/td3_bc.py`. 

### Custom Environments 
To train and collect demonstrations for new environments, please see the details outlined on the main page, linked [here](../README.md/#collecting-demonstration-data).

### Hyperparameter Search

There are a number of parameters we found to be particularly important in stablizing `HyPE` and `HyPER` performance across various environments, which we highlight below using hydra command-line syntax:

For all experiments:
* `overrides.discriminator.lr`: initial learning rate of discriminator. We recommend searching over `[1e-3, 8e-3, 1e-4, 8e-4]` as an initial starting point.
* `overrides.discriminator.train_every`: after how many actor steps to update the discriminator. We recommend searching over `[2000, 5000, 10000]` as an initial starting point. 

For model-free experiments specifically:
* `overrides.sampling_schedule`: what percentage of expert samples to use in shared buffer update. We recommend experimenting with both a constant and decaying percentage. 

For model-based experiments specifically:
* `overrides.model_hid_size`: the size of the model. For some environments such as `Humanoid`, we find that a larger model size is importance for quicker convergence.
* `overrides.policy_updates_every_steps`: how many policy updates per model step. In practice, we find a number between 2 to 5 to work best for our environments.
* `overrides.freq_train_model`: how frequently to update the model. Especially in the case where the model is pretrained (as in `antmaze`), this value can be rather large (~1k).
* `overrides.ema_agent`: whether to use the EMA of the policy weights during inference. 
* `overrides.schedule_model_lr`: whether to decay the learning rate of the model.
* `overrides.schedule_actor_lr`: whether to decay the learning rate of the actor.
* `overrides.sac_automatic_entropy_tuning`: we find entropy to be helpful for some environments, detrimental to others.
* `overrides.decay_horizon`: the decay horizon of the actor's learning rate. For some environments, this number should match `overrides.discriminator.train_every`, while others benefitted from a longer horizon.
* `overrides.model_clip_output`: whether to clip the output of the model.
* `overrides.discriminator.clip_output`: whether to clip the output of the discriminator.
* `overrides.discriminator.weight_decay`: regularization on weights of learner.
* `overrides.discriminator.ensemble_size`: while some environments benefited from an ensemble of discriminators, others (such as `antmaze` environments) did better without.


### A Note on Deterministic Resets
We found that exactly replaying expert actions after `env.set_state(qpos, qvel)` leaves to compounding divergences in the states. This is likely due to the [warmstart acceleration](https://mujoco.readthedocs.io/en/stable/computation/index.html#warmstart-acceleration) in MuJoCo. One way to fix this is by creating a copy of the environment xmls and adding the following.
```
  <option timestep=".005">
    <flag warmstart="disable"/>
  </option>
```
Then, create the environment via `env = gym.make("EnvName-v3", xml_file="path/to/modified/xml")`. 

However, we found that making this change to the XML led to worse-performing experts when using the default SB3 [hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/e1a40be64aa7a731ce2a69ddc8d0ad510222326a/hyperparams/sac.yml#L190) from RLZoo. Therefore, we elected to perform resets to the *t*-th timestep of a trajectory by deterministically resetting to the *start state* via `env.reset(seed=seed)`, then rolling out the first *t-1* actions in the expert demonstration. This implementation can be found [here](https://github.com/jren03/garage/blob/main/garage/utils/gym_wrappers.py#L151). 

***
[[Top](#garage)] | [[Home](../README.md)]

