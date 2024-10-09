import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.utils.prune
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PER import PER
# from torchsummary import summary
from networks import ImpalaCNNLarge, ImpalaCNNLargeIQN, NatureIQN, ImpalaCNNLargeC51
import networks
from copy import deepcopy
from functools import partial
from Analytic import Analytics
import matplotlib.pyplot as plt
from utils import cal_dormant_ratio, TruncatedNormal, perturb
import math
from collections import defaultdict


class EpsilonGreedy:
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)


def randomise_action_batch(x, probs, n_actions):
    mask = torch.rand(x.shape) < probs

    # Generate random values to replace the selected elements
    random_values = torch.randint(0, n_actions, x.shape)

    # Apply the mask to replace elements in the tensor with random values
    x[mask] = random_values[mask]

    return x


def choose_eval_action(observation, eval_net, n_actions, device, rng):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()

        if rng > 0.:
            # Generate a mask with the given probability
            x = randomise_action_batch(x, 0.01, n_actions)

    return x


def create_network(impala, iqn, input_dims, n_actions, spectral_norm, device, noisy, maxpool, model_size, maxpool_size,
                   linear_size, num_tau, dueling, ncos, non_factorised, arch,
                   layer_norm=False, activation="relu", c51=False):
    if impala:
        if iqn:
            return ImpalaCNNLargeIQN(input_dims[0], n_actions, spectral=spectral_norm, device=device, noisy=noisy,
                                     maxpool=maxpool, model_size=model_size, num_tau=num_tau, maxpool_size=maxpool_size,
                                     dueling=dueling, linear_size=linear_size, ncos=ncos,
                                     arch=arch, layer_norm=layer_norm, activation=activation)
        if c51:
            return ImpalaCNNLargeC51(input_dims[0], n_actions, spectral=spectral_norm, device=device,
                                  noisy=noisy, maxpool=maxpool, model_size=model_size, linear_size=linear_size)
        else:
            return ImpalaCNNLarge(input_dims[0], n_actions, spectral=spectral_norm, device=device,
                                  noisy=noisy, maxpool=maxpool, model_size=model_size, maxpool_size=maxpool_size,
                                  linear_size=linear_size)

    else:
        return NatureIQN(input_dims[0], n_actions, device=device, noisy=noisy, num_tau=num_tau, linear_size=linear_size,
                         non_factorised=non_factorised, dueling=dueling)


class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=256
                 , rr=1, maxpool_size=6, lr=1e-4, ema=False, trust_regions=False, target_replace=500, ema_tau=0.001,
                 noisy=True, spectral=True, munch=True, iqn=True, double=False, dueling=True, impala=True,
                 discount=0.997, adamw=False, discount_anneal=False, per=True,
                 taus=8, model_size=2, linear_size=512, ncos=64, rainbow=False, maxpool=True,
                 non_factorised=False, replay_period=1, analytics=False, framestack=4,
                 rgb=False, imagex=84, imagey=84, pruning=False, arch='impala', per_alpha=0.2,
                 per_beta_anneal=False, layer_norm=False, max_mem_size=262144, c51=False,
                 eps_steps=2000000, eps_disable=True, stoch=False, tr_plus_target=False, dormant_exp=False,
                perturb=False, activation="relu", var_bs=False, learn_layer_norm=False):

        if rainbow:
            lr = 6.25e-5
            spectral = False
            munch = False
            iqn = False
            c51 = True
            double = True
            dueling = True
            impala = False
            discount = 0.99
            adamw = False
            per = True
            noisy = True
            linear_size = 512
            self.per_alpha = 0.5
        else:
            self.per_alpha = per_alpha

        self.procgen = True if input_dims[1] == 64 else False

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.learn_layer_norm = learn_layer_norm
        self.layer_norm = layer_norm
        self.activation = activation
        self.perturb = perturb


        self.loading_checkpoint = False

        self.per_beta = 0.45
        self.per_beta_anneal = per_beta_anneal
        if self.per_beta_anneal:
            self.per_beta = 0

        self.replay_ratio = int(rr) if rr > 0.99 else float(rr)
        self.total_frames = total_frames
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 4000
        else:
            self.min_sampling_size = 200000

        self.lr = lr

        self.analytics = analytics
        if self.analytics:
            self.analytic_object = Analytics(agent_name, testing)

        # this is the number of env steps per grad step
        self.replay_period = replay_period

        # replay ratio however does not take into account parallel envs

        # in this code, every {replay period} steps, we take {replay_ratio} grad steps

        self.total_grad_steps = (self.total_frames - self.min_sampling_size) / (self.replay_period / self.replay_ratio)

        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        if self.perturb:
            # perturb once every million frames
            self.perturb_every = int(self.total_grad_steps / ((total_frames*4) / 1000000))

        self.chkpt_dir = ""

        self.n = 3
        if discount_anneal:
            self.discount_anneal = True
            self.gamma = 0.97
            self.final_gamma = 0.997
            self.annealing_period = self.total_grad_steps // 2  # first half of training
            self.gamma_inc = (self.final_gamma - self.gamma) / self.annealing_period
        else:
            self.gamma = discount
            self.discount_anneal = False
        self.batch_size = batch_size
        self.var_bs = var_bs
        if self.var_bs:
            self.var_bs_sizes = np.array([8, 16, 32, 64, 128, 256])

        self.model_size = model_size  # Scaling of IMPALA network
        self.maxpool_size = maxpool_size

        self.spectral_norm = spectral  # rememberance of the bug that passed gpu tensor into env
        # and caused nans which somehow showed up in the PER sample function.

        self.noisy = noisy
        self.dormant_exp = dormant_exp
        if self.dormant_exp:
            # cannot use noisy and dormant exploration
            self.noisy = False
            self.stoch = True

            # need to make select action and update use std. This will also change with awaken

            # Dormant-ratio-guided exploitation

            self.dormant_perturb_interval = int(200000 / self.replay_period)

            self.dormant_ratio = 1
            self.min_perturb_factor = 0.2
            self.max_perturb_factor = 0.9
            self.perturb_rate = 2
            self.target_dormant_ratio = 0.2
            self.dormant_temp = 10

        # this option is only available for non-impala. I could add it, but factorised seemed
        # to perform the same and is faster
        self.non_factorised = non_factorised

        self.per_splits = 1
        if self.per_splits > num_envs:
            self.per_splits = num_envs

        self.impala = impala  # non impala only implemented for iqn
        self.dueling = dueling

        # Don't use both of these, they are mutually exclusive
        self.c51 = c51
        self.iqn = iqn

        self.ncos = ncos

        self.adamw = adamw

        self.double = double  # Not implemented for IQN and Munchausen
        self.maxpool = maxpool
        self.munchausen = munch

        self.stoch = stoch

        if self.munchausen:
            self.entropy_tau = 0.03
            self.lo = -1
            self.alpha = 0.9

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = max_mem_size
        self.tr_plus_target = tr_plus_target

        self.soft_update_tau = ema_tau  # 0.001 for non-sample-eff
        self.replace_target_cnt = target_replace  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio


        # target_net, ema, trust_region
        if ema:
            self.stabiliser = "ema"
        elif tr_plus_target:
            self.stabiliser = "trust_regions"
            self.tr_period = 1500
            self.replace_target_cnt = 500
            self.trust_regions = True
        elif trust_regions:
            self.stabiliser = "trust_regions"
            self.tr_period = 1500
        else:
            self.stabiliser = "target"

        if self.stabiliser == "ema":
            self.soft_updates = True
        else:
            self.soft_updates = False

        # NOT IMPLEMENTED
        if self.stabiliser == "trust_regions":
            self.trust_regions = True
            self.running_std = -999
        else:
            self.trust_regions = False


        # Best used value is 32000 frames per replace. For bs 256, this is 500. For bs 16, this is every 8000!

        self.tr_alpha = 1

        self.loss_type = "huber"  # This is only for non-iqn, non-munchausen, c51
        if self.loss_type == "huber":
            loss_fn_cls = nn.SmoothL1Loss
            self.loss_fn = loss_fn_cls(reduction=('none'))

        self.num_tau = taus

        if self.loading_checkpoint:
            self.min_sampling_size = 300000

        # c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        if not self.loading_checkpoint and not self.testing:
            self.eps_start = 1.0
            # divided by 4 is due to frameskip
            self.eps_steps = eps_steps
            self.eps_final = 0.01
        else:
            self.eps_start = 0.01
            self.eps_steps = eps_steps
            self.eps_final = 0.01

        self.eps_disable = eps_disable
        self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)

        self.per = per

        self.linear_size = linear_size
        self.arch = arch

        # self.memories = []
        # if self.per:
        #     for i in range(num_envs):
        #         self.memories.append(ReplayMemory(self.max_mem_size // num_envs, self.n, self.gamma, device, alpha=self.per_alpha, beta=self.per_beta, procgen=self.procgen))
        # else:
        #     for i in range(num_envs):
        #         self.memories.append(RegularReplayMemory(self.max_mem_size // num_envs, self.n, self.gamma, device))

        self.framestack = framestack
        self.rgb = rgb
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=self.rgb, imagex=imagex, imagey=imagey)

        self.network_creator_fn = partial(create_network, self.impala, self.iqn, self.input_dims, self.n_actions,
                                          self.spectral_norm, self.device,
                                          self.noisy, self.maxpool, self.model_size, self.maxpool_size,
                                          self.linear_size,
                                          self.num_tau, self.dueling, self.ncos,
                                          self.non_factorised, self.arch, layer_norm=self.layer_norm,
                                          activation=self.activation, c51=self.c51)

        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()
        if tr_plus_target:
            self.tr_net = self.network_creator_fn()

        if self.learn_layer_norm:
            for name, layer in model.named_modules():
                if name not in ["advantage_branch.1", "value_branch.1"]:  # Adjust names as needed
                    print(name)
                    layer.register_forward_hook(activation_hook)

            asda

        if self.adamw:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size,
                                         weight_decay=1e-4)  # weight decay taken from museli
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015

        self.net.train()

        self.eval_net = None

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.replay_ratio_cnt = 0
        self.eval_mode = False

        if self.loading_checkpoint:
            self.load_models("insert_model_name")

        self.prune = pruning
        if self.prune and self.spectral_norm:
            raise Exception("Cannot use both pruning and spectral norm")
        if self.prune:
            self.desired_sparsity = 0.95
            self.last_sparsity = 0.
            self.start_prune = 0.2
            self.end_prune = 0.8

    @property
    def dormant_explore_rate(self):
        return 1 / (1 +
                    math.exp(-self.dormant_temp *
                             (self.dormant_ratio - self.target_dormant_ratio)))

    def prune_network(self):
        # applies pruning
        # doesn't apply to first and last layers
        # doesn't work with spec norm, and untested with NoisyLayers
        current_progress = self.grad_steps / self.total_grad_steps

        if current_progress < self.start_prune or current_progress > self.end_prune:
            return

        cur_sparsity = self.desired_sparsity * \
                       (1 - (1 - (current_progress - self.start_prune) / (self.end_prune - self.start_prune)) ** 3)

        apply_sparsity = (cur_sparsity - self.last_sparsity) / (1 - self.last_sparsity)

        # print("Applied Sparsity")
        # print(apply_sparsity)
        #
        # print("Target Sparsity")
        # print(cur_sparsity)

        for name, module in self.net.named_modules():
            if name.endswith("2") or name == "cos_embedding" or name.startswith("conv.0"):
                continue

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=apply_sparsity)

        self.last_sparsity = cur_sparsity

    def get_grad_steps(self):
        return self.grad_steps

    def prep_evaluation(self):
        self.eval_net = deepcopy(self.net)
        self.disable_noise(self.eval_net)

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation):
        # this chooses an action for a batch. Can be used with a batch of 1 if needed though
        with T.no_grad():
            if self.noisy and not self.eval_mode:
                self.reset_noise(self.net)

            state = T.tensor(observation, dtype=T.float).to(self.net.device)

            if self.dormant_exp or self.stoch:

                qvals = self.net.qvals(state, advantages_only=False)
                probs = F.softmax(qvals / self.entropy_tau, dim=1)
                x = torch.multinomial(probs, num_samples=1).cpu()
            else:
                qvals = self.net.qvals(state, advantages_only=True)
                x = T.argmax(qvals, dim=1).cpu()

            if self.env_steps < self.min_sampling_size or not self.noisy or \
                    (self.env_steps < self.total_frames / 2 and self.eps_disable):

                probs = self.epsilon.eps
                x = randomise_action_batch(x, probs, self.n_actions)

            return x

    def store_transition(self, state, action, reward, next_state, done, stream, prio=True):

        if self.rgb:
            # expand dims to create "framestack" dim, so it works with my replay buffer
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

        self.memory.append(state, action, reward, next_state, done, stream, prio=prio)

        self.epsilon.update_eps()
        self.env_steps += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)

    def perturb_factor(self):
        # when dormant ratio is LOW ie 0
        # this number (perturb factor) will be high
        # in the perturb method, they do 1- perturb factor.

        # This means that high dormant ratio results in more noise
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        with torch.no_grad():
            for target_param, local_param in zip(self.tgt_net.parameters(), self.net.parameters()):
                target_param.data.copy_(
                    self.soft_update_tau * local_param.data + (1.0 - self.soft_update_tau) * target_param.data)

    def activation_hook(module, input, output):
        activations[module].append(output.detach())

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % (int(1 / self.replay_ratio))
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        # if self.per:
        #     for i in range(self.num_envs):
        #         self.memories[i].priority_weight = min(self.memories[i].priority_weight + self.priority_weight_increase, 1)

        if self.per and self.per_beta_anneal:
            self.memory.beta = min(self.memory.beta + self.priority_weight_increase, 1)

        if self.discount_anneal:
            self.gamma = min(self.gamma + self.gamma_inc, self.final_gamma)
            for i in self.memories:
                i.discount = self.gamma

        if self.noisy:
            self.reset_noise(self.tgt_net)

        if not self.soft_updates:

            if self.tr_plus_target:
                if self.grad_steps % self.tr_period == 0:
                    self.tr_net.load_state_dict(self.net.state_dict())
                if self.grad_steps % self.replace_target_cnt == 0:
                    self.replace_target_network()
            elif self.trust_regions:
                if self.grad_steps % self.tr_period == 0:
                    self.replace_target_network()
            else:
                if self.grad_steps % self.replace_target_cnt == 0:
                    self.replace_target_network()
        else:
            self.soft_update()

        if self.prune:
            self.prune_network()

        if self.dormant_exp:
            if self.grad_steps % self.dormant_perturb_interval == self.dormant_perturb_interval - 1:
                perturb(self.net, self.optimizer, self.perturb_factor())
                perturb(self.tgt_net, self.optimizer, self.perturb_factor())

        if self.perturb and self.grad_steps % self.perturb_every == 0:
            # this also resets the optimzer
            perturb(self.net, self.optimizer, 0.2)
            perturb(self.tgt_net, self.optimizer, 0.2)

        # states, rewards, actions, next_states, dones, weights, idxs, mems, mem = self.sample()
        # states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size)

        if self.var_bs and self.grad_steps > self.total_grad_steps * 0.25:
            self.batch_size = np.random.choice(self.var_bs_sizes)

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        if self.learn_layer_norm:
            activations = defaultdict(list)

        self.optimizer.zero_grad()

        # use this code to check your states are correct, check states

        # plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()


        if self.c51:
            distr_v, qvals_v = self.net.both(states)
            state_action_values = distr_v[range(self.batch_size), actions.data]
            state_log_sm_v = F.log_softmax(state_action_values, dim=1)

            with torch.no_grad():
                next_distr_v, next_qvals_v = self.tgt_net.both(next_states)
                action_distr_v, action_qvals_v = self.net.both(next_states)

                next_actions_v = action_qvals_v.max(1)[1]

                next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
                next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
                next_best_distr = next_best_distr_v.data.cpu()

                proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax,
                                              self.N_ATOMS, self.gamma ** self.n)

                proj_distr_v = proj_distr.to(self.net.device)

            loss_v = -state_log_sm_v * proj_distr_v
            if self.per:
                weights = T.squeeze(weights)
                loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

            loss = loss_v.mean()

        elif not self.iqn and not self.c51 and not self.munchausen:  # non distributional

            indices = np.arange(self.batch_size)

            q_pred = self.net.forward(states)
            q_pred = q_pred[indices, actions]

            with torch.no_grad():
                q_targets = self.tgt_net.forward(next_states)
                if self.double:
                    q_actions = self.net.forward(next_states)
                else:
                    q_actions = q_targets.clone().detach()

                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_target = rewards + (self.gamma ** self.n) * q_targets[indices, max_actions]

            # loss_v should be absolute error for PER
            td_error = q_target - q_pred
            loss_v = torch.abs(td_error)

            if self.loss_type == "mse":
                if self.per:
                    loss_squared = (td_error.pow(2) * weights.to(self.net.device))
                else:
                    loss_squared = td_error.pow(2)

                loss = loss_squared.mean().to(self.net.device)

            elif self.loss_type == "huber":
                losses = self.loss_fn(q_target, q_pred)
                loss = torch.mean(weights.to(self.net.device) * losses)
            else:
                raise Exception("Unknown loss type")

        elif not self.iqn and not self.c51 and self.munchausen:  # non-distributional but with munchausen

            with torch.no_grad():

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                # if self.per:
                #     weights = weights.unsqueeze(1)

                Q_targets_next = self.tgt_net.forward(next_states)

                logsum = torch.logsumexp((Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1)) / self.entropy_tau,
                                         1).unsqueeze(-1)

                tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum

                # target policy
                pi_target = F.softmax(Q_targets_next / self.entropy_tau, dim=1)
                # Q_target = (self.gamma * (pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(1)).unsqueeze(-1)
                Q_target = (self.gamma ** self.n * (
                        pi_target * (Q_targets_next - tau_log_pi_next) * (~dones)).sum(1)).unsqueeze(1)

                # calculate munchausen addon with logsum trick
                q_k_targets = self.tgt_net(states)
                v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
                logsum = torch.logsumexp((q_k_targets - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)
                log_pi = q_k_targets - v_k_target - self.entropy_tau * logsum
                munchausen_addon = log_pi.gather(1, actions)

                # calc munchausen reward:
                munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0))

                Q_targets = munchausen_reward + Q_target

            q_k = self.net(states)
            Q_expected = q_k.gather(1, actions)

            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).squeeze()

            if self.per:
                loss_squared = (td_error.pow(2) * weights.to(self.net.device))
            else:
                loss_squared = td_error.pow(2)

            loss = loss_squared.mean().to(self.net.device)

        elif self.iqn and not self.munchausen:

            with torch.no_grad():

                if self.trust_regions:
                    Q_targets_next, _ = self.net(next_states)
                else:
                    Q_targets_next, _ = self.tgt_net(next_states)

                if self.double:  # this may be wrong - seems to perform better without. Could just be chance though
                    indices = np.arange(self.batch_size)
                    q_actions = self.net.qvals(next_states)
                    max_actions = T.argmax(q_actions, dim=1)
                    Q_targets_next = Q_targets_next[indices, :, max_actions].detach().unsqueeze(1)
                else:
                    Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                if self.per:
                    weights = weights.unsqueeze(1)

                # Compute Q targets for current states
                Q_targets = rewards.unsqueeze(-1) + (
                        self.gamma ** self.n * Q_targets_next * (~dones.unsqueeze(-1)))

            # Get expected Q values from local model
            Q_expected, taus = self.net(states)
            Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

            # Quantile Huber loss
            td_error = Q_targets - Q_expected

            # get absolute losses for all taus
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            # assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"

            # calculate huber loss between prediction and target
            huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)  # note this gives all positive values

            # Multiply by the taus - this is what actually makes the quantiles, and also applies the sign
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            # sum the losses
            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # keepdim=True if using PER

            if self.per:
                loss = loss * weights.to(self.net.device)

            if self.trust_regions:
                loss = self.calculate_trust_regions(loss, loss_v, states, actions, Q_expected, Q_targets)

            loss = loss.mean()

        elif self.iqn and self.munchausen:
            with torch.no_grad():

                if self.trust_regions and not self.tr_plus_target:
                    Q_targets_next, _ = self.net(next_states)
                else:
                    Q_targets_next, _ = self.tgt_net(next_states)

                # (batch, num_tau, actions)
                q_t_n = Q_targets_next.mean(dim=1)

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                if self.per:
                    weights = weights.unsqueeze(1)

                # calculate log-pi
                logsum = torch.logsumexp(
                    (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
                # assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)

                pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

                Q_target = (self.gamma ** self.n * (
                        pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)

                # assert Q_target.shape == (self.batch_size, 1, self.num_tau)

                q_k_target = self.net.qvals(states)
                v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
                tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp(
                    (q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

                # assert tau_log_pik.shape == (self.batch_size, self.n_actions), "shape instead is {}".format(
                # tau_log_pik.shape)
                munchausen_addon = tau_log_pik.gather(1, actions)

                # calc munchausen reward:
                munchausen_reward = (
                        rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
                # assert munchausen_reward.shape == (self.batch_size, 1, 1)
                # Compute Q targets for current states
                Q_targets = munchausen_reward + Q_target

            # Get expected Q values from local model
            q_k, taus = self.net(states)
            Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))
            # assert Q_expected.shape == (self.batch_size, self.num_tau, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            # assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl

            if self.per:
                loss = loss * weights.to(self.net.device)

            if self.trust_regions:
                loss = self.calculate_trust_regions(loss, loss_v, states, actions, Q_expected, Q_targets)

            loss = loss.mean()

        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())

        if self.analytics:
            with torch.no_grad():
                self.analytic_object.add_loss(loss.cpu().detach())

        if self.learn_layer_norm:
            total_mean = 0
            total_std = 0
            num_layers = 0

            # Compute mean and std for each layer, and aggregate
            for layer, acts in activations.items():
                acts_tensor = torch.cat(acts, dim=0)  # Combine all batches
                mean = acts_tensor.mean()
                std = acts_tensor.std()
                total_mean += mean
                total_std += std
                num_layers += 1
                print(f"Layer: {layer}")
                print(f"Mean: {mean.item()}")
                print(f"Standard Deviation: {std.item()}")

            average_mean = total_mean / num_layers
            average_std = total_std / num_layers

            print(f"\nAverage Mean across all layers (excluding output layers): {average_mean.item()}")
            print(f"Average Standard Deviation across all layers (excluding output layers): {average_std.item()}")
            asdafs

        loss.backward()

        if self.analytics:
            with torch.no_grad():
                grad_magnitude = self.compute_gradient_magnitude()
                self.analytic_object.add_grad_mag(grad_magnitude.cpu().detach().item())

                qvals = q_k_target.mean(dim=1)
                self.analytic_object.add_qvals(qvals.cpu().detach())

                if self.grad_steps % 50 == 0:
                    _, churn_states, _, _, _, _, _ = self.memory.sample(self.batch_size)

                    churn_qvals_before = self.net.qvals(churn_states)
                    churn_actions_before = T.argmax(churn_qvals_before, dim=1).cpu()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()

        if self.dormant_exp:
            # calculate dormant ratio
            self.dormant_ratio = cal_dormant_ratio(self.net, states.detach(), self.dormant_ratio)

            # when dormant ratio is high, dormant explore rate is high
            self.entropy_tau = 0.01 + 0.04 * self.dormant_explore_rate

            if self.grad_steps % 50 == 0:
                print(f"Dormant Rate: {self.dormant_ratio}")


        if self.analytics and self.grad_steps % 50 == 0:
            with torch.no_grad():
                churn_qvals_after = self.net.qvals(churn_states)
                churn_actions_after = T.argmax(churn_qvals_after, dim=1).cpu()

                difference = torch.mean(churn_qvals_after - churn_qvals_before, dim=0)
                self.analytic_object.add_churn_dif(difference.cpu().detach())

                difference_actions = torch.sum((churn_actions_before != churn_actions_after).int(), dim=0)
                policy_churn = difference_actions / self.batch_size

                self.analytic_object.add_churn(policy_churn.cpu().detach().item())

                self.analytic_object.add_churn_actions(actions.cpu().detach())

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

    @torch.no_grad()
    def calculate_parameter_norms(self, norm_type=2):
        self.net.load_state_dict(self.net.state_dict())
        # Dictionary to store the norms
        norms = {}
        # Iterate through all named parameters
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                # Calculate the norm of the parameter
                norm = torch.norm(param, p=norm_type).item()  # .item() converts a one-element tensor to a scalar
                # Store the norm in the dictionary
                norms[name] = norm

        norms_tot = 0
        count = 0
        for key, value in norms.items():
            count += 1
            norms_tot += value

        norms_tot /= count

        return norms_tot

    def compute_gradient_magnitude(self):
        # Calculate the magnitude of the average gradient
        total_grad = 0.0
        total_params = 0

        for param in self.net.parameters():
            if param.grad is not None:
                param_grad = param.grad.data
                total_grad += torch.sum(torch.abs(param_grad))
                total_params += param_grad.numel()

        average_grad_magnitude = total_grad / total_params
        return average_grad_magnitude

    def calculate_trust_regions(self, loss, loss_v, states, actions, Q_expected, Q_targets):
        with torch.no_grad():
            if self.running_std != -999:
                current_std = torch.std(loss_v).item()
                self.running_std += current_std

                if self.tr_plus_target:
                    q_k_tgt_net, taus = self.tr_net(states)
                else:
                    q_k_tgt_net, taus = self.tgt_net(states)
                target_network_pred = q_k_tgt_net.gather(2,
                                                         actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

                # get average across quantiles
                target_pred_mean = target_network_pred.mean(dim=1)
                Q_expected_mean = Q_expected.mean(dim=1)

                # q_targets has shape (bs, 1, num_taus), so need to squeeze
                Q_targets_mean = Q_targets.squeeze().mean(dim=1).unsqueeze(1)

                #  sigma_j calculations
                sigma_j = self.running_std / self.grad_steps

                sigma_j = max(sigma_j, current_std)
                sigma_j = max(sigma_j, 0.01)

                # These all need shape checking
                outside_region = torch.abs(Q_expected_mean - target_pred_mean) > \
                                 self.tr_alpha * sigma_j

                diff_sign = torch.sign(Q_expected_mean - target_pred_mean) != \
                            torch.sign(Q_expected_mean - Q_targets_mean)

                # create mask if conditions are true
                mask = torch.logical_and(outside_region, diff_sign)
                loss[mask] = 0

                # Some Testing Code
                """
                if np.random.random() > 0.995:
                    print("Mask")
                    print(mask)

                    # mask out losses
                    loss[mask] = 0
                    print(loss)

                    print(Q_expected_mean)

                    x = input(";lol")
                """
                return loss

            else:
                self.running_std = torch.std(loss_v).detach().cpu()
                return loss


def calculate_huber_loss(td_errors, k=1.0, taus=8):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], taus, taus), "huber loss has wrong shape"
    return loss

def huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


def generate_random_sum_array(length, total):
    # Create an array of zeros
    arr = np.zeros(length, dtype=int)

    # Randomly distribute 'total' across the array
    indices = np.random.choice(np.arange(length), size=total, replace=True)
    for idx in indices:
        arr[idx] += 1  # Increment element at randomly chosen index

    # Shuffle the array to randomize the distribution
    np.random.shuffle(arr)

    return arr
