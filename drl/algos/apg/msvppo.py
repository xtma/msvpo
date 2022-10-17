import torch
from drl.algos.apg.base import AveragePolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import valid_mean
LossInputs = namedarraytuple("LossInputs", ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])


class MSVPPO(AveragePolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
        self,
        longrun=True,
        learning_rate=0.001,
        lr_eta=0.1,  # learning rate of average performance
        rm_vbias_coeff=0.1,  # remove value bias
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        OptimCls=torch.optim.Adam,
        optim_kwargs=None,
        clip_grad_norm=1.,
        initial_optim_state_dict=None,
        discount=1.,
        gae_lambda=0.9,
        minibatches=4,
        epochs=4,
        ratio_clip=0.1,
        ratio_max=2.,
        advantage_max=10.,
        linear_lr_schedule=True,
        normalize_advantage=False,
        bootstrap_timelimit=True,
        reward_norm=True,
        msv_coef_up=1.,
        msv_coef_down=1.,
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.eta = None  # initial estimation of objective
        self.mean = None  # initial estimation of mean reward
        self.mean_up = None  # initial estimation of mean upside eta
        self.mean_down = None  # initial estimation of mean downside eta
        self.sv_up = None  # initial estimation of mean upside sv
        self.sv_down = None  # initial estimation of mean downside sv
        self.value_bias = None  # initial estimation of average value bias
        if not self.longrun:  # don't consider the long-run tricks
            self.eta = 0
            self.mean = 0
            self.mean_up = 0
            self.mean_down = 0
            self.sv_up = 0
            self.sv_down = 0
            self.value_bias = 0
            self.lr_eta = 0
            self.rm_vbias_coeff = 0

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                                  lr_lambda=lambda itr:
                                                                  (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def update_eta(self, samples):
        """
        Update the estimated average performance with new samples.
        """
        reward, value = (samples.env.reward, samples.agent.agent_info.value)
        rew_mean, value_mean = reward.mean().item(), value.mean().item()
        rew_mean_up = torch.clamp(reward - rew_mean, min=0).mean().item()
        rew_mean_down = torch.clamp(reward - rew_mean, max=0).mean().item()
        rew_sv_up = torch.clamp(reward - rew_mean, min=0).pow(2).mean().item()
        rew_sv_down = torch.clamp(reward - rew_mean, max=0).pow(2).mean().item()
        if self.mean is None:
            self.mean = rew_mean
            self.mean_up = rew_mean_up
            self.mean_down = rew_mean_down
            self.sv_up = rew_sv_up
            self.sv_down = rew_sv_down
            self.value_bias = value_mean
        else:
            self.mean = (1 - self.lr_eta) * self.mean + self.lr_eta * rew_mean
            self.mean_up = (1 - self.lr_eta) * self.mean_up + self.lr_eta * rew_mean_up
            self.mean_down = (1 - self.lr_eta) * self.mean_down + self.lr_eta * rew_mean_down
            self.sv_up = (1 - self.lr_eta) * self.sv_up + self.lr_eta * rew_sv_up
            self.sv_down = (1 - self.lr_eta) * self.sv_down + self.lr_eta * rew_sv_down
            self.value_bias = (1 - self.lr_eta) * self.value_bias + self.lr_eta * value_mean

        self.eta = ((1 - 2 * self.msv_coef_up * self.mean_up + 2 * self.msv_coef_down * self.mean_down) * self.mean +
                    self.msv_coef_up * self.sv_up - self.msv_coef_down * self.sv_down)

        return rew_mean, value_mean

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        rew_mean, valueBias = self.update_eta(samples)
        reward = (
            (1 - 2 * self.msv_coef_up * self.mean_up + 2 * self.msv_coef_down * self.mean_down) * samples.env.reward +
            self.msv_coef_up * torch.clamp(samples.env.reward - self.mean, min=0).pow(2) -
            self.msv_coef_down * torch.clamp(samples.env.reward - self.mean, max=0).pow(2))  # replace the origin reward
        return_, advantage, valid = self.process_returns(reward, samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        # if (valid == 0).any():
        #     import pdb
        #     pdb.set_trace()
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                # if any([v.isnan().any() for k, v in self.agent.state_dict().items()]):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, value_loss, entropy, perplexity = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                opt_info.pLoss.append(pi_loss.item())
                opt_info.vLoss.append(value_loss.item())
                opt_info.gradNorm.append(grad_norm.item())
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        opt_info.eta.append(rew_mean)
        opt_info.valueBias.append(valueBias)
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info, init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        advantage = advantage.clamp(-self.advantage_max, self.advantage_max)  # keep adv safe
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        ratio = ratio.clamp(max=self.ratio_max)  # keep ratio safe
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_)**2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss
        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, pi_loss, value_loss, entropy, perplexity
