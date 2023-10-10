import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

class SparseDispatcher(object):

    def __init__(self, n_experts, gates):

        self._gates = gates
        self._n_experts = n_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):

        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            _nonzero_gates = self._nonzero_gates.unsqueeze(-1) if stitched.dim() == 3 else self._nonzero_gates
            stitched = stitched.mul(_nonzero_gates)
        if stitched.dim() == 2:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(1), requires_grad=True).to(stitched.device)
        else:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(-2), expert_out[-1].size(-1),
                                requires_grad=True).to(stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(stitched.device)
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):

        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SingleExpert(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = activation

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        if self.activation is not None:
            output = self.activation(output)
        return output


class MoE(nn.Module):

    def __init__(self, input_size, hidden_size, config, noisy_gating=True, reduce_factor=8):
        super(MoE, self).__init__()
        self.k = config.k
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.input_size, self.output_size = input_size, hidden_size
        # instantiate experts
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleList([SingleExpert(input_size, hidden_size) for i in range(config.n_experts)])
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        # if self.moe_level == 'token':
        #     self.row_pooler, self.col_pooler = Pooler(input_size), Pooler(config.max_seq_length)
        # elif self.moe_level == 'task':
        #     self.task_proj = nn.Linear(config.description_size,
        #                                config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)

        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):

        original_shape = list(x.shape[:-1])
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.sum(1)
            gates, load = self.noisy_top_k_gating(task_embeddings, self.training)
        elif self.moe_level == 'sentence':
            gates, load = self.noisy_top_k_gating(x.mean(-2), self.training)
        else:
            x = x.reshape(-1, self.input_size)
            gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class SMoE(nn.Module):

    def __init__(self, input_size, output_size, config, noisy_gating=True, reduce_factor=8):
        super(SMoE, self).__init__()
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        # todo sparse coding
        self.alpha = config.alpha
        self.gating = config.gating
        if self.gating == 'spc':
            rng = np.random.RandomState(seed=2)
            dictionary = rng.normal(loc=0.0, scale=1.0,
                                    size=(config.description_size // reduce_factor, self.n_experts))
            self.dictionary = nn.Parameter(torch.tensor(dictionary, dtype=torch.float32), requires_grad=False)
            self.gates = nn.Parameter(torch.Tensor(1, config.n_experts), requires_grad=False)
        else:
            self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
            self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
            self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        # instantiate experts
        self.weight = nn.Parameter(torch.Tensor(
            config.n_experts, output_size, self.input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(
            config.n_experts, output_size), requires_grad=True)
        if self.moe_level == 'token':
            self.row_pooler, self.col_pooler = Pooler(input_size), Pooler(config.max_seq_length)
        elif self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=-1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)

        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):

        loss = None
        batch_size = x.shape[0]
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.sum(1)
        if self.gating == 'spc':
            weights = torch.sum(torch.mul(self.weight, self.gates.view(-1, 1, 1)), dim=0)
            bias = torch.sum(torch.mul(self.bias, self.gates.view(-1, 1)), dim=0)
            y = F.linear(x, weights, bias)
        else:
            if self.moe_level == 'task':
                gates, load = self.noisy_top_k_gating(task_embeddings, self.training)
            else:
                gates, load = self.noisy_top_k_gating(x.mean(-2), self.training)
            # calculate importance loss
            importance = gates.view(-1, self.n_experts).sum(0)
            loss = cv_squared(importance) + cv_squared(load)
            loss *= loss_coef
            expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size, -1, 1, 1)), dim=1)
            expert_bias = torch.sum(torch.mul(self.bias, gates.view(batch_size, -1, 1)), dim=1)
            if self.moe_level == 'token':
                row_attention, col_attention = self.row_pooler(x[:, 0, :]), self.col_pooler(x[:, :, 0])
                x = x * row_attention.unsqueeze(-2) * col_attention.unsqueeze(-1)
            y = torch.einsum('bij,bkj->bik', x, expert_weights) + expert_bias.unsqueeze(1)
        return y, loss


class AMoE(nn.Module):

    def __init__(self, input_size, hidden_size, config, noisy_gating=True, reduce_factor=8, activation=None):
        super(AMoE, self).__init__()
        self.k = config.k
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.input_size, self.output_size = input_size, hidden_size
        # instantiate experts
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleList(
            [SingleExpert(input_size, hidden_size, activation) for i in range(config.n_experts)])
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        if self.moe_level == 'token':
            self.row_pooler, self.col_pooler = Pooler(input_size), Pooler(config.max_seq_length)
        elif self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)

        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):

        original_shape = list(x.shape[:-1])
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.sum(1)
            gates, load = self.noisy_top_k_gating(task_embeddings, self.training)
        elif self.moe_level == 'sentence':
            gates, load = self.noisy_top_k_gating(x.mean(-2), self.training)
        else:
            x = x.reshape(-1, self.input_size)
            gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class STRMoE(nn.Module):

    def __init__(self, input_size, output_size, config, noisy_gating=True, reduce_factor=8):
        super(STRMoE, self).__init__()
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        # todo sparse coding
        self.alpha = config.alpha
        self.gating = config.gating
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        # instantiate experts

        self.f = torch.sigmoid
        self.activation = torch.relu
        self.sparseThreshold = nn.Parameter(initialize_sInit(self.n_experts))
        self.weight = nn.Parameter(torch.Tensor(
            config.n_experts, output_size, input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(
            config.n_experts, output_size), requires_grad=True)
        if self.moe_level == 'token':
            self.row_pooler, self.col_pooler = Pooler(input_size), Pooler(config.max_seq_length)
        elif self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch, m = clean_values.size(0), noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def str_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = self.softmax(logits)
        gates = sparseFunction(gates, self.sparseThreshold, self.activation, self.f)
        if self.noisy_gating and self.k < self.n_experts and train:
            top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        batch_size = x.shape[0]
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.sum(1)
        if self.moe_level == 'task':
            gates, load = self.str_gating(task_embeddings, self.training)
        else:
            gates, load = self.str_gating(x.mean(-2), self.training)
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size, -1, 1, 1)), dim=1)
        expert_bias = torch.sum(torch.mul(self.bias, gates.view(batch_size, -1, 1)), dim=1)
        if self.moe_level == 'token':
            row_attention, col_attention = self.row_pooler(x[:, 0, :]), self.col_pooler(x[:, :, 0])
            x = x * row_attention.unsqueeze(-2) * col_attention.unsqueeze(-1)
        y = torch.einsum('bij,bkj->bik', x, expert_weights) + expert_bias.unsqueeze(1)
        return y, loss



class PadMoE(nn.Module):
    def __init__(self, input_size, output_size, config, noisy_gating=True):
        super(PadMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.n_experts, self.k = config.n_experts, config.k
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.config = config
        self.input_size, self.output_size = input_size, output_size
        self.phi = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # todo change it temporally
        # self.experts = nn.ModuleList([SingleExpert_Pad(input_size, output_size, config) for i in range(config.n_experts)])
        self.experts = nn.ModuleList([SingleExpert_Structured(input_size, output_size, config) for i in range(config.n_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, config.n_experts), requires_grad=True)
        # self.weight = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.static_fc = nn.Linear(input_size, output_size)
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mask', torch.ones(output_size, input_size))
        assert(self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)
        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        original_shape = list(x.shape[:-1])
        x = x.reshape(-1, self.input_size)
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # static_outputs = self.static_fc(x)
        dynamic_outputs = [self.experts[i](expert_inputs[i], self.static_fc.weight) for i in range(self.n_experts)]
        # y = dispatcher.combine(dynamic_outputs) + static_outputs
        y = dispatcher.combine(dynamic_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class SingleExpert_Pad(nn.Module):

    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(input_size, hidden_size)

    def forward(self, hidden_states, weight, mask, phi):
        phi = 2 * torch.sigmoid(phi)
        # expert_weight = self.get_sparse_weights(self.dense.weight)
        if (not self.config) or not (self.config.Lambda) or self.config.Lambda == 'both':
            hidden_states = F.linear(hidden_states,
                                     (self.dense.weight * mask * phi + weight * (1 - mask) * (2 - phi)),
                                        self.dense.bias)
        elif self.config.Lambda == 's':
            hidden_states = F.linear(hidden_states,
                                     (self.dense.weight * mask + weight * (1 - mask) * (2 - phi)),
                                        self.dense.bias)
        elif self.config.Lambda == 'd':
            hidden_states = F.linear(hidden_states,
                                     (self.dense.weight * mask * phi + weight * (1 - mask)),
                                        self.dense.bias)
        elif self.config.Lambda == 'p':
            hidden_states = F.linear(hidden_states, self.dense.weight * mask, self.dense.bias)
        else:
            hidden_states = F.linear(hidden_states, phi * self.dense.weight, self.dense.bias)
        return hidden_states


class PadMoE_Structured(nn.Module):
    def __init__(self, input_size, output_size, config, noisy_gating=True):
        super(PadMoE_Structured, self).__init__()
        self.noisy_gating = noisy_gating
        self.n_experts, self.k = config.n_experts, config.k
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.config = config
        self.input_size, output_size = input_size, output_size // 2
        self.output_size = output_size
        self.phi = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.experts = nn.ModuleList([SingleExpert_Structured(input_size, output_size, config) for i in range(config.n_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, config.n_experts), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(output_size, input_size), requires_grad=True)
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mask', torch.ones(output_size, input_size))
        assert(self.k <= config.n_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        normal = Normal(self.mean, self.std)
        threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)
        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = _gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        original_shape = list(x.shape[:-1])
        x = x.reshape(-1, self.input_size)
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # static_outputs = self.static_fc(x)
        static_outputs = F.linear(x, self.weight)
        dynamic_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(dynamic_outputs) + static_outputs
        y = torch.cat([dispatcher.combine(dynamic_outputs), static_outputs], dim=-1)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class SingleExpert_Structured(nn.Module):

    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(input_size, hidden_size)

    def forward(self, hidden_states, weight):
        # return self.dense(hidden_states)
        return F.linear(hidden_states, self.dense.weight + weight, self.dense.bias)


def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))


def initialize_sInit(n_experts):
    # if parser_args.sInit_type == "constant":
    return math.log(1 / (n_experts - 1))*torch.ones([1, 1])


def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0]).to(x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def _gates_to_load(gates):
    return (gates > 0).sum(0)


class Pooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Sigmoid()
        with torch.no_grad():
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = 2 * self.activation(pooled_output)
        return pooled_output

