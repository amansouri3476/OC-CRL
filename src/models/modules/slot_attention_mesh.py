from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import mesh


class SlotAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs

        self.in_features = kwargs["d_in"]
        self.num_iterations = kwargs["n_sa_iters"]
        self.num_slots = kwargs["n_slots"]
        self.slot_size = kwargs["d_slot"]  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = kwargs["d_mlp"]
        self.epsilon = 1e-8

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.in_features, self.slot_size, bias=False)

        # OT approach
        self.mlp_weight_input = nn.Linear(self.cfg["d_in"], 1)
        self.mlp_weight_slots = nn.Linear(self.cfg["d_slot"], 1)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))
        # self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: torch.Tensor, num_slots=None, num_iterations=None, slots_init=None):
        # `inputs` has shape [batch_size, num_inputs, inputs_size]. example: [256, 4096, 64]
        batch_size, num_inputs, inputs_size = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        n_i = num_iterations if num_iterations is not None else self.num_iterations
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_init is not None:
            slots = slots_init.clone()
        else:
            slots_init = torch.randn((batch_size, n_s, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s  # <- this is new

        # Multiple rounds of attention.
        for _ in range(n_i):
            # NOTE: detach for approx implicit diff
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # <- this is new

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, n_s, self.slot_size))

            # attn_norm_factor = self.slot_size ** -0.5
            # attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            # attn_logits = -attn_logits
            attn_logits = torch.cdist(k, q)  # <- this is new
            # k = k / k.norm(dim=2, keepdim=True)
            # q = q / q.norm(dim=2, keepdim=True)
            # attn_logits = 1.0 - 6* torch.matmul(k, q.transpose(2, 1))

            # attn = F.softmax(attn_logits, dim=-1)
            attn_logits, p, q = mesh.minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=5)   # <- this is new
            attn, _, _ = mesh.sinkhorn(attn_logits, a, b, u=p, v=q)  # <- this is new

            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, n_s))

            # Weighted mean.
            # attn = attn + self.epsilon
            # attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, n_s, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * n_s, self.slot_size),
                slots_prev.view(batch_size * n_s, self.slot_size),
            )
            slots = slots.view(batch_size, n_s, self.slot_size)
            assert_shape(slots.size(), (batch_size, n_s, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, n_s, self.slot_size))

        slots_init = slots.clone()
        return slots, attn.permute(0,2,1), slots_init # the last argument is None
    


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"