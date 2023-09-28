import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Slot Attention module."""
    
    def __init__(self, num_iterations = 3, num_slots = 7, slot_size = 32, mlp_hidden_size = 128, epsilon = 1e-8, **kwargs):
        """Builds the Slot Attention module.
        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        
        self.scale = slot_size ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_size))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, self.slot_size))

        # Linear maps for the attention module.
        # TODO: for now, we have forced the mlp_hidden_size of slot attention to be the same as the feature size
        # from the encoder. They can be different, and thus the input_dim of the following fc layers should be
        # the feature size, and mlp_hidden_size should be used for the projection layer at the end of the for loop.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.mlp_hidden_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.mlp_hidden_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        hidden_dim = max(self.slot_size, self.mlp_hidden_size)
        self.mlp = nn.Sequential(*[nn.Linear(self.slot_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.slot_size)])
       
        self.norm_inputs  = nn.LayerNorm(hidden_dim)
        self.norm_slots  = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)


    def forward(self, inputs, num_slots = None, num_iterations = None, slots_init = None):
        
        # `num_slots` keyword let's you try the model with a different number of slots in each pass
        # `num_iterations` keyword let's you try the model with a different number of attention iterations in each pass
        
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    
        batch_size, num_inputs, inputs_size = inputs.size()
        n_s = num_slots if num_slots is not None else self.num_slots
        n_i = num_iterations if num_iterations is not None else self.num_iterations
        
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].

        mu = self.slots_mu.expand(batch_size, n_s, -1)
        sigma = torch.exp(self.slots_log_sigma.expand(batch_size, n_s, -1))
        slots = torch.normal(mu, sigma) if slots_init is None else slots_init

        # for _ in range(n_i):
            
        #     # `slots` shape: [batch_size, num_slots, slot_size]
        #     slots_prev = slots
        #     slots = self.norm_slots(slots)
            
        #     # Attention.
        #     q = self.project_q(slots) # Shape: [batch_size, num_slots, slot_size].

        #     # einsum is computing the dot product between queries coming from slots (num_slots query of dim
        #     # slot_size), and keys coming from the input. each element in the input batch is composed of a
        #     # sequence of embeddings of size input_size (output of cnn_encoder), which is then projected via
        #     # self.project_k, thus having a shape of [batch_size, num_inputs, slot_size]
        #     dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # Shape: [batch_size, num_slots, num_inputs].
            
        #     # note that softmax is over dim=1, i.e. num_slots, so slots have to compete over input patches.
        #     attn = dots.softmax(dim=1) + self.epsilon # Shape: [batch_size, num_slots, num_inputs].

        #     # Weigted mean.
        #     attn = attn / attn.sum(dim=-1, keepdim=True) / n_s # Shape: [batch_size, num_slots, num_inputs].
        #     updates = torch.einsum('bjd,bij->bid', v, attn) # Shape: [batch_size, num_slots, slot_size].
            
        #     # Slot update.
        #     slots = self.gru(
        #         updates.reshape(-1, self.slot_size),
        #         slots_prev.reshape(-1, self.slot_size)
        #     )

        #     slots = slots.reshape(batch_size, -1, self.slot_size)
        #     slots = slots + self.mlp(self.norm_mlp(slots)) # residual mlp


        # Multiple rounds of attention.
        for m in range(n_i+1):
            
            # `slots` shape: [batch_size, num_slots, slot_size]
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots) # Shape: [batch_size, num_slots, slot_size].

            # einsum is computing the dot product between queries coming from slots (num_slots query of dim
            # slot_size), and keys coming from the input. each element in the input batch is composed of a
            # sequence of embeddings of size input_size (output of cnn_encoder), which is then projected via
            # self.project_k, thus having a shape of [batch_size, num_inputs, slot_size]
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # Shape: [batch_size, num_slots, num_inputs].
            
            # note that softmax is over dim=1, i.e. num_slots, so slots have to compete over input patches.
            attn = dots.softmax(dim=1) + self.epsilon # Shape: [batch_size, num_slots, num_inputs].

            # Weigted mean.
            attn = attn / attn.sum(dim=-1, keepdim=True) # Shape: [batch_size, num_slots, num_inputs].
            updates = torch.einsum('bjd,bij->bid', v, attn) # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(
                updates.reshape(-1, self.slot_size),
                slots_prev.reshape(-1, self.slot_size)
            )

            slots = slots.reshape(batch_size, -1, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots)) # residual mlp

            # truncate the backprop
            if m == n_i-1:
                slots = slots.detach()
            
        slots_init = slots.clone()        

        return slots, attn, slots_init
    