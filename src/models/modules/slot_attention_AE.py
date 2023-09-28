import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import numpy as np

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)) # Shape: [1, resolution[0], resolution[1], 4]

def spatial_flatten(x):
    # x: [batch_size, num_channels, width, height]
    return torch.flatten(x, 2, 3)

def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = slots.reshape(-1, slots.shape[-1]).unsqueeze(1).unsqueeze(2)
    grid = slots.repeat((1, resolution[0], resolution[1], 1))
    # `grid` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    return grid

def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1]
    channels, masks = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3]).split([num_channels,1], dim=-1)
    return channels, masks


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(pl.LightningModule):
    def __init__(self, hid_dim, resolution):
        """Builds the soft position embedding layer.
        Args:
        hid_dim: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        
        # 4 comes from concat([grid, 1.0-grid]) in build_grid, because grid has x,y coordinates.
        # Then these 4 dimensional embeddings will be projected to a higher dimension, i.e. the
        # output dimension of cnn encoder so they can be added. This embedding is learnable.
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        
        # in practice, `inputs` will be the output of a cnn encoder. for instance: [batch_size, last_hid_dim, width, height]
        # self.grid will be of shape: [1, resolution[0], resolution[1], 4]
        # self.embedding(self.grid) will be of shape: [1, resolution[0], resolution[1], hid_dim] and we will permute the axes
        # so that the channel dimension aligns in both tensors.
        grid = self.embedding(self.grid.to(inputs.device)).permute(0,3,1,2)
        return inputs + grid # Shape: [batch_size, resolution[0], resolution[1], last_hid_dim]

class Encoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.encoder_layers.items()]
        )        

    def forward(self, x):
        
        # input `x` or `image` has shape: [batch_size, width, height, num_channels].
        # passing x through the encoder and adding positional embedding
        return self.layers(x)
    

class Decoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.decoder_layers.items()]
        )
                
    def forward(self, x):
        
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        return self.layers(x)
        

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_cnn = hydra.utils.instantiate(self.hparams.encoder_cnn)
        self.decoder_cnn = hydra.utils.instantiate(self.hparams.decoder_cnn)
        
        self.encoder_pos_emb = hydra.utils.instantiate(self.hparams.encoder_pos_emb)
        self.decoder_pos_emb = hydra.utils.instantiate(self.hparams.decoder_pos_emb)
        
        self.layer_norm = nn.LayerNorm(self.hparams.hid_dim)

        self.mlp = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.mlp.items()])
        
        self.slot_attention = hydra.utils.instantiate(self.hparams.slot_attention)        

        
    def forward(self, image, num_slots = None, num_iterations = None, slots_init = None):
        # `image` has shape: [batch_size, num_channels, width, height].
        
        # `num_slots` and `num_iterations` keywords let us try the model with a different number of slots and iterations
        # in each pass
        num_slots = num_slots if num_slots is not None else self.hparams.num_slots
        num_iterations = num_iterations if num_iterations is not None else self.hparams.num_iterations

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone. Shape: [batch_size, last_hid_dim, width, height]
        x = self.encoder_pos_emb(x)  # Position embedding. Shape: same as above
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set). Shape: [batch_size, last_hid_dim, width*height]
        x = x.permute(0,2,1) # Shape: [batch_size, width * height, last_hid_dim]
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set. Shape: [batch_size, width * height, last_hid_dim]

        # Slot Attention module.
        slots, attention_scores, slots_init = self.slot_attention(x, num_slots=num_slots, num_iterations=num_iterations, slots_init=slots_init)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.hparams.decoder_init_resolution)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = x.permute(0,3,1,2)

        x = self.decoder_pos_emb(x) # Shape: [batch_size*num_slots, slot_size, width_init, height_init]
        x = self.decoder_cnn(x) # Shape: [batch_size*num_slots, num_channels+1, width, height].
        x = x.permute(0,2,3,1) # Shape: [batch_size*num_slots, width, height, num_channels+1].
        recons, masks = unstack_and_split(x, image.shape[0], num_channels=int(x.shape[-1]-1))
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image. Shape: [batch_size, width, height, num_channels]
        recon_combined = recon_combined.permute(0,3,1,2) # Shape: [batch_size, num_channels, width, height]
        # `recon_combined` has shape: [batch_size, num_channels, width, height].

        return recon_combined, recons, masks, slots, attention_scores, slots_init
    