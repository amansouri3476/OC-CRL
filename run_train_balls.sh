#!/bin/bash

#SBATCH --account=user_account
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_out/slot-attention-%j.out
#SBATCH --error=./slurm_err/slot-attention-%j.err

source /home/usr/env/bin/activate

export WANDB_API_KEY=YOUR_API_KEY
wandb login

export WANDB_MODE=offline

# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# -------------------------------- INERTIA BALLS RUNS --------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #

# --------------------------------------------------- #
# -------------------- n_balls=2 -------------------- #

# ---------- SA ---------- #
# x,y (square, random colour)
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=default datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y"] datamodule.dataset.z_dim=2 datamodule.n_balls=2 datamodule.batch_size=128 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping trainer.max_epochs=2000 ckpt_path=null

# ---------- ConvNet Injective ---------- #
# Note: We can't have the colour as a target property because we need something to make the observation function injective 
# x,y (square, random colour)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y"] model.z_dim=2 model.disentangle_z_dim=2 datamodule.dataset.z_dim=2 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="c" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xy"] trainer.max_epochs=50 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c (square)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c"] model.z_dim=3 model.disentangle_z_dim=3 datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="s" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xyc"] trainer.max_epochs=100 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,s (fixed colour)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","s"] model.z_dim=3 model.disentangle_z_dim=3 datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="c" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xys"] trainer.max_epochs=100 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c,s
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s"] model.z_dim=4 model.disentangle_z_dim=4 datamodule.dataset.z_dim=4 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="l" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xycs"] trainer.max_epochs=200 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c,l,p
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","l","p"] model.z_dim=5 model.disentangle_z_dim=5 datamodule.dataset.z_dim=5 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="s" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xyclp"] trainer.max_epochs=500 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# ---------- ConvNet Non-injective ---------- #
# x,y (square, random colour)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y"] model.z_dim=2 model.disentangle_z_dim=2 datamodule.dataset.z_dim=2 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=False datamodule.dataset.injective_property="c" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","non-injective","xy"] trainer.max_epochs=50 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c (square)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c"] model.z_dim=3 model.disentangle_z_dim=3 datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=False datamodule.dataset.injective_property="s" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","non-injective","xyc"] trainer.max_epochs=100 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,s (fixed colour)
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","s"] model.z_dim=3 model.disentangle_z_dim=3 datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=False datamodule.dataset.injective_property="c" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","non-injective","xys"] trainer.max_epochs=100 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c,s
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s"] model.z_dim=4 model.disentangle_z_dim=4 datamodule.dataset.z_dim=4 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=False datamodule.dataset.injective_property="l" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","non-injective","xycs"] trainer.max_epochs=200 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"

# x,y,c,l,p
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","l","p"] model.z_dim=5 model.disentangle_z_dim=5 datamodule.dataset.z_dim=5 datamodule.n_balls=2 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=False datamodule.dataset.injective_property="s" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","non-injective","xyclp"] trainer.max_epochs=500 ckpt_path=null seed=1234,12567,86523 --multirun
# ckpt: "???"


# ---------- SA-Mesh ---------- #
# x,y (square, random colour)
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y"] datamodule.dataset.z_dim=2 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xy"] trainer.max_epochs=2000 ckpt_path=null

# x,y,c (square)
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c"] datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xyc"] trainer.max_epochs=2000 ckpt_path=null

# x,y,s (fixed colour)
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","s"] datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xys"] trainer.max_epochs=2000 ckpt_path=null

# x,y,c,s
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s"] datamodule.dataset.z_dim=4 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xycs"] trainer.max_epochs=2000 ckpt_path=null

# x,y,c,l,p
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","l","p"] datamodule.dataset.z_dim=5 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xyclp"] trainer.max_epochs=2000 ckpt_path=null

# x,y,c,s,l,p
# python3 run_training.py model=inertia_balls_slot_attention_ae model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s","l","p"] datamodule.dataset.z_dim=6 datamodule.n_balls=2 datamodule.batch_size=64 model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 callbacks=vanilla_slot_attention ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["SA-Mesh","xycslp"] trainer.max_epochs=2000 ckpt_path=null

# ---------- Disentanglement + SA-Mesh ---------- #
# x,y (square, random colour)
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y"] datamodule.dataset.z_dim=2 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=2 model.disentangle_z_dim=2 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xy"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# ckpt:

# x,y,c (square)
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c"] datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=3 model.disentangle_z_dim=3 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xyc"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# ckpt:

# x,y,s (fixed colour)
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","s"] datamodule.dataset.z_dim=3 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=3 model.disentangle_z_dim=3 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xys"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# ckpt:

# x,y,c,s
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s"] datamodule.dataset.z_dim=4 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=4 model.disentangle_z_dim=4 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xycs"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s"] datamodule.dataset.z_dim=4 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=4 model.disentangle_z_dim=4 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xycs"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=2793
# ckpt:

# x,y,c,l,p
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","l","p"] datamodule.dataset.z_dim=5 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=5 model.disentangle_z_dim=5 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xyclp"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# ckpt:

# x,y,c,s,l,p
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.dataset.properties_list=["x","y","c","s","l","p"] datamodule.dataset.z_dim=6 datamodule.n_balls=2 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=6 model.disentangle_z_dim=6 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=100.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xycslp"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun
# ckpt:

# --------------------# --------------------
# -------------------- n_balls= 3,4
# --------------------# --------------------

# repeat the above experiments by datamodule.n_balls=3

# ----------------------------------------
# -------------------- Data Efficiency Experiments --------------------
# ----------------------------------------

# ---------- ConvNet Injective ---------- #
# x,y,c,s
# python3 run_training.py model=inertia_balls_cnn_encoder datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.num_samples.train=100,200,500 datamodule.dataset.properties_list=["x","y","c","s"] model.z_dim=4 model.disentangle_z_dim=4 datamodule.dataset.z_dim=4 datamodule.n_balls=4 datamodule.dataset.output_sparse_offsets=False datamodule.dataset.injective=True datamodule.dataset.injective_property="l" model.known_mechanism=False datamodule.batch_size=64 model/optimizer=adamw model.optimizer.lr=0.0002 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["cnn","injective","xycs"] trainer.max_epochs=200 ckpt_path=null seed=1234,12567,86523 --multirun

# ---------- Disentanglement + SA-MESH ---------- #
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model/encoder/slot_attention=mesh datamodule=inertia_balls datamodule/dataset=all_p_sparse_offset datamodule.sparsity_degree=1 datamodule.num_samples.train=1000,2000,5000,10000 datamodule.dataset.properties_list=["x","y","c","s"] datamodule.dataset.z_dim=4 datamodule.n_balls=4 datamodule.batch_size=64 model.visualization_method="2D" model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.z_dim=4 model.disentangle_z_dim=4 model.latent_matching="argmin" model.ball_matching=True model.double_matching=False model.use_all_balls_mcc=True model.known_action=True model.rm_background_in_matching=True model.known_mechanism=False model.pair_recons=True model.w_recons_loss=100.0 model.w_latent_loss=10.0 model.wait_steps=0 model.linear_steps=1 model/optimizer=adamw model.optimizer.lr=0.0002 model.encoder.slot_size=64 model.encoder_freeze=False model.additional_logger.logging_interval=20 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["disentanglement","xycs"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="path/to/ckpt.ckpt" seed=1234,12567,86523 --multirun

deactivate
module purge
