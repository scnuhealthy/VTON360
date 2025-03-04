import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import itertools
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from einops import rearrange, repeat
from omegaconf import OmegaConf

from models.unet import UNet3DConditionModel
# from models.condition_encoder import FrozenOpenCLIPImageEmbedderV2
# from animatediff.pipelines.pipeline_animation import AnimationPipeline

from models.hack_poseguider import Hack_PoseGuider as PoseGuider

from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention_multi_fp16 import ReferenceNetAttention
from models.ReferenceEncoder import ReferenceEncoder
from data.Thuman2_multi import Thuman2_Dataset, collate_fn  # here
from data.MVHumanNet_multi import MVHumanNet_Dataset
from utils import camera_para_embed, calculate_weight_matrix, register_mv_attn, update_mv_attn
logger = get_logger(__name__, log_level="INFO")

def get_parameters_without_gradients(model):
    """
    Returns a list of names of the model parameters that have no gradients.

    Args:
    model (torch.nn.Module): The model to check.
    
    Returns:
    List[str]: A list of parameter names without gradients.
    """
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name} : {param.grad}")
            no_grad_params.append(name)
    return no_grad_params


def main():
    # args = OmegaConf.load('config//train_stage1_tiktok.yaml')
    args = OmegaConf.load('config//train_tryon_multi.yaml')  # here
    print(args)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    # from accelerate import DistributedDataParallelKwargs
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
        #kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            # shutil.copy('config//train_stage1_tiktok.yaml', args.output_dir)
            shutil.copy('config//train_tryon.yaml', args.output_dir) # here
    # image_embedder = FrozenOpenCLIPImageEmbedderV2(freeze=True, model_path=args.pretrained_clip_path)
    clip_image_encoder = ReferenceEncoder(model_path=args.clip_model_path)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path, subfolder="vae")
    if args.from_scratch:
        poseguider = PoseGuider(noise_latent_channels=320)
        referencenet = ReferenceNet.from_pretrained(args.pretrained_model_path, subfolder="unet")
    else:
        poseguider = PoseGuider.from_pretrained(pretrained_model_path=args.trained_pose_guider_path)
        referencenet = ReferenceNet.from_pretrained(args.trained_referencenet_path, subfolder="referencenet")

    if args.from_scratch:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
        in_channels = 8
        unet.register_to_config(in_channels = in_channels)
        unet.conv_in = nn.Conv2d(
            in_channels, 320, kernel_size=3, padding=1
        )
        # referencenet.register_to_config(in_channels = in_channels)
        # referencenet.conv_in = nn.Conv2d(
        #     in_channels, 320, kernel_size=3, padding=1
        # )
        # encoder hidden proj
        #encoder_hid_dim = args.unet_additional_kwargs.encoder_hid_dim
        #unet.register_to_config(encoder_hid_dim=encoder_hid_dim, encoder_hid_dim_type='text_proj')
        #unet.encoder_hid_proj = nn.Linear(encoder_hid_dim, unet.cross_attention_dim)  # 下一步改referencenet的text输入
        #referencenet.register_to_config(encoder_hid_dim=encoder_hid_dim, encoder_hid_dim_type='text_proj')
        #referencenet.encoder_hid_proj = nn.Linear(encoder_hid_dim, unet.cross_attention_dim)
    else:
        unet = UNet2DConditionModel.from_pretrained(args.trained_unet_path, subfolder="unet")
        print('load pretrain!!!!!!!!!!!!!!!')

    reference_control_writer = ReferenceNetAttention(referencenet, do_classifier_free_guidance=False, mode='write', fusion_blocks=args.fusion_blocks, batch_size=args.train_batch_size ,is_image=args.image_finetune)
    reference_control_reader = ReferenceNetAttention(unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=args.fusion_blocks, batch_size=args.train_batch_size ,is_image=args.image_finetune)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    # referencenet.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    # unet.requires_grad_(True)
    for name, param in unet.named_parameters():
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                # print(name)
                param.requires_grad = True
                break

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            print('xformer_version', xformers_version)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            referencenet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                if i==0:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                elif i==1:
                    torch.save(model.state_dict(), os.path.join(output_dir, "pose.ckpt"))
                elif i==2:
                    model.save_pretrained(os.path.join(output_dir, "referencenet"))
                    pass 
                else:
                    print('!!!!!!!!!!!!')
                    pass
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                if i==2:
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                elif i==0:
                    load_model = ReferenceNet.from_pretrained(input_dir, subfolder="referencenet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                elif i==1:
                    load_model = PoseGuider.from_pretrained(os.path.join(input_dir, 'pose.ckpt'))
                    model.load_state_dict(load_model.state_dict())
                else:
                    print('!!!!!!!!!!!!')
                    pass
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    #if args.gradient_checkpointing:
    #   unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    train_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.image_finetune:
        train_params += list(filter(lambda p: p.requires_grad, poseguider.parameters())) 
        train_params += list(filter(lambda p: p.requires_grad, referencenet.parameters()))
    
    print(len(train_params))
    optimizer = optimizer_cls(
        train_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # dataset
    train_data_mvhumannet_config = args.train_data
    train_dataset = MVHumanNet_Dataset(**train_data_mvhumannet_config)
    # train_data_config = args.train_data
    # train_dataset = Thuman2_Dataset(**train_data_config)
    print('Length DataSet', len(train_dataset))

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # print("debug", args.dataloader_num_workers)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, poseguider, referencenet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, poseguider, referencenet, optimizer, train_dataloader, lr_scheduler
    )
    register_mv_attn(unet.module)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu and cast to weight_dtype
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("animate_anyone")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_step = global_step
            # resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    step = 0 
    # print(resume_step)
    while args.resume_from_checkpoint and step < resume_step:
        progress_bar.update(1)
        step +=1
        # continue

    from datetime import datetime
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        poseguider.train()
        referencenet.train()

        train_loss = 0.0
        # for step, batch in enumerate(train_dataloader):
        for step_, batch in enumerate(train_dataloader):
            #print("begin",datetime.now())
            step +=1
            # Skip steps until we reach the resumed step

            pixel_values = batch["pixel_values"].to(weight_dtype)
            pixel_values_pose = batch["pixel_values_pose"].to(weight_dtype)
            pixel_values_agnostic = batch["pixel_values_agnostic"].to(weight_dtype)
            clip_ref_front = batch["clip_ref_front"].to(weight_dtype)
            clip_ref_back = batch["clip_ref_back"].to(weight_dtype)
            pixel_values_ref_front = batch["pixel_values_ref_front"].to(weight_dtype)
            pixel_values_ref_back = batch["pixel_values_ref_back"].to(weight_dtype)
            drop_image_embeds = batch["drop_image_embeds"]
            dino_fea_front = clip_image_encoder(clip_ref_front)
            dino_fea_back = clip_image_encoder(clip_ref_back)
            camera_pose = batch["camera_parm"]

            # rearrange
            bs = pixel_values.shape[0]
            frame_length = pixel_values.shape[1]
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> (b f) c h w")
            pixel_values_agnostic = rearrange(pixel_values_agnostic, "b f c h w -> (b f) c h w")
            drop_image_embeds = rearrange(drop_image_embeds, "b f->(b f)")

            weight_matrix = calculate_weight_matrix(camera_pose.reshape(bs, frame_length, 3, 3))
            weight_matrix = weight_matrix.to(device=pixel_values.device, dtype=weight_dtype)
            # print('weight_matrix',weight_matrix)
            update_mv_attn(unet.module, weight_matrix)


            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # modified
                agnostic = vae.encode(pixel_values_agnostic).latent_dist.sample()
                agnostic = agnostic * vae.config.scaling_factor

                if not args.image_finetune:
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    agnostic = rearrange(agnostic, "(b f) c h w -> b c f h w", f=video_length)

                latents_ref_front = vae.encode(pixel_values_ref_front).latent_dist
                latents_ref_front = latents_ref_front.sample()
                latents_ref_front = latents_ref_front * vae.config.scaling_factor

                latents_ref_back = vae.encode(pixel_values_ref_back).latent_dist
                latents_ref_back = latents_ref_back.sample()
                latents_ref_back = latents_ref_back * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = bs * frame_length
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                ref_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device)
                timesteps = timesteps.long()
                ref_timesteps = ref_timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # modified
                noisy_latents = torch.cat((noisy_latents, agnostic),dim=1)
                
                # Get the pose embedding for conditioning.
                front_encoder_hidden_states = dino_fea_front
                back_encoder_hidden_states = dino_fea_back

                # camera embed,full embed and rearrange 

                # camera_embed = camera_para_embed(camera_pose, embed_dim=43)[:,:768]
                # camera_embed = camera_embed.unsqueeze(1).to(weight_dtype)

                camera_embed = []
                for b in range(bs):
                    camera_embed.append(camera_para_embed(camera_pose[b], embed_dim=43)[:,:768])
                camera_embed = torch.stack(camera_embed)
                camera_embed = rearrange(camera_embed, "b f c -> (b f) c")
                camera_embed = camera_embed.unsqueeze(1).to(device=latents.device, dtype=weight_dtype)
                
                full_encoder_hidden_states = torch.cat([front_encoder_hidden_states, back_encoder_hidden_states], dim=1)
                full_encoder_hidden_states = repeat(full_encoder_hidden_states, 'b n c -> (b f) n c', f=frame_length)
                full_encoder_hidden_states = torch.cat([full_encoder_hidden_states, camera_embed], dim=1)

                # support cfg train
                mask = drop_image_embeds > 0
                mask1 = mask.unsqueeze(1).unsqueeze(2).expand_as(full_encoder_hidden_states)
                full_encoder_hidden_states[mask1] = 0

                latents_pose = poseguider(pixel_values_pose)

                # ref_timesteps = torch.zeros_like(timesteps)
                # print(latents_ref_front.shape,ref_timesteps.shape,front_encoder_hidden_states.shape)
                referencenet(latents_ref_front, ref_timesteps, front_encoder_hidden_states)
                referencenet(latents_ref_back, ref_timesteps, back_encoder_hidden_states)
                reference_control_reader.update(reference_control_writer)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                # print(44, noisy_latents.dtype, main_encoder_hidden_states.dtype, latents_pose.dtype)
                model_pred = unet(noisy_latents, timesteps, full_encoder_hidden_states, latent_pose=latents_pose).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                # get_parameters_without_gradients(unet)
                # print('11111111111111111111111111')
                # get_parameters_without_gradients(referencenet)
                # print('22222222222222222222222222222')
                # get_parameters_without_gradients(poseguider)
                # exit()
                # for name, p in unet.named_parameters():
                #     if p.grad is None:
                #         # print('000', name)
                #         pass
                #     else:
                #         #if 'temp' in name:
                #         print(name, torch.max(p.grad))
        
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(train_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                reference_control_reader.clear()
                reference_control_writer.clear()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
            #print("end", datetime.now())

    # # Create the pipeline using the trained modules and save it.
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = accelerator.unwrap_model(unet)

    #     pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         text_encoder=accelerator.unwrap_model(text_encoder),
    #         vae=accelerator.unwrap_model(vae),
    #         unet=unet,
    #         revision=args.revision,
    #     )
    #     pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
