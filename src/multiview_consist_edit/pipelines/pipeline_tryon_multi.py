# Copyright 2023 The InstructPix2Pix Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
from models.hack_poseguider import Hack_PoseGuider as PoseGuider
from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention_multi_fp16 import ReferenceNetAttention
from models.ReferenceEncoder import ReferenceEncoder
from data.dataset import TikTok, collate_fn, UBC_Fashion
from utils import camera_para_embed, calculate_weight_matrix, register_mv_attn, update_mv_attn
from einops import rearrange, repeat

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

class TryOnPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        referencenet: ReferenceNet,
        pose_guider: PoseGuider,
        scheduler: KarrasDiffusionSchedulers,
        requires_safety_checker: bool = True,
        fusion_blocks = 'full'
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            referencenet=referencenet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.fusion_blocks = fusion_blocks
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # self._execution_device = torch.device('cuda')
    @torch.no_grad()
    def __call__(
        self,
        front_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        back_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        pose_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        agnostic_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        camera_pose: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        front_dino_fea: Optional[torch.FloatTensor] = None, # b n d  b 50 768
        back_dino_fea: Optional[torch.FloatTensor] = None, # b n d  b 50 768
    ):

        # 0. Check inputs
        # self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        batch_size = pose_image.shape[0]
        frame_length = pose_image.shape[1]
        height, width = front_image.shape[-2:]

        # reshape
        pose_image = rearrange(pose_image, "b f c h w -> (b f) c h w")
        agnostic_image = rearrange(agnostic_image, "b f c h w -> (b f) c h w")

        device = front_image.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        reference_control_writer = ReferenceNetAttention(self.referencenet, do_classifier_free_guidance=do_classifier_free_guidance, mode='write', fusion_blocks=self.fusion_blocks, batch_size=batch_size, is_image=True)
        reference_control_reader = ReferenceNetAttention(self.unet, do_classifier_free_guidance=do_classifier_free_guidance, mode='read', fusion_blocks=self.fusion_blocks, batch_size=batch_size, is_image=True)

        # camera
        weight_matrix = calculate_weight_matrix(camera_pose.reshape(batch_size, frame_length, 3, 3))
        # weight_matrix = torch.ones_like(weight_matrix)
        weight_matrix = weight_matrix.to(device=device, dtype=torch.float16)
        # print('weight_matrix',weight_matrix)
        register_mv_attn(self.unet)  # here use mv attn
        update_mv_attn(self.unet, weight_matrix)
        # exit()

        # 2. image embedding
        front_image_embeddings = front_dino_fea.to(device=device, dtype=torch.float16)
        back_image_embeddings = back_dino_fea.to(device=device, dtype=torch.float16)

        camera_embed = []
        for b in range(batch_size):
            camera_embed.append(camera_para_embed(camera_pose[b], embed_dim=43)[:,:768])
        camera_embed = torch.stack(camera_embed)
        camera_embed = rearrange(camera_embed, "b f c -> (b f) c")
        camera_embed = camera_embed.unsqueeze(1).to(device=device, dtype=torch.float16)
        
        full_image_embeddings = torch.cat([front_image_embeddings, back_image_embeddings], dim=1)
        full_image_embeddings = repeat(full_image_embeddings, 'b n c -> (b f) n c', f=frame_length)
        full_image_embeddings = torch.cat([full_image_embeddings, camera_embed], dim=1)

        # print('111', camera_pose.shape)
        # camera_embed = camera_para_embed(camera_pose, embed_dim=43)[:,:768]
        # print('222', camera_embed.shape)
        # camera_embed = camera_embed.unsqueeze(1).to(device=device, dtype=torch.float16)
        # print('333',camera_embed.shape)
        # full_image_embeddings = torch.cat([front_image_embeddings, back_image_embeddings, camera_embed], dim=1)
        # print('444',full_image_embeddings.shape)
        # full_image_embeddings = repeat(full_image_embeddings, 'b n c -> (b f) n c', f=frame_length)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        ref_front_latents = self.prepare_image_latents(
            front_image,
            num_images_per_prompt,
            front_image_embeddings.dtype,
            device,
            generator,
        )

        ref_back_latents = self.prepare_image_latents(
            back_image,
            num_images_per_prompt,
            front_image_embeddings.dtype,
            device,
            generator,
        )

        # 5.5 Prepare agnostic latents
        agnostic_latents = self.prepare_image_latents(
            agnostic_image,
            num_images_per_prompt,
            front_image_embeddings.dtype,
            device,
            generator,
        )

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * frame_length,
            num_channels_latents,
            height,
            width,
            front_image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        # check here
        # print('check', pose_image.shape, agnostic_image.shape, full_image_embeddings.shape, front_image_embeddings.shape, latents.shape)

        latents = torch.cat((latents,agnostic_latents),dim=1)
        if do_classifier_free_guidance:
            # referencenet
            # negative_image_embeddings1 = torch.zeros_like(front_image_embeddings)
            # front_image_embeddings = torch.cat([negative_image_embeddings1, front_image_embeddings])
            # back_image_embeddings = torch.cat([negative_image_embeddings1, back_image_embeddings])

            # ref_front_latents =  torch.cat([ref_front_latents]*2)
            # ref_back_latents =  torch.cat([ref_back_latents]*2)
            #unet
            negative_image_embeddings2 = torch.zeros_like(full_image_embeddings)
            full_image_embeddings = torch.cat([negative_image_embeddings2, full_image_embeddings])
        
        # 7. prepare pose
        latent_pose = self.pose_guider(pose_image)

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_pose_input = torch.cat([latent_pose] * 2 ) if do_classifier_free_guidance else latent_pose 

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i == 0:
                    # just once
                    self.referencenet.to(device=latents.device)
                    self.referencenet(
                        ref_front_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=front_image_embeddings,
                        return_dict=False,
                    )
                    self.referencenet(
                        ref_back_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=back_image_embeddings,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)
                    # Put it on the cpu when it is not needed to save memory.
                    self.referencenet.to('cpu')   

                # predict the noise residual
                noise_pred = self.unet(scaled_latent_model_input, 
                    t, 
                    encoder_hidden_states=full_image_embeddings,
                    latent_pose=latent_pose_input
                    ).sample

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents_0 = self.scheduler.step(noise_pred, t, latents[:,0:4,:,:], **extra_step_kwargs).prev_sample
                latents = torch.cat((latents_0,agnostic_latents),dim=1)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                        
        # 10. Post-processing
        image = self.decode_latents(latents[:,0:4,:,:])

        # # 11. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 12. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_model_cpu_offload
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(
        self, prompt, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self, source_image, num_images_per_prompt, dtype, device, generator=None
    ):

        source_image = source_image.to(device=device, dtype=dtype)
        ref_image_latents = self.vae.encode(source_image).latent_dist.mean
        ref_image_latents = ref_image_latents * self.vae.config.scaling_factor
        #ref_image_latents = torch.cat([ref_image_latents], dim=0)
        return ref_image_latents
