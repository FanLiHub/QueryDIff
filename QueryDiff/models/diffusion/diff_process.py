import logging
import os
from typing import List, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers import DDPMScheduler
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from PIL import Image
import torch.nn as nn
import random
from .diffusion_component import DiffImgPipeline, DiffImgOutput
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    # LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from .tools import init_block_func, set_timestep, collect_feats

def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
        # "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method


def resize_max_res(
        img: torch.Tensor,
        max_edge_resolution: int,
        resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def generate_seed_sequence(
        initial_seed: int,
        length: int,
        min_val=-0x8000_0000_0000_0000,
        max_val=0xFFFF_FFFF_FFFF_FFFF,
):
    if initial_seed is None:
        logging.warning("initial_seed is None, reproducibility is not guaranteed")
    random.seed(initial_seed)

    seed_sequence = []

    for _ in range(length):
        seed = random.randint(min_val, max_val)

        seed_sequence.append(seed)

    return seed_sequence


class DiffusionTrainer_(nn.Module):

    rgb_latent_scale_factor = 0.18215

    # @configurable
    def __init__(
            self,
            model_id,
            layer_flag,
            mode,
            layer_indexes,
            timestep_list,
            guidance_scale,
            max_iter,
            num_inference_steps,
            data_type,
            features_flag,
            hook_flag,
            multi_res_noise_flag,
            strength,
            annealed,
            downscale_strategy,
            channel_modific,
            channel_num,
            seed,
            low_resources,
            q_ada,
            query_num_layers,
            query_embed_dims,
            query_patch_sizes,
    ):
        super().__init__()
        self.seed: Union[int, None] = seed  # used to generate seed sequence, set to `None` to train w/o seeding
        self.timestep_list = timestep_list
        self.guidance_scale = guidance_scale
        self.strength = strength,
        self.annealed = annealed,
        self.downscale_strategy = downscale_strategy,
        self.low_resources = low_resources
        self.num_inference_steps = num_inference_steps

        if data_type == 'bf':
            self.dtype = torch.bfloat16
        elif data_type == 'fp16':
            self.dtype = torch.float16
        elif data_type == 'fp32':
            self.dtype = torch.float32

        self.diff = DiffImgPipeline(model_id, self.dtype)
        # self.scheduler=self.diff.pipe.scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            # prediction_type="v_prediction",
        )
        # Adapt input layers
        if channel_modific == True and channel_num != self.diff.unet.config["in_channels"]:
            self._replace_unet_conv_in()
        # Encode empty text prompt
        self.diff.encode_empty_text()
        # self.empty_text_embed = self.empty_text_embed.detach().clone()

        # Training noise scheduler
        ## Can be replaced by self.scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(model_id, "scheduler")
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
                self.prediction_type == self.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Multi-resolution noise
        self.apply_multi_res_noise = multi_res_noise_flag
        if self.apply_multi_res_noise:
            self.mr_noise_strength = strength
            self.annealed_mr_noise = annealed
            self.mr_noise_downscale_strategy = downscale_strategy

        # Internal variables
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
        self.max_iter = max_iter

        self.hook_flag = hook_flag
        self.features_flag = features_flag
        if self.features_flag == True:
            ## collect features
            self.layer_flag = layer_flag  ## resnets attentions
            self.mode = mode  ## up_blocks down_blocks
            self.layer_indexes = layer_indexes
            if self.hook_flag == True:
                ## hook for features
                self.layers = []
                for l in self.layer_indexes:
                    if self.mode == 'up_blocks':
                        if self.layer_flag == 'resnets':
                            self.diff.unet.up_blocks[l[0]].resnets[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                        elif self.layer_flag == 'attentions':
                            # self.diff.unet.up_blocks[l[0]].attentions[l[1]].register_forward_hook(
                            #     lambda m, _, o: self.layers.append(o))
                            self.diff.unet.up_blocks[l[0]].attentions[l[1]].transformer_blocks[0].attn2.to_out[
                                1].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                    elif self.mode == 'down_blocks':
                        if self.layer_flag == 'resnets':
                            self.diff.unet.down_blocks[l[0]].resnets[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                        elif self.layer_flag == 'attentions':
                            # self.diff.unet.down_blocks[l[0]].attentions[l[1]].register_forward_hook(
                            #     lambda m, _, o: self.layers.append(o))
                            self.diff.unet.down_blocks[l[0]].attentions[l[1]].transformer_blocks[0].attn2.to_out[
                                1].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                    elif self.mode == 'mid_block':
                        if self.layer_flag == 'resnets':
                            self.diff.unet.mid_block.resnets[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                        elif self.layer_flag == 'attentions':
                            # self.diff.unet.mid_block.attentions[l[1]].register_forward_hook(
                            #     lambda m, _, o: self.layers.append(o))
                            self.diff.unet.mid_block.attentions[l[1]].transformer_blocks[0].attn2.to_out[
                                1].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
            else:
                ## rewrite forward function
                init_block_func(self.diff.unet, 'up', save_hidden=True, reset=True, idxs=self.layer_indexes,
                                save_timestep=[0], flag_layer='resnet')

            if q_ada:
                init_block_func(self.diff.unet, None, save_hidden=True, reset=True, idxs=None,
                                save_timestep=[0], flag_layer='attention_module_query', adapter_=self.query_ada)



        # # frozen params
        # for name, params in self.diff.unet.named_parameters():
        #     params.requires_grad = True
        #     # if "attentions" in name:
        #     #     params.requires_grad = True if "to_q" in name or "to_v" in name else False
        #     # if "attentions" in name:
        #     #     params.requires_grad = True
        #     if "up_blocks.3.attentions.2" in name:
        #         params.requires_grad = False
        #     if "up_blocks.3.resnets.2.conv_shortcut" in name:
        #         params.requires_grad = False
        #     if "conv_norm_out" in name:
        #         params.requires_grad = False
        #     if "conv_out" in name:
        #         params.requires_grad = False
        #     # if "mid_block" in name:
        #     #     params.requires_grad = False
        #     # if "up_blocks" in name:
        #     #     params.requires_grad = False


    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.diff.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.diff.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.diff.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.diff.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.diff.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def feature_extraction(
            self,
            rgb_in: torch.Tensor,
            timestep_list: int,
            add_noise_flag=False,
            prompt: str = None,
    ) -> torch.Tensor:

        device = self.diff.text_encoder.device
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        # globally consistent random generators
        if self.seed is not None:
            local_seed = self._get_next_seed()
            rand_num_generator = torch.Generator('cuda')
            rand_num_generator.manual_seed(local_seed)
        else:
            rand_num_generator = None

        rgb, batch_size = self.diff.img_process_(rgb_in)
        # Encode image
        rgb_latent = self.diff.norm_img_to_latent(rgb)

        if prompt == None:
            # Batched empty text embedding
            if self.empty_text_embed is None:
                self.diff.encode_empty_text()
            text_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, 2, 1024]
        else:
            negative_text_embed, text_embed = self.diff.get_text_embedding_(prompt, self.diff.text_encoder.device)
            text_embed = torch.cat([negative_text_embed, text_embed])

        x_0_list = []
        # Denoising list
        for t in timestep_list:
            if self.hook_flag == False:
                set_timestep(self.diff.unet, self.layer_indexes, t)

            # Sample a random timestep for each image
            t = torch.tensor(t).to(device=self.diff.unet.device).long()
            if add_noise_flag == True:
                ## add noise for latent
                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (t / self.scheduler_timesteps)
                    noise = self.multi_res_noise_like(
                        rgb_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        rgb_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    rgb_latent, noise, t
                )  # [B, 4, h, w]
                unet_input = noisy_latents
            else:
                unet_input = rgb_latent

            # predict the noise residual
            noise_pred = self.predict_(unet_input, t, text_embed, prompt)

            ouput_ = self.scheduler.step(
                noise_pred, t, unet_input, generator=None
            )
            x_0_list.append(ouput_.pred_original_sample)

        if self.hook_flag == False:
            feats = collect_feats(self.diff.unet, 'up', idxs=self.layer_indexes, flag_layer='resnet')
            return feats
        return x_0_list

    def img_denoise(
            self,
            rgb_in: torch.Tensor,
            timestep_list: int,
            add_noise_flag=False,
            prompt: str = None,
    ) -> torch.Tensor:

        device = self.diff.text_encoder.device
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        # globally consistent random generators
        if self.seed is not None:
            local_seed = self._get_next_seed()
            rand_num_generator = torch.Generator('cuda')
            rand_num_generator.manual_seed(local_seed)
        else:
            rand_num_generator = None

        rgb, batch_size = self.diff.img_process_(rgb_in)
        # Encode image
        rgb_latent = self.diff.norm_img_to_latent(rgb)

        if prompt == None:
            # Batched empty text embedding
            if self.empty_text_embed is None:
                self.diff.encode_empty_text()
            text_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, 2, 1024]
        else:
            negative_text_embed, text_embed = self.diff.get_text_embedding_(prompt, self.diff.text_encoder.device)
            text_embed = torch.cat([negative_text_embed, text_embed])

        # Denoising list
        x_0_list = []
        x_next_list = []
        for t in timestep_list:
            # Sample a random timestep for each image
            t = torch.tensor(t).to(device=self.diff.unet.device).long()
            if add_noise_flag == True:
                ## add noise for latent
                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (t / self.scheduler_timesteps)
                    noise = self.multi_res_noise_like(
                        rgb_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        rgb_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    rgb_latent, noise, t
                )  # [B, 4, h, w]
                unet_input = noisy_latents
            else:
                unet_input = rgb_latent
                noisy_latents = rgb_latent

            # predict the noise residual
            noise_pred = self.predict_(unet_input, t, text_embed, prompt)
            # compute the previous noisy sample x_t -> x_t-1
            ouput_ = self.scheduler.step(
                noise_pred, t, noisy_latents, generator=None
            )
            x_0_list.append(ouput_.pred_original_sample)
            x_next_list.append(ouput_.prev_sample)
        return x_0_list


    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def multi_res_noise_like(self, x, strength=0.9, downscale_strategy="original", generator=None):
        if torch.is_tensor(strength):
            strength = strength.reshape((-1, 1, 1, 1))
        b, c, w, h = x.shape

        device = x.device

        up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
        noise = torch.randn(x.shape, device=x.device, generator=generator)

        if "original" == downscale_strategy:
            for i in range(10):
                r = (
                        torch.rand(1, generator=generator, device=device) * 2 + 2
                )  # Rather than always going 2x,
                w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        elif "every_layer" == downscale_strategy:
            for i in range(int(math.log2(min(w, h)))):
                w, h = max(1, int(w / 2)), max(1, int(h / 2))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
        elif "power_of_two" == downscale_strategy:
            for i in range(10):
                r = 2
                w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        elif "random_step" == downscale_strategy:
            for i in range(10):
                r = (
                        torch.rand(1, generator=generator, device=device) * 2 + 2
                )  # Rather than always going 2x,
                w, h = max(1, int(w / (r))), max(1, int(h / (r)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        else:
            raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

        noise = noise / noise.std()  # Scaled back to roughly unit variance
        return noise

    # @torch.no_grad()
    # def __call__(
    #         self,
    #         input_image: Union[Image.Image, torch.Tensor],
    #         denoising_steps: Optional[int] = None,
    #         processing_res: Optional[int] = None,
    #         resample_method: str = "bilinear",
    #         generator: Union[torch.Generator, None] = None,
    #         prompt: str = None,
    # ) -> DiffImgOutput:
    #     """
    #     Function invoked when calling the pipeline.
    #
    #     Args:
    #         input_image (`Image`):
    #             Input RGB (or gray-scale) image.
    #         denoising_steps (`int`, *optional*, defaults to `None`):
    #             Number of denoising diffusion steps during inference. The default value `None` results in automatic
    #             selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
    #             for Marigold-LCM models.
    #             Number of predictions to be ensembled.
    #         processing_res (`int`, *optional*, defaults to `None`):
    #             Effective processing resolution. When set to `0`, processes at the original image resolution. This
    #             produces crisper predictions, but may also lead to the overall loss of global context. The default
    #             value `None` resolves to the optimal value from the model config.
    #         resample_method: (`str`, *optional*, defaults to `bilinear`):
    #             Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
    #         generator (`torch.Generator`, *optional*, defaults to `None`)
    #             Random generator for initial noise generation.
    #     Returns:
    #         `DiffImgOutput`: Output class for prediction pipeline, including:
    #         - **gt_np** (`np.ndarray`) Predicted gt map, with values in the range of [0, 1]
    #     """
    #     # Model-specific optimal default values leading to fast and reasonable results.
    #     if denoising_steps is None:
    #         denoising_steps = self.default_denoising_steps
    #     if processing_res is None:
    #         processing_res = self.default_processing_resolution
    #
    #     assert processing_res >= 0
    #
    #     # Check if denoising step is reasonable
    #     self._check_inference_step(denoising_steps)
    #
    #     resample_method: InterpolationMode = get_tv_resample_method(resample_method)
    #
    #     # ----------------- Image Preprocess -----------------
    #     # Convert to torch tensor
    #     if isinstance(input_image, Image.Image):
    #         input_image = input_image.convert("RGB")
    #         # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
    #         rgb = pil_to_tensor(input_image)
    #         rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
    #     elif isinstance(input_image, torch.Tensor):
    #         rgb = input_image
    #     else:
    #         raise TypeError(f"Unknown input type: {type(input_image) = }")
    #     input_size = rgb.shape
    #     assert (
    #             4 == rgb.dim() and 3 == input_size[-3]
    #     ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"
    #
    #     # Resize image
    #     if processing_res > 0:
    #         rgb = resize_max_res(
    #             rgb,
    #             max_edge_resolution=processing_res,
    #             resample_method=resample_method,
    #         )
    #
    #     # Normalize rgb values
    #     rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
    #     rgb_norm = rgb_norm.to(self.dtype)
    #     assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
    #
    #     # ----------------- Predicting depth -----------------
    #     # Predict depth maps
    #     gt_pred_raw = self.single_infer(
    #         rgb_in=rgb_norm,
    #         num_inference_steps=denoising_steps,
    #         generator=generator,
    #     )
    #
    #     gt_preds = gt_pred_raw.detach()
    #     torch.cuda.empty_cache()  # clear vram cache for ensembling
    #
    #     # Convert to numpy
    #     gt_pred = gt_preds.squeeze()
    #
    #     # Clip output range
    #     assert gt_pred.min() >= -1.0 and gt_pred.max() <= 1.0
    #
    #     return DiffImgOutput(
    #         gt_np=gt_pred,
    #     )

    @torch.no_grad()
    def single_infer(
            self,
            rgb_in: torch.Tensor,
            num_inference_steps: int,
            generator: Union[torch.Generator, None] = None,
            prompt: str = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`): range [-1,1]
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.diff.text_encoder.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.diff.norm_img_to_latent(rgb_in)

        # Initial depth map (noise)
        gt_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        if prompt == None:
            # Batched empty text embedding
            if self.empty_text_embed is None:
                self.diff.encode_empty_text()
            text_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, 2, 1024]
        else:
            negative_text_embed, text_embed = self.diff.get_text_embedding_(prompt, self.diff.text_encoder.device)
            text_embed = torch.cat([negative_text_embed, text_embed])

        # Denoising loop
        iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, gt_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.predict_(unet_input, t, text_embed, prompt)

            # compute the previous noisy sample x_t -> x_t-1
            gt_latent = self.scheduler.step(
                noise_pred, t, gt_latent, generator=generator
            ).prev_sample

        gt = self.diff.latent_to_img_tensor(gt_latent)
        return gt

    def predict_(self, unet_input, t, text_embed, prompt):
        if prompt == None:
            noise_pred = self.diff.unet(
                unet_input, t, encoder_hidden_states=text_embed
            ).sample  # [B, 4, h, w]
        else:
            if self.low_resources == False:
                noise_pred = self.diff.unet(
                    torch.cat([unet_input] * 2), t,
                    encoder_hidden_states=text_embed
                ).sample  # [B, 4, h, w]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                negative_text_embed, text_embed = text_embed.chunk(2)
                noise_pred_uncond = self.diff.unet(
                    unet_input, t, encoder_hidden_states=negative_text_embed
                ).sample  # [B, 4, h, w]
                noise_pred_text = self.diff.unet(
                    unet_input, t, encoder_hidden_states=text_embed
                ).sample  # [B, 4, h, w]
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred


    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, ):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")
