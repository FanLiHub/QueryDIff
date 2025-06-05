import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from ..third_party.diffusers import (
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
import torch.nn as nn
from diffusers import DDIMScheduler
import random
import logging



def init_models(model_id, data_type):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=data_type)
    unet = pipe.unet
    vae = pipe.vae
    clip = pipe.text_encoder
    clip_tokenizer = pipe.tokenizer
    return unet, vae, clip, clip_tokenizer, pipe


class DiffImgOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        gt_np (`np.ndarray`):
            Predicted gt map, with values in the range of [0, 1].
    """

    gt_np: np.ndarray


class DiffImgPipeline(nn.Module):
    """
    function:
       Ⅰ img->norm_img: img_process_
       Ⅱ norm_img->latent: norm_img_to_latent
       Ⅲ latent->norm_img: latent_to_img_tensor
       Ⅳ norm_img->PIL: img_tensor_to_image
       Ⅴ seg_gt->norm_seg_gt: seg_gt_process
       Ⅵ empty_text->embeddings: encode_empty_text
       Ⅶ text->embeddings: get_text_embedding_

    model:
    unet (`UNet2DConditionModel`):
        Conditional U-Net to denoise the depth latent, conditioned on image latent.
    vae (`AutoencoderKL`):
        Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
        to and from latent representations.
    scheduler (`DDIMScheduler`):
        A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    text_encoder (`CLIPTextModel`):
        Text-encoder, for empty text embedding.
    tokenizer (`CLIPTokenizer`):
        CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215

    def __init__(
            self,
            model_id,
            data_type,
            low_resources: Optional[bool] = True,
            default_denoising_steps: Optional[int] = None,
            default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.unet, self.vae, self.text_encoder, self.tokenizer, self.pipe = init_models(
            model_id=model_id,
            data_type=data_type)

        self.low_resources = low_resources
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        # Trainability
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.unet.enable_xformers_memory_efficient_attention()


    # def predict_(self, unet_input, t, text_embed, prompt):
    #     if prompt == None:
    #         noise_pred = self.unet(
    #             unet_input, t, encoder_hidden_states=text_embed
    #         ).sample  # [B, 4, h, w]
    #     else:
    #         if self.low_resources == False:
    #             noise_pred = self.unet(
    #                 torch.cat([unet_input] * 2), t,
    #                 encoder_hidden_states=text_embed
    #             ).sample  # [B, 4, h, w]
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    #         else:
    #             negative_text_embed, text_embed = text_embed.chunk(2)
    #             noise_pred_uncond = self.unet(
    #                 unet_input, t, encoder_hidden_states=negative_text_embed
    #             ).sample  # [B, 4, h, w]
    #             noise_pred_text = self.unet(
    #                 unet_input, t, encoder_hidden_states=text_embed
    #             ).sample  # [B, 4, h, w]
    #             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    #     return noise_pred


    def get_text_embedding_(self, prompt, device, negative_prompt=None,
                            do_classifier_free_guidance=True, num_images_per_prompt=1):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        return negative_prompt_embeds, prompt_embeds

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def norm_img_to_latent(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode norm RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`): range [-1,1]
                Input norm RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        # ## other way
        # latents = self.vae.encode(rgb_in).latent_dist.sample(generator=None) * self.rgb_latent_scale_factor
        return rgb_latent

    def latent_to_img_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into img.

        Args:
            latent (`torch.Tensor`):
                latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded img range [0,1].
        """
        # scale latent
        latent = latent / self.rgb_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(latent)
        img_norm = self.vae.decoder(z)
        img = (img_norm / 2 + 0.5).clamp(0, 1)
        return img

    def img_tensor_to_image(self, img_norm: torch.Tensor) -> Image:
        '''

        Args:
            img_norm (`torch.Tensor`): range [0,1]

        Returns:
            PIL.Image
        '''
        images = img_norm.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def img_process_(self, img):
        '''

        Args:
            img: tensor - [B, 3, H, W] - range [0,255]

        Returns:
            batch_size: int
            img_norm - [B, 3, H, W] - range [-1,1]

        '''
        batch_size = img.shape[0]
        img_norm = img / 255.0 * 2.0 - 1.0
        return img_norm, batch_size


    def seg_gt_process(self, gt, ingore_id=255):
        '''

        Args:
            gt: tensor - [B, H, W] - range >=0

        Returns:
            gt_norm: tensor - [B, 3, H, W] - range [-1,1]
        '''
        valid_mask = (gt != ingore_id)
        if valid_mask.sum()==0:
            print('------------------------------------------------------0')
        norm_max = gt[valid_mask].max()
        norm_min = gt[valid_mask].min()
        gt_norm = (gt - norm_min) / (norm_max - norm_min + 0.0000001)
        gt_norm = gt_norm * 2.0 - 1.0
        gt_norm = torch.clip(gt_norm, -1, 1)
        if 4 == len(gt_norm.shape) and gt_norm.shape[1] == 1:
            gt_norm = gt_norm.repeat(1, 3, 1, 1)
        elif 3 == len(gt_norm.shape):
            gt_norm = gt_norm.unsqueeze(1)
            if gt_norm.shape[1] == 1:
                gt_norm = gt_norm.repeat(1, 3, 1, 1)
        return gt_norm
