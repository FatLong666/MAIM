import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor

from maim.models.unet import UNet3DConditionModel
from maim.pipelines.pipeline_maim import MAIMPipeline
from maim.util import save_videos_grid


def his_match(src, dst):
    src = src * 255.0
    dst = dst * 255.0
    src = src.astype(np.uint8)
    dst = dst.astype(np.uint8)
    res = np.zeros_like(dst)

    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(0, 256), density=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
        np.clip(index, 0, 255, out=index)
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='outputs/flower/unet', help='Path for of model weights')
    parser.add_argument('--pretrain_weight', type=str, default='./checkpoints/stable-diffusion-v1-4', help='Path for pretrained weight (SD v1.4)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--first_frame_path', type=str, default='benchmark/flower_in_bloom/flower_22.png', help='The path for first frame image')
    parser.add_argument('-p', '--prompt', type=str, default='a flowers in bloom, watercolor style.', help='The video prompt. Default value: same to the filename of the first frame image')
    parser.add_argument('-hs', '--height', type=int, default=320, help='video height')
    parser.add_argument('-ws', '--width', type=int, default=512, help='video width')
    # 如果是编辑视频的话length=24
    parser.add_argument('-l', '--length', type=int, default=16, help='video length')
    parser.add_argument('--cfg', type=float, default=12.5, help='classifier-free guidance scale')
    parser.add_argument('--editing', default=False, help='video editing')
    args = parser.parse_args()

    # load weights
    pretrained_model_path = args.pretrain_weight
    my_model_path = args.weight
    unet = UNet3DConditionModel.from_pretrained('/'.join(my_model_path.split('/')[:-1]), subfolder=my_model_path.split('/')[-1], torch_dtype=torch.float16).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
    # 增加clip对应的模块
    image_encoder = CLIPVisionModel.from_pretrained("./checkpoints/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("./checkpoints/clip-vit-base-patch32")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # image_encoder.to('cuda')

    print(args.editing)
    if args.editing:
        ddim_inv_latent = torch.load(f"{'/'.join(my_model_path.split('/')[:-1])}/inv_latents/ddim_latent-500.pt").to(torch.float16)
    else:
        ddim_inv_latent = None

    # build pipeline
    unet.enable_xformers_memory_efficient_attention()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    pipe = MAIMPipeline(image_encoder=image_encoder, image_processor=image_processor, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")).to("cuda")
    pipe.enable_vae_slicing()
    generator = torch.Generator(device='cuda')

    # read the first frame
    img_path = args.first_frame_path
    if args.prompt is None:
        prompt = img_path.split('/')[-1][:-4].replace('_', ' ')
    else:
        prompt = args.prompt
    print(prompt)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (512, 320))[:, :, ::-1]
    first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
    first_frame_latents = first_frame_latents / 127.5 - 1.0
    first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample() * 0.18215
    first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

    timesteps = torch.randint(300, 400, (1,), device='cuda')

    timesteps = timesteps.long()
    noise = torch.randn_like(first_frame_latents)
    noisy_first_frame_latents = noise_scheduler.add_noise(first_frame_latents, noise, timesteps)


    # video generation
    video = pipe(prompt, generator=generator, latents=first_frame_latents, video_length=args.length, height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.cfg, use_inv_latent=False, num_inv_steps=50, ddim_inv_latent=ddim_inv_latent, image_path=args.first_frame_path, noisy_first_frame_latents=noisy_first_frame_latents).videos
    for f in range(1, video.shape[2]):
        former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
        frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
        result = his_match(former_frame, frame)
        result = torch.Tensor(result).type_as(video).to(video.device)
        video[0, :, f, :, :] = result.permute(2, 0, 1)
    save_path = ''

    save_path = args.output
    save_path = os.path.join(save_path, img_path.split('/')[-1][:-4] + '.gif')
    save_videos_grid(video, save_path)

if __name__ == '__main__':
    main()