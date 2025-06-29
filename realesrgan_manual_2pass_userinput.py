import argparse
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os

def enhance_image(input_path, output_path, model_path, upscale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)['params_ema']
    model = RRDBNet(3, 3, 64, 23, 32, scale=4)
    model.load_state_dict(state_dict, strict=True)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    if upscale_factor not in [1, 2, 4, 8, 16]:
        raise ValueError("scale must be 1, 2, 4, 8, or 16")

    output, _ = upsampler.enhance(img_np, outscale=min(upscale_factor, 4))

    if upscale_factor > 4:
        second_pass_scale = upscale_factor // 4
        output, _ = upsampler.enhance(np.array(Image.fromarray(output)), outscale=second_pass_scale)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(output).save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/RealESRGAN_x4plus.pth")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    enhance_image(args.input, args.output, args.model, args.scale)
