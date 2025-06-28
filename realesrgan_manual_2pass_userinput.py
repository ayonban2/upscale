import argparse
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os

def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN 2-pass manual upscale")
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--model', default='models/RealESRGAN_x4plus.pth', help='Path to the .pth model file')
    parser.add_argument('--scale', type=int, default=4, help='Desired upscale factor: 1, 2, 4, 8, or 16')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}")
    state_dict = torch.load(args.model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['params_ema']
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state_dict, strict=True)

    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path=args.model,
        model=model,
        tile=512,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    # Load input image
    img = Image.open(args.input).convert('RGB')
    img_np = np.array(img)

    upscale_factor = args.scale
    if upscale_factor not in [1, 2, 4, 8, 16]:
        raise ValueError("Invalid scale. Use one of: 1, 2, 4, 8, 16")

    perform_second_pass = upscale_factor > 4
    print(f"Upscaling to {upscale_factor}x {'in two passes' if perform_second_pass else 'in one pass'}...")

    # First pass (max 4x)
    first_pass_scale = 4 if perform_second_pass else upscale_factor
    output, _ = upsampler.enhance(img_np, outscale=first_pass_scale)

    # Second pass if needed
    if perform_second_pass:
        second_pass_scale = upscale_factor // 4
        output, _ = upsampler.enhance(np.array(Image.fromarray(output)), outscale=second_pass_scale)

    # Save output
    output_img = Image.fromarray(output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_img.save(args.output)

    print(f"Saved enhanced image to: {args.output}")

if __name__ == "__main__":
    main()
