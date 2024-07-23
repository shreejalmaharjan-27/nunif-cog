# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # todo

    def predict(
        self,
        model_type: str = Input(
            description="Type of model to use", 
            default="art", 
            choices=['art', 'art_scan', 'photo', 'swin_unet/art', 'swin_unet/art_scan', 'swin_unet/photo', 'cunet/art', 'upconv_7/art', 'upconv_7/photo'],
        ),
        upscaling: str = Input(
            description="Upscaling factor", 
            default="2x", 
            choices=['noise', 'scale','2x', '4x'],
        ),
        noise_level: int = Input(
            description="Noise level", 
            default=3, 
            choices=[-1, 0, 1, 2, 3],
        ),
        tile_size: int = Input(
            description="Tile size", 
            default=256, 
        ),
        batch_size: int = Input(
            description="Batch size", 
            default=4, 
        ),
        transparency: bool = Input(
            description="Transparency", 
            default=False, 
        ),
        file: Path = Input(
            description="Input image or video file", 
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if upscaling not in ['noise', 'scale']:
            method = f"scale{upscaling}"
        else:
            method = upscaling
        model = torch.hub.load(
            "./", 
            "waifu2x",
            method=method, 
            noise_level=noise_level, 
            trust_repo=True, 
            source='local', 
            tile_size=tile_size, 
            batch_size=batch_size, 
            keep_alpha=transparency,
            model_type=model_type
        ).to("cuda")

        input_image = Image.open(file)
        result = model.infer(input_image)
        result.save("output.jpg")
        return Path("output.jpg")
