import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load sample images and run inference
image_paths = [
    "assets/examples/SOH/000.png",
    "assets/examples/SOH/010.png"
]

# Run inference
prediction = model.inference(
    image=image_paths,
    process_res=504,
    process_res_method="upper_bound_resize",
    export_dir=None,
    export_format="glb"
)
print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics matrix: {prediction.extrinsics if prediction.extrinsics is not None else 'None'}")
print(f"Intrinsics matrix: {prediction.intrinsics if prediction.intrinsics is not None else 'None'}")
print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")


# Visualize input images and depth maps
n_images = prediction.depth.shape[0]

fig, axes = plt.subplots(2, n_images, figsize=(12, 6))

if n_images == 1:
    axes = axes.reshape(2, 1)

for i in range(n_images):
    # Show original image
    if prediction.processed_images is not None:
        axes[0, i].imshow(prediction.processed_images[i])
    axes[0, i].set_title(f"Input {i+1}")
    axes[0, i].axis('off')
    
    # Show depth map
    depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
    axes[1, i].imshow(depth_vis)
    axes[1, i].set_title(f"Depth {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

