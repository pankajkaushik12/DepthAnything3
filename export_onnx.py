import os
import argparse
from typing import Optional
import onnx

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from depth_anything_3.api import DepthAnything3

class DA3ONNXWrapper(nn.Module):
    """
    A wrapper around DepthAnything3 for ONNX export.
    """
    def __init__(self, da3_model: DepthAnything3, infer_gs: bool = False, use_ray_pose: bool = False, ref_view_strategy: str = "saddle_balanced"):
        super().__init__()
        self.model = da3_model
        self.infer_gs = infer_gs
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy

    def forward(self, image: torch.Tensor, extrinsics: Optional[torch.Tensor] = None, intrinsics: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Call the underlying DA3 forward pass
        out = self.model(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            infer_gs=self.infer_gs,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )

        # Unpack dictionary into a strict tuple for deterministic ONNX output names
        depth = out.get("depth", torch.empty(0, device=image.device))
        conf = out.get("depth_conf", torch.empty(0, device=image.device))
        extrinsics_out = out.get("extrinsics", torch.empty(0, device=image.device))
        intrinsics_out = out.get("intrinsics", torch.empty(0, device=image.device))

        return depth, conf, extrinsics_out, intrinsics_out

def load_model(model_name: str = "depth-anything/DA3-BASE", device: str = "cpu") -> DepthAnything3:
    """
    Loads the pretrained Depth Anything 3 model.
    
    Args:
        model_name: Name or HuggingFace path of the DA3 model preset.
        device: Target device for loading ('cpu' or 'cuda').
    """
    print(f"Loading {model_name} onto {device}...")
    
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    return model.eval()

def export_onnx(
    model: DepthAnything3,
    onnx_path: str,
    device: str = "cpu",
    opset_version: int = 17,
    num_views: int = 2,
    height: int = 504,
    width: int = 504,
):
    """
    Exports the DA3 model to ONNX format with dynamic axes for resolution and view count.
    
    Args:
        model: Loaded DepthAnything3 model instance.
        onnx_path: Destination path for the .onnx file.
        device: Device to run the tracing on.
        opset_version: ONNX opset version (17+ recommended for modern attention/vision ops).
        num_views: Number of views for the dummy trace input.
        height: Image height for the dummy trace input (must be patch-divisible).
        width: Image width for the dummy trace input (must be patch-divisible).
        with_camera_inputs: If True, exports graph expecting (image, extrinsics, intrinsics).
                            If False, exports graph expecting only (image).
    """
    if onnx_path is None:
        raise ValueError("onnx_path must be provided.")

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    # Wrap the model to isolate non-tensor arguments and dictionary outputs
    wrapped_model = DA3ONNXWrapper(
        da3_model=model,
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ).to(device)
    wrapped_model.eval()

    print(f"Generating dummy inputs (Views: {num_views}, H: {height}, W: {width})...")
    # DA3 expects batch dimension B=1, and view dimension N
    dummy_image = torch.randn(1, num_views, 3, height, width, device=device, dtype=torch.float32)

    # Extrinsics: (1, N, 4, 4), Intrinsics: (1, N, 3, 3)
    dummy_ext = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(1, num_views, 1, 1)
    dummy_int = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, num_views, 1, 1)
    dummy_inputs = (dummy_image, dummy_ext, dummy_int)

    input_names = ["image", "extrinsics_in", "intrinsics_in"]
    output_names = ["depth", "depth_conf", "extrinsics_out", "intrinsics_out"]
    
    dynamic_axes = {
        "image": {1: "num_views", 3: "height", 4: "width"},
        "extrinsics_in": {1: "num_views"},
        "intrinsics_in": {1: "num_views"},
        "depth": {1: "num_views"},
        "depth_conf": {1: "num_views"},
        "extrinsics_out": {1: "num_views"},
        "intrinsics_out": {1: "num_views"},
    }

    print(f"Exporting model to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print("ONNX export successful!")


def check_onnx(onnx_path: str):
    """
    Validates the exported ONNX model graph.
    """
    print(f"Checking ONNX model integrity at {onnx_path}...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    model = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model=model)
        print("The ONNX graph is clean and valid!")
    except onnx.checker.ValidationError as e:
        print(f"Graph validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 (DA3) to ONNX format.")
    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-BASE", help="DA3 model preset or HuggingFace path.")
    parser.add_argument("--onnx_path", type=str, default="weights/da3_base.onnx", help="Path to save the exported ONNX model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for exporting (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--views", type=int, default=2, help="Number of dummy views for tracing.")
    parser.add_argument("--height", type=int, default=504, help="Dummy height (must be patch divisible).")
    parser.add_argument("--width", type=int, default=504, help="Dummy width (must be patch divisible).")
    parser.add_argument("--with_cams", action="store_true", help="Include extrinsics and intrinsics as ONNX graph inputs.")
    args = parser.parse_args()

    # Load Model
    model = load_model(model_name=args.model_name, device=args.device)

    # Export to ONNX
    export_onnx(
        model=model,
        onnx_path=args.onnx_path,
        device=args.device,
        opset_version=args.opset,
        num_views=args.views,
        height=args.height,
        width=args.width,
    )

    # Verify Graph
    check_onnx(onnx_path=args.onnx_path)

