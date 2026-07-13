# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input processor for Depth Anything 3 (parallelized).

This version removes the square center-crop step for "*crop" methods (same as your note).
In addition, it parallelizes per-image preprocessing using the provided `parallel_execution`.
"""

import cv2
import numpy as np

from multiprocessing.pool import ThreadPool
from typing import Callable, Dict, List

def parallel_execution(
    *args,
    action: Callable,
    num_processes=32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    **kwargs,
):
    # Partially copy from EasyVolumetricVideo (parallel_execution)
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved.
    # `*args` packs all positional arguments passed to the function into a tuple
    args = list(args)

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [
            (arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args
        ]
        # TODO: Support all types of iterable
        action_kwargs = {
            key: (
                kwargs[key][i]
                if isinstance(kwargs[key], list) and len(kwargs[key]) == length
                else kwargs[key]
            )
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for async_result in asyncs:
                results.append(async_result.get())  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results


class InputProcessor:
    """Prepares a batch of images for model inference.
    This processor converts a list of image file paths into a single, model-ready
    tensor. The processing pipeline is executed in parallel across multiple workers
    for efficiency.

    Pipeline:
      1) Load image and convert to RGB
      2) Boundary resize (upper/lower bound, preserving aspect ratio)
      3) Enforce divisibility by PATCH_SIZE:
         - "*resize" methods: each dimension is rounded to nearest multiple
           (may up/downscale a few px)
         - "*crop"   methods: each dimension is floored to nearest multiple via center crop
      4) Convert to tensor and apply ImageNet normalization
      5) Stack into (1, N, 3, H, W)

    Parallelization:
      - Each image is processed independently in a worker.
      - Order of outputs matches the input order.
    """
    
    PATCH_SIZE = 14

    def __init__(self):
        pass

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        image: list[np.ndarray | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        num_workers: int = 8,
        print_progress: bool = False,
        sequential: bool | None = None,
        desc: str | None = "Preprocess",
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Returns:
            (tensor, extrinsics_list, intrinsics_list)
            tensor shape: (1, N, 3, H, W)
        """
        sequential = self._resolve_sequential(sequential, num_workers)
        exts_list, ixts_list = self._validate_and_pack_meta(image, extrinsics, intrinsics)

        results = self._run_parallel(
            image=image,
            exts_list=exts_list,
            ixts_list=ixts_list,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
        )

        proc_imgs, out_sizes, out_ixts, out_exts = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(proc_imgs, out_sizes, out_ixts)

        batch_tensor = self._stack_batch(proc_imgs)
        out_exts = (
            np.asarray(out_exts)
            if out_exts is not None and out_exts[0] is not None
            else None
        )
        out_ixts = (
            np.asarray(out_ixts)
            if out_ixts is not None and out_ixts[0] is not None
            else None
        )
        return (batch_tensor, out_exts, out_ixts)

    # -----------------------------
    # __call__ helpers
    # -----------------------------
    def _resolve_sequential(self, sequential: bool | None, num_workers: int) -> bool:
        return (num_workers <= 1) if sequential is None else sequential

    def _validate_and_pack_meta(
        self,
        images: list[np.ndarray | str],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[list[np.ndarray | None] | None, list[np.ndarray | None] | None]:
        if extrinsics is not None and len(extrinsics) != len(images):
            raise ValueError("Length of extrinsics must match images when provided.")
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError("Length of intrinsics must match images when provided.")
        exts_list = [e for e in extrinsics] if extrinsics is not None else None
        ixts_list = [k for k in intrinsics] if intrinsics is not None else None
        return exts_list, ixts_list

    def _run_parallel(
        self,
        *,
        image: list[np.ndarray | str],
        exts_list: list[np.ndarray | None] | None,
        ixts_list: list[np.ndarray | None] | None,
        process_res: int,
        process_res_method: str,
        num_workers: int,
        print_progress: bool,
        sequential: bool,
        desc: str | None,
    ):
        results = parallel_execution(
            image,
            exts_list,
            ixts_list,
            action=self._process_one,  # (img, extrinsic, intrinsic, ...)
            num_processes=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if not results:
            raise RuntimeError(
                "No preprocessing results returned. Check inputs and parallel_execution."
            )
        return results

    def _unpack_results(self, results):
        """
        results: List[Tuple[torch.Tensor, Tuple[H, W], Optional[np.ndarray], Optional[np.ndarray]]]
        -> processed_images, out_sizes, out_intrinsics, out_extrinsics
        """
        try:
            processed_images, out_sizes, out_intrinsics, out_extrinsics = zip(*results)
        except Exception as e:
            raise RuntimeError(
                "Unexpected results structure from parallel_execution: "
                f"{type(results)} / sample: {results[0]}"
            ) from e

        return list(processed_images), list(out_sizes), list(out_intrinsics), list(out_extrinsics)

    def _unify_batch_shapes(
        self,
        processed_images: list[np.ndarray],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[np.ndarray], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all tensors to the smallest H, W; adjust intrinsics' cx, cy accordingly."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)

        # center_crop = T.CenterCrop((min_h, min_w))
        new_imgs, new_sizes, new_ixts = [], [], []
        for img_np, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
            crop_top = max(0, (H - min_h) // 2)
            crop_left = max(0, (W - min_w) // 2)
            new_imgs.append(img_np[:, crop_top : crop_top + min_h, crop_left : crop_left + min_w])
            new_sizes.append((min_h, min_w))
            if K is None:
                new_ixts.append(None)
            else:
                K_adj = K.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)
        return new_imgs, new_sizes, new_ixts

    def _stack_batch(self, processed_images: list[np.ndarray]) -> np.ndarray:
        return np.stack(processed_images)

    # -----------------------------
    # Per-item worker
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[np.ndarray, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        # Load & remember original size
        img = self._load_image(img)
        orig_h, orig_w = img.shape[:2]

        # Boundary resize
        img = self._resize_image(img, process_res, process_res_method)
        h, w = img.shape[:2]
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            img = self._make_divisible_by_resize(img, self.PATCH_SIZE)
            new_h, new_w = img.shape[:2]
            intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        elif process_res_method.endswith("crop"):
            img = self._make_divisible_by_crop(img, self.PATCH_SIZE)
            new_h, new_w = img.shape[:2]
            intrinsic = self._crop_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Convert to tensor & normalize
        img_tensor = self._normalize_image(img)
        _, H, W = img_tensor.shape
        assert (W, H) == (w, h), "Tensor size mismatch with image size after processing."

        # Return: (img_tensor, (H, W), intrinsic, extrinsic)
        return img_tensor, (H, W), intrinsic, extrinsic

    # -----------------------------
    # Intrinsics transforms
    # -----------------------------
    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def _crop_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        crop_h = (orig_h - h) // 2
        crop_w = (orig_w - w) // 2
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        return K

    # -----------------------------
    # I/O & normalization
    # -----------------------------
    def _load_image(self, img: np.ndarray | str) -> np.ndarray:
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"Failed to read image: {img}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            # Assume HxWxC uint8/RGB
            return img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        # Convert Image to float32 array and scale to [0, 1]
        img_np = img.astype(np.float32) / 255.0
        
        # Transpose from (H, W, C) to (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        return (img_np - mean) / std

    # -----------------------------
    # Boundary resizing
    # -----------------------------
    def _resize_image(self, img: np.ndarray, target_size: int, method: str) -> np.ndarray:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side(self, img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    def _resize_shortest_side(self, img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # -----------------------------
    # Make divisible by PATCH_SIZE
    # -----------------------------
    def _make_divisible_by_crop(self, img: np.ndarray, patch: int) -> np.ndarray:
        """
        Floor each dimension to the nearest multiple of PATCH_SIZE via center crop.
        Example: 504x377 -> 504x364
        """
        h, w = img.shape[:2]
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img[top: top + new_h, left: left + new_w]

    def _make_divisible_by_resize(self, img: np.ndarray, patch: int) -> np.ndarray:
        """
        Round each dimension to nearest multiple of PATCH_SIZE via small resize.
        """
        h, w = img.shape[:2]

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        upscale = (new_w > w) or (new_h > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


# Backward compatibility alias
InputAdapter = InputProcessor

