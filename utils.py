import numpy as np
import matplotlib

def transpose_last_two_axes(arr: np.ndarray):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)

def set_sky_regions_to_max_depth_np(depth: np.ndarray, depth_conf: np.ndarray | None, non_sky_mask: np.ndarray, max_depth: float = 200.0,) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Set sky regions to maximum depth and high confidence using pure NumPy.

    Args:
        depth: Depth array (N, H, W) or (H, W)
        depth_conf: Optional depth confidence array (N, H, W) or (H, W)
        non_sky_mask: Boolean mask where True indicates non-sky regions
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.copy()
    sky_mask = ~non_sky_mask

    # Set sky regions to max_depth
    depth[sky_mask] = max_depth

    if depth_conf is not None:
        depth_conf = depth_conf.copy()
        depth_conf[sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None

def process_mono_sky_estimation_np(depth: np.ndarray, depth_conf: np.ndarray | None, sky: np.ndarray | None, sky_threshold: float = 0.3,) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Process mono sky estimation in NumPy.
        """
        if sky is None:
            print("No sky estimation found in outputs. Skipping sky processing.")
            return depth, depth_conf

        # non_sky_mask is True where sky prediction is below threshold
        non_sky_mask = sky < sky_threshold

        # Guard against dynamic shape / small pixel count edge cases
        if np.sum(non_sky_mask) <= 10 or np.sum(~non_sky_mask) <= 10:
            return depth, depth_conf

        non_sky_depth = depth[non_sky_mask]
        
        # Subsample if large (replicates torch.randint)
        if non_sky_depth.size > 100000:
            idx = np.random.choice(non_sky_depth.size, size=100000, replace=False)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth

        # Replicates torch.quantile(sampled_depth, 0.99)
        non_sky_max = float(np.percentile(sampled_depth, 99))

        # Apply sky depth and confidence updates
        updated_depth, updated_conf = set_sky_regions_to_max_depth_np(depth=depth, depth_conf=depth_conf, non_sky_mask=non_sky_mask, max_depth=non_sky_max,)

        return updated_depth, updated_conf

def align_nested_depth_np(
    main_depth: np.ndarray,
    main_conf: np.ndarray,
    metric_depth: np.ndarray,
    metric_sky: np.ndarray,
    intrinsics: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """
    Pure NumPy implementation of DA3 Nested depth alignment.
    Aligns the main relative depth map to the metric depth map using least squares.
    """
    # Apply metric scaling to the metric depth output using Intrinsics
    if intrinsics is not None and intrinsics.shape[-2:] == (3, 3):
        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        focal = (fx + fy) / 2.0
        # Broadcast focal length over spatial dimensions
        focal = focal.reshape(-1, 1, 1) if focal.ndim > 0 else focal
        metric_depth = metric_depth * (focal / 300.0)
    else:
        print("WARNING: Intrinsics missing. Nested alignment will use unscaled metric depth.")

    # Compute Sky Mask
    non_sky_mask = metric_sky < 0.3
    if np.sum(non_sky_mask) <= 10:
        return main_depth, 1.0  # Fallback if no valid non-sky pixels

    # Sample confidence to find median
    conf_ns = main_conf[non_sky_mask]
    if conf_ns.size > 100000:
        conf_sampled = np.random.choice(conf_ns, 100000, replace=False)
    else:
        conf_sampled = conf_ns
    median_conf = float(np.median(conf_sampled))

    # Compute Alignment Mask
    align_mask = (
        (main_conf >= median_conf) & 
        non_sky_mask & 
        (metric_depth > 1e-2) & 
        (main_depth > 1e-3)
    )

    valid_main = main_depth[align_mask]
    valid_metric = metric_depth[align_mask]

    if valid_main.size == 0:
        return main_depth, 1.0  # Fallback

    # Compute Least Squares Scale (a ≈ s * b) --> metric ≈ s * main
    num = np.dot(valid_metric.ravel(), valid_main.ravel())
    den = np.maximum(np.dot(valid_main.ravel(), valid_main.ravel()), 1e-12)
    scale_factor = float(num / den)

    # Apply Scale
    aligned_depth = main_depth * scale_factor
    
    return aligned_depth, scale_factor

def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )

def visualize_depth(depth: np.ndarray, depth_min=None, depth_max=None, percentile=2, ret_minmax=False, ret_type=np.uint8, cmap="Spectral",):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np

def normalize_extrinsics(ex_t: np.ndarray | None) -> np.ndarray | None:
    """Normalize extrinsics"""
    if ex_t is None:
        return None
    
    transform = affine_inverse_np(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    
    c2ws = affine_inverse_np(ex_t_norm)
    translations = c2ws[..., :3, 3]
    dists = np.linalg.norm(translations, axis=-1)
    median_dist = np.median(dists)
    median_dist = np.clip(median_dist, min=1e-1, a_max=None)
    
    ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
    
    return ex_t_norm

def _to44(ext):
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
        return out
    return ext

def _poses_from_ext(ext_ref: np.ndarray, ext_est: np.ndarray):
    ext_ref = _to44(ext_ref)
    ext_est = _to44(ext_est)
    pose_ref = affine_inverse_np(ext_ref)
    pose_est = affine_inverse_np(ext_est)
    return pose_ref, pose_est

def _umeyama_sim3_from_paths_evo_rep(pose_ref: np.ndarray, pose_est: np.ndarray, with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Pure NumPy replication of evo's Umeyama alignment and PosePath3D transformation.
    
    Args:
        pose_ref: (N, 4, 4) ground truth / reference SE(3) poses
        pose_est: (N, 4, 4) estimated SE(3) poses
        
    Returns:
        r (3,3): Rotation matrix
        t (3,): Translation vector
        s (float): Scale factor
        pose_est_aligned (N, 4, 4): The fully aligned SE(3) poses
    """
    if pose_ref.shape != pose_est.shape:
        raise ValueError("Data matrices must have the same shape")

    # 1. Extract translation vectors (m=3 dimensions, n=N points)
    # Transposing to (3, N) to perfectly match evo's data structure
    x = pose_est[:, :3, 3].T
    y = pose_ref[:, :3, 3].T
    m, n = x.shape

    # 2. Umeyama Algorithm (Evo exact replication)
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # Variance
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # Vectorized covariance calculation (replacing evo's for-loop)
    cov_xy = (1.0 / n) * ((y - mean_y[:, np.newaxis]) @ (x - mean_x[:, np.newaxis]).T)

    u, d, v = np.linalg.svd(cov_xy)

    # Check for degenerate rank
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("Degenerate covariance rank, Umeyama alignment is not possible")

    # Ensure RHS coordinate system (handle reflections)
    s_mat = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s_mat[m - 1, m - 1] = -1

    # Final R, t, s parameters
    r = u.dot(s_mat).dot(v)
    s = 1.0 / sigma_x * np.trace(np.diag(d).dot(s_mat)) if with_scale else 1.0
    t = mean_y - s * r.dot(mean_x)

    # 3. Apply transformation exactly as evo's `scale()` and `transform()` methods do
    pose_est_aligned = pose_est.copy()

    # evo left-multiplies: new_pose = T_align @ p_scaled
    # Rotation: R_new = r @ R_old
    pose_est_aligned[:, :3, :3] = np.matmul(r, pose_est[:, :3, :3])

    # Translation: t_new = r @ (s * t_old) + t
    t_scaled = s * pose_est[:, :3, 3]
    pose_est_aligned[:, :3, 3] = (t_scaled @ r.T) + t

    return r, t, s, pose_est_aligned

def _umeyama_sim3_from_paths(pose_ref, pose_est):
    r, t, s, pose_est_aligned = _umeyama_sim3_from_paths_evo_rep(pose_ref, pose_est, with_scale=True)
    return r, t, s, pose_est_aligned

def _apply_sim3_to_poses(poses, r, t, s):
    out = poses.copy()
    Ri = poses[:, :3, :3]
    ti = poses[:, :3, 3]
    out[:, :3, :3] = r @ Ri
    out[:, :3, 3] = (r @ (s * ti.T)).T + t
    return out

def _median_nn_thresh(pose_ref, pose_est_aligned):
    P_ref = pose_ref[:, :3, 3]
    P_est = pose_est_aligned[:, :3, 3]
    dists = []
    for p in P_est:
        dd = np.linalg.norm(P_ref - p[None, :], axis=1)
        dists.append(dd.min())
    return float(np.median(dists)) if dists else 0.0

def _ransac_align_sim3(
    pose_ref, pose_est, sub_n=None, inlier_thresh=None, max_iters=10, random_state=None
):
    rng = np.random.default_rng(random_state)
    N = pose_ref.shape[0]
    idx_all = np.arange(N)
    if sub_n is None:
        sub_n = max(3, (N + 1) // 2)
    else:
        sub_n = max(3, min(sub_n, N))

    # Pre-alignment + default threshold
    r0, t0, s0, pose_est0 = _umeyama_sim3_from_paths(pose_ref, pose_est)
    if inlier_thresh is None:
        inlier_thresh = _median_nn_thresh(pose_ref, pose_est0)

    P_ref_all = pose_ref[:, :3, 3]

    best_model = (r0, t0, s0)
    best_inliers = None
    best_score = (-1, np.inf)  # (num_inliers, mean_err)

    for _ in range(max_iters):
        sample = rng.choice(idx_all, size=sub_n, replace=False)
        try:
            r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[sample], pose_est[sample])
        except Exception:
            continue
        pose_h = _apply_sim3_to_poses(pose_est, r, t, s)
        P_h = pose_h[:, :3, 3]
        errs = np.linalg.norm(P_h - P_ref_all, axis=1)  # Match by same index
        inliers = errs <= inlier_thresh
        k = int(inliers.sum())
        mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
        if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
            best_score = (k, mean_err)
            best_model = (r, t, s)
            best_inliers = inliers

    # Fit again with best inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[best_inliers], pose_est[best_inliers])
    else:
        r, t, s = best_model
    return r, t, s

def align_poses_umeyama(
    ext_ref: np.ndarray,
    ext_est: np.ndarray,
    return_aligned=False,
    ransac=False,
    sub_n=None,
    inlier_thresh=None,
    ransac_max_iters=10,
    random_state=None,
):
    """
    Align estimated trajectory to reference using Umeyama Sim(3).
    Default no RANSAC; if ransac=True, use RANSAC (max iterations default 10).
    - sub_n defaults to half the number of frames (rounded up, at least 3)
    - inlier_thresh defaults to median of "distance from each estimated pose to
      nearest reference pose after pre-alignment"
    Returns rotation (3x3), translation (3,), scale; optionally returns aligned extrinsics (4x4).
    """
    pose_ref, pose_est = _poses_from_ext(ext_ref, ext_est)

    if not ransac:
        r, t, s, pose_est_aligned = _umeyama_sim3_from_paths(pose_ref, pose_est)
    else:
        r, t, s = _ransac_align_sim3(
            pose_ref,
            pose_est,
            sub_n=sub_n,
            inlier_thresh=inlier_thresh,
            max_iters=ransac_max_iters,
            random_state=random_state,
        )
        pose_est_aligned = _apply_sim3_to_poses(pose_est, r, t, s)

    if return_aligned:
        ext_est_aligned = affine_inverse_np(pose_est_aligned)
        return r, t, s, ext_est_aligned
    return r, t, s

