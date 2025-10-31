import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LatentSample:
    """
    Container for a single latent with metadata
    """
    latent: np.ndarray  # (d,) flattened latent vector
    global_idx: int  # Global index in dataset
    shape: Tuple[int, ...]  # Original shape before flattening (C, H, W)
    latents_npz_path: str  # Path to the latent cache file
    metadata: Optional[Dict] = None  # Any extra info (prompt, filename, etc.)


class CarreDuChampComputer:
    """
    Core CDC-FM computation - agnostic to data source
    Just handles the math for a batch of same-size latents
    """

    def __init__(
        self,
        k_neighbors: int = 256,
        k_bandwidth: int = 8,
        d_cdc: int = 8,
        gamma: float = 1.0,
        device: str = 'cuda'
    ):
        self.k = k_neighbors
        self.k_bw = k_bandwidth
        self.d_cdc = d_cdc
        self.gamma = gamma
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def compute_knn_graph(self, latents_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph using pure PyTorch

        Args:
            latents_np: (N, d) numpy array of same-dimensional latents

        Returns:
            distances: (N, k_actual+1) distances (k_actual may be less than k if N is small)
            indices: (N, k_actual+1) neighbor indices
        """
        N, d = latents_np.shape

        # Clamp k to available neighbors (can't have more neighbors than samples)
        k_actual = min(self.k, N - 1)

        # Convert to torch tensor
        latents_tensor = torch.from_numpy(latents_np).to(self.device)

        # Compute pairwise L2 distances efficiently
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # This is more memory efficient than computing all pairwise differences
        # For large batches, we'll chunk the computation
        chunk_size = 1000  # Process 1000 queries at a time to manage memory

        if N <= chunk_size:
            # Small batch: compute all at once
            distances_sq = torch.cdist(latents_tensor, latents_tensor, p=2) ** 2
            distances_k_sq, indices_k = torch.topk(
                distances_sq, k=k_actual + 1, dim=1, largest=False
            )
            distances = torch.sqrt(distances_k_sq).cpu().numpy()
            indices = indices_k.cpu().numpy()
        else:
            # Large batch: chunk to avoid OOM
            distances_list = []
            indices_list = []

            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk = latents_tensor[i:end_i]

                # Compute distances for this chunk
                distances_sq = torch.cdist(chunk, latents_tensor, p=2) ** 2
                distances_k_sq, indices_k = torch.topk(
                    distances_sq, k=k_actual + 1, dim=1, largest=False
                )

                distances_list.append(torch.sqrt(distances_k_sq).cpu().numpy())
                indices_list.append(indices_k.cpu().numpy())

                # Free memory
                del distances_sq, distances_k_sq, indices_k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            distances = np.concatenate(distances_list, axis=0)
            indices = np.concatenate(indices_list, axis=0)

        return distances, indices

    @torch.no_grad()
    def compute_gamma_b_single(
        self,
        point_idx: int,
        latents_np: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
        epsilon: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Γ_b for a single point

        Args:
            point_idx: Index of point to process
            latents_np: (N, d) all latents in this batch
            distances: (N, k+1) precomputed distances
            indices: (N, k+1) precomputed neighbor indices
            epsilon: (N,) bandwidth per point

        Returns:
            eigenvectors: (d_cdc, d) as half precision tensor
            eigenvalues: (d_cdc,) as half precision tensor
        """
        d = latents_np.shape[1]

        # Get neighbors (exclude self)
        neighbor_idx = indices[point_idx, 1:]  # (k,)
        neighbor_points = latents_np[neighbor_idx]  # (k, d)

        # Clamp distances to prevent overflow (max realistic L2 distance)
        MAX_DISTANCE = 1e10
        neighbor_dists = np.clip(distances[point_idx, 1:], 0, MAX_DISTANCE)
        neighbor_dists_sq = neighbor_dists ** 2  # (k,)

        # Compute Gaussian kernel weights with numerical guards
        eps_i = max(epsilon[point_idx], 1e-10)  # Prevent division by zero
        eps_neighbors = np.maximum(epsilon[neighbor_idx], 1e-10)

        # Compute denominator with guard against overflow
        denom = eps_i * eps_neighbors
        denom = np.maximum(denom, 1e-20)  # Additional guard

        # Compute weights with safe exponential
        exp_arg = -neighbor_dists_sq / denom
        exp_arg = np.clip(exp_arg, -50, 0)  # Prevent exp overflow/underflow
        weights = np.exp(exp_arg)

        # Normalize weights, handle edge case of all zeros
        weight_sum = weights.sum()
        if weight_sum < 1e-20 or not np.isfinite(weight_sum):
            # Fallback to uniform weights
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weight_sum

        # Compute local mean
        m_star = np.sum(weights[:, None] * neighbor_points, axis=0)

        # Center and weight for SVD
        centered = neighbor_points - m_star
        weighted_centered = np.sqrt(weights)[:, None] * centered  # (k, d)

        # Validate input is finite before SVD
        if not np.all(np.isfinite(weighted_centered)):
            logger.warning(f"Non-finite values detected in weighted_centered for point {point_idx}, using fallback")
            # Fallback: use uniform weights and simple centering
            weights_uniform = np.ones(len(neighbor_points)) / len(neighbor_points)
            m_star = np.mean(neighbor_points, axis=0)
            centered = neighbor_points - m_star
            weighted_centered = np.sqrt(weights_uniform)[:, None] * centered

        # Move to GPU for SVD
        weighted_centered_torch = torch.from_numpy(weighted_centered).to(
            self.device, dtype=torch.float32
        )

        try:
            U, S, Vh = torch.linalg.svd(weighted_centered_torch, full_matrices=False)
        except RuntimeError as e:
            logger.debug(f"GPU SVD failed for point {point_idx}, using CPU: {e}")
            try:
                U, S, Vh = np.linalg.svd(weighted_centered, full_matrices=False)
                U = torch.from_numpy(U).to(self.device)
                S = torch.from_numpy(S).to(self.device)
                Vh = torch.from_numpy(Vh).to(self.device)
            except np.linalg.LinAlgError as e2:
                logger.error(f"CPU SVD also failed for point {point_idx}: {e2}, returning zero matrix")
                # Return zero eigenvalues/vectors as fallback
                return (
                    torch.zeros(self.d_cdc, d, dtype=torch.float16),
                    torch.zeros(self.d_cdc, dtype=torch.float16)
                )

        # Eigenvalues of Γ_b
        eigenvalues_full = S ** 2

        # Keep top d_cdc
        if len(eigenvalues_full) >= self.d_cdc:
            top_eigenvalues, top_idx = torch.topk(eigenvalues_full, self.d_cdc)
            top_eigenvectors = Vh[top_idx, :]  # (d_cdc, d)
        else:
            # Pad if k < d_cdc
            top_eigenvalues = eigenvalues_full
            top_eigenvectors = Vh
            if len(eigenvalues_full) < self.d_cdc:
                pad_size = self.d_cdc - len(eigenvalues_full)
                top_eigenvalues = torch.cat([
                    top_eigenvalues,
                    torch.zeros(pad_size, device=self.device)
                ])
                top_eigenvectors = torch.cat([
                    top_eigenvectors,
                    torch.zeros(pad_size, d, device=self.device)
                ])

        # Eigenvalue Rescaling (per CDC-FM paper Appendix E, Equation 33)
        # Paper formula: c_i = (1/λ_1^i) × min(neighbor_distance²/9, c²_max)
        # Then apply gamma: γc_i Γ̂(x^(i))
        #
        # Our implementation:
        # 1. Normalize by max eigenvalue (λ_1^i) - aligns with paper's 1/λ_1^i factor
        # 2. Apply gamma hyperparameter - aligns with paper's γ global scaling
        # 3. Clamp for numerical stability
        #
        # Raw eigenvalues from SVD can be very large (100-5000 for 65k-dimensional FLUX latents)
        # Without normalization, clamping to [1e-3, 1.0] would saturate all values at upper bound

        # Step 1: Normalize by the maximum eigenvalue to get relative scales
        # This is the paper's 1/λ_1^i normalization factor
        max_eigenval = top_eigenvalues[0].item() if len(top_eigenvalues) > 0 else 1.0

        if max_eigenval > 1e-10:
            # Scale so max eigenvalue = 1.0, preserving relative ratios
            top_eigenvalues = top_eigenvalues / max_eigenval

        # Step 2: Apply gamma and clamp to safe range
        # Gamma is the paper's tuneable hyperparameter (defaults to 1.0)
        # Clamping ensures numerical stability and prevents extreme values
        top_eigenvalues = torch.clamp(top_eigenvalues * self.gamma, 1e-3, self.gamma * 1.0)

        # Convert to fp16 for storage - now safe since eigenvalues are ~0.01-1.0
        # fp16 range: 6e-5 to 65,504, our values are well within this
        eigenvectors_fp16 = top_eigenvectors.cpu().half()
        eigenvalues_fp16 = top_eigenvalues.cpu().half()

        # Cleanup
        del weighted_centered_torch, U, S, Vh, top_eigenvectors, top_eigenvalues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return eigenvectors_fp16, eigenvalues_fp16

    def compute_for_batch(
        self,
        latents_np: np.ndarray,
        global_indices: List[int]
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Γ_b for all points in a batch of same-size latents

        Args:
            latents_np: (N, d) numpy array
            global_indices: List of global dataset indices for each latent

        Returns:
            Dict mapping global_idx -> (eigenvectors, eigenvalues)
        """
        N, d = latents_np.shape

        # Validate inputs
        if len(global_indices) != N:
            raise ValueError(f"Length mismatch: latents has {N} samples but got {len(global_indices)} indices")

        print(f"Computing CDC for batch: {N} samples, dim={d}")

        # Handle small sample cases - require minimum samples for meaningful k-NN
        MIN_SAMPLES_FOR_CDC = 5  # Need at least 5 samples for reasonable geometry estimation

        if N < MIN_SAMPLES_FOR_CDC:
            print(f"  Only {N} samples (< {MIN_SAMPLES_FOR_CDC}) - using identity matrix (no CDC correction)")
            results = {}
            for local_idx in range(N):
                global_idx = global_indices[local_idx]
                # Return zero eigenvectors/eigenvalues (will result in identity in compute_sigma_t_x)
                eigvecs = np.zeros((self.d_cdc, d), dtype=np.float16)
                eigvals = np.zeros(self.d_cdc, dtype=np.float16)
                results[global_idx] = (eigvecs, eigvals)
            return results

        # Step 1: Build k-NN graph
        print("  Building k-NN graph...")
        distances, indices = self.compute_knn_graph(latents_np)

        # Step 2: Compute bandwidth
        # Use min to handle case where k_bw >= actual neighbors returned
        k_bw_actual = min(self.k_bw, distances.shape[1] - 1)
        epsilon = distances[:, k_bw_actual]

        # Step 3: Compute Γ_b for each point
        results = {}
        print("  Computing Γ_b for each point...")
        for local_idx in tqdm(range(N), desc="  Processing", leave=False):
            global_idx = global_indices[local_idx]
            eigvecs, eigvals = self.compute_gamma_b_single(
                local_idx, latents_np, distances, indices, epsilon
            )
            results[global_idx] = (eigvecs, eigvals)

        return results


class LatentBatcher:
    """
    Collects variable-size latents and batches them by size
    """

    def __init__(self, size_tolerance: float = 0.0):
        """
        Args:
            size_tolerance: If > 0, group latents within tolerance % of size
                           If 0, only exact size matches are batched
        """
        self.size_tolerance = size_tolerance
        self.samples: List[LatentSample] = []

    def add_sample(self, sample: LatentSample):
        """Add a single latent sample"""
        self.samples.append(sample)

    def add_latent(
        self,
        latent: Union[np.ndarray, torch.Tensor],
        global_idx: int,
        latents_npz_path: str,
        shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a latent vector with automatic shape tracking

        Args:
            latent: Latent vector (any shape, will be flattened)
            global_idx: Global index in dataset
            latents_npz_path: Path to the latent cache file (e.g., "image_0512x0768_flux.npz")
            shape: Original shape (if None, uses latent.shape)
            metadata: Optional metadata dict
        """
        # Convert to numpy and flatten
        if isinstance(latent, torch.Tensor):
            latent_np = latent.cpu().numpy()
        else:
            latent_np = latent

        original_shape = shape if shape is not None else latent_np.shape
        latent_flat = latent_np.flatten()

        sample = LatentSample(
            latent=latent_flat,
            global_idx=global_idx,
            shape=original_shape,
            latents_npz_path=latents_npz_path,
            metadata=metadata
        )

        self.add_sample(sample)

    def get_batches(self) -> Dict[Tuple[int, ...], List[LatentSample]]:
        """
        Group samples by exact shape to avoid resizing distortion.

        Each bucket contains only samples with identical latent dimensions.
        Buckets with fewer than k_neighbors samples will be skipped during CDC
        computation and fall back to standard Gaussian noise.

        Returns:
            Dict mapping exact_shape -> list of samples with that shape
        """
        batches = {}
        shapes = set()

        for sample in self.samples:
            shape_key = sample.shape
            shapes.add(shape_key)

            # Group by exact shape only - no aspect ratio grouping or resizing
            if shape_key not in batches:
                batches[shape_key] = []

            batches[shape_key].append(sample)

        # If more than one unique shape, log a warning
        if len(shapes) > 1:
            logger.warning(
                "Dimension mismatch: %d unique shapes detected. "
                "Shapes: %s. Using Gaussian fallback for these samples.",
                len(shapes),
                shapes
            )

        return batches

    def _get_aspect_ratio_key(self, shape: Tuple[int, ...]) -> str:
        """
        Get aspect ratio category for grouping.
        Groups images by aspect ratio bins to ensure sufficient samples.

        For shape (C, H, W), computes aspect ratio H/W and bins it.
        """
        if len(shape) < 3:
            return "unknown"

        # Extract spatial dimensions (H, W)
        h, w = shape[-2], shape[-1]
        aspect_ratio = h / w

        # Define aspect ratio bins (±15% tolerance)
        # Common ratios: 1.0 (square), 1.33 (4:3), 0.75 (3:4), 1.78 (16:9), 0.56 (9:16)
        bins = [
            (0.5, 0.65, "9:16"),   # Portrait tall
            (0.65, 0.85, "3:4"),   # Portrait
            (0.85, 1.15, "1:1"),   # Square
            (1.15, 1.50, "4:3"),   # Landscape
            (1.50, 2.0, "16:9"),   # Landscape wide
            (2.0, 3.0, "21:9"),    # Ultra wide
        ]

        for min_ratio, max_ratio, label in bins:
            if min_ratio <= aspect_ratio < max_ratio:
                return label

        # Fallback for extreme ratios
        if aspect_ratio < 0.5:
            return "ultra_tall"
        else:
            return "ultra_wide"

    def _shapes_similar(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
        """Check if two shapes are within tolerance"""
        if len(shape1) != len(shape2):
            return False

        size1 = np.prod(shape1)
        size2 = np.prod(shape2)

        ratio = abs(size1 - size2) / max(size1, size2)
        return ratio <= self.size_tolerance

    def __len__(self):
        return len(self.samples)


class CDCPreprocessor:
    """
    High-level CDC preprocessing coordinator
    Handles variable-size latents by batching and delegating to CarreDuChampComputer
    """

    def __init__(
        self,
        k_neighbors: int = 256,
        k_bandwidth: int = 8,
        d_cdc: int = 8,
        gamma: float = 1.0,
        device: str = 'cuda',
        size_tolerance: float = 0.0,
        debug: bool = False,
        adaptive_k: bool = False,
        min_bucket_size: int = 16,
        dataset_dirs: Optional[List[str]] = None
    ):
        self.computer = CarreDuChampComputer(
            k_neighbors=k_neighbors,
            k_bandwidth=k_bandwidth,
            d_cdc=d_cdc,
            gamma=gamma,
            device=device
        )
        self.batcher = LatentBatcher(size_tolerance=size_tolerance)
        self.debug = debug
        self.adaptive_k = adaptive_k
        self.min_bucket_size = min_bucket_size
        self.dataset_dirs = dataset_dirs or []
        self.config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """
        Compute a short hash of CDC configuration for filename uniqueness.

        Hash includes:
        - Sorted dataset/subset directory paths
        - CDC parameters (k_neighbors, d_cdc, gamma)

        This ensures CDC files are invalidated when:
        - Dataset composition changes (different dirs)
        - CDC parameters change

        Returns:
            8-character hex hash
        """
        import hashlib

        # Sort dataset dirs for consistent hashing
        dirs_str = "|".join(sorted(self.dataset_dirs))

        # Include CDC parameters
        config_str = f"{dirs_str}|k={self.computer.k}|d={self.computer.d_cdc}|gamma={self.computer.gamma}"

        # Create short hash (8 chars is enough for uniqueness in this context)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:8]

    def add_latent(
        self,
        latent: Union[np.ndarray, torch.Tensor],
        global_idx: int,
        latents_npz_path: str,
        shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a single latent to the preprocessing queue

        Args:
            latent: Latent vector (will be flattened)
            global_idx: Global dataset index
            latents_npz_path: Path to the latent cache file
            shape: Original shape (C, H, W)
            metadata: Optional metadata
        """
        self.batcher.add_latent(latent, global_idx, latents_npz_path, shape, metadata)

    @staticmethod
    def get_cdc_npz_path(
        latents_npz_path: str,
        config_hash: Optional[str] = None,
        latent_shape: Optional[Tuple[int, ...]] = None
    ) -> str:
        """
        Get CDC cache path from latents cache path

        Includes optional config_hash to ensure CDC files are unique to dataset/subset
        configuration and CDC parameters. This prevents using stale CDC files when
        the dataset composition or CDC settings change.

        IMPORTANT: When using multi-resolution training, you MUST pass latent_shape to ensure
        CDC files are unique per resolution. Without it, different resolutions will overwrite
        each other's CDC caches, causing dimension mismatch errors.

        Args:
            latents_npz_path: Path to latent cache (e.g., "image_0512x0768_flux.npz")
            config_hash: Optional 8-char hash of (dataset_dirs + CDC params)
                        If None, returns path without hash (for backward compatibility)
            latent_shape: Optional latent shape tuple (C, H, W) to make CDC resolution-specific
                         For multi-resolution training, this MUST be provided

        Returns:
            CDC cache path examples:
            - With shape + hash: "image_0512x0768_flux_cdc_104x80_a1b2c3d4.npz"
            - With hash only: "image_0512x0768_flux_cdc_a1b2c3d4.npz"
            - Without hash: "image_0512x0768_flux_cdc.npz"

        Example multi-resolution scenario:
            resolution=512 → latent_shape=(16,64,48) → "image_flux_cdc_64x48_hash.npz"
            resolution=768 → latent_shape=(16,104,80) → "image_flux_cdc_104x80_hash.npz"
        """
        path = Path(latents_npz_path)

        # Build filename components
        components = [path.stem, "cdc"]

        # Add latent resolution if provided (for multi-resolution training)
        if latent_shape is not None:
            if len(latent_shape) >= 3:
                # Format: HxW (e.g., "104x80" from shape (16, 104, 80))
                h, w = latent_shape[-2], latent_shape[-1]
                components.append(f"{h}x{w}")
            else:
                raise ValueError(f"latent_shape must have at least 3 dimensions (C, H, W), got {latent_shape}")

        # Add config hash if provided
        if config_hash:
            components.append(config_hash)

        # Build final filename
        new_stem = "_".join(components)
        return str(path.with_stem(new_stem))

    def compute_all(self) -> int:
        """
        Compute Γ_b for all added latents and save individual CDC files next to each latent cache

        Returns:
            Number of CDC files saved
        """

        # Get batches by exact size (no resizing)
        batches = self.batcher.get_batches()

        # Count samples that will get CDC vs fallback
        k_neighbors = self.computer.k
        min_threshold = self.min_bucket_size if self.adaptive_k else k_neighbors

        if self.adaptive_k:
            samples_with_cdc = sum(len(samples) for samples in batches.values() if len(samples) >= min_threshold)
        else:
            samples_with_cdc = sum(len(samples) for samples in batches.values() if len(samples) >= k_neighbors)
        samples_fallback = len(self.batcher) - samples_with_cdc

        if self.debug:
            print(f"\nProcessing {len(self.batcher)} samples in {len(batches)} exact size buckets")
            if self.adaptive_k:
                print(f"  Adaptive k enabled: k_max={k_neighbors}, min_bucket_size={min_threshold}")
            print(f"  Samples with CDC (≥{min_threshold} per bucket): {samples_with_cdc}/{len(self.batcher)} ({samples_with_cdc/len(self.batcher)*100:.1f}%)")
            print(f"  Samples using Gaussian fallback: {samples_fallback}/{len(self.batcher)} ({samples_fallback/len(self.batcher)*100:.1f}%)")
        else:
            mode = "adaptive" if self.adaptive_k else "fixed"
            logger.info(f"Processing {len(self.batcher)} samples in {len(batches)} buckets ({mode} k): {samples_with_cdc} with CDC, {samples_fallback} fallback")

        # Storage for results
        all_results = {}

        # Process each bucket with progress bar
        bucket_iter = tqdm(batches.items(), desc="Computing CDC", unit="bucket", disable=self.debug) if not self.debug else batches.items()

        for shape, samples in bucket_iter:
            num_samples = len(samples)

            if self.debug:
                print(f"\n{'='*60}")
                print(f"Bucket: {shape} ({num_samples} samples)")
                print(f"{'='*60}")

            # Determine effective k for this bucket
            if self.adaptive_k:
                # Adaptive mode: skip if below minimum, otherwise use best available k
                if num_samples < min_threshold:
                    if self.debug:
                        print(f"  ⚠️  Skipping CDC: {num_samples} samples < min_bucket_size={min_threshold}")
                        print("  → These samples will use standard Gaussian noise (no CDC)")

                    # Store zero eigenvectors/eigenvalues (Gaussian fallback)
                    C, H, W = shape
                    d = C * H * W

                    for sample in samples:
                        eigvecs = np.zeros((self.computer.d_cdc, d), dtype=np.float16)
                        eigvals = np.zeros(self.computer.d_cdc, dtype=np.float16)
                        all_results[sample.global_idx] = (eigvecs, eigvals)

                    continue

                # Use adaptive k for this bucket
                k_effective = min(k_neighbors, num_samples - 1)
            else:
                # Fixed mode: skip if below k_neighbors
                if num_samples < k_neighbors:
                    if self.debug:
                        print(f"  ⚠️  Skipping CDC: {num_samples} samples < k={k_neighbors}")
                        print("  → These samples will use standard Gaussian noise (no CDC)")

                    # Store zero eigenvectors/eigenvalues (Gaussian fallback)
                    C, H, W = shape
                    d = C * H * W

                    for sample in samples:
                        eigvecs = np.zeros((self.computer.d_cdc, d), dtype=np.float16)
                        eigvals = np.zeros(self.computer.d_cdc, dtype=np.float16)
                        all_results[sample.global_idx] = (eigvecs, eigvals)

                    continue

                k_effective = k_neighbors

            # Collect latents (no resizing needed - all same shape)
            latents_list = []
            global_indices = []

            for sample in samples:
                global_indices.append(sample.global_idx)
                latents_list.append(sample.latent)  # Already flattened

            latents_np = np.stack(latents_list, axis=0)  # (N, C*H*W)

            # Compute CDC for this batch with effective k
            if self.debug:
                if self.adaptive_k and k_effective < k_neighbors:
                    print(f"  Computing CDC with adaptive k={k_effective} (max_k={k_neighbors}), d_cdc={self.computer.d_cdc}")
                else:
                    print(f"  Computing CDC with k={k_effective} neighbors, d_cdc={self.computer.d_cdc}")

            # Temporarily override k for this bucket
            original_k = self.computer.k
            self.computer.k = k_effective
            batch_results = self.computer.compute_for_batch(latents_np, global_indices)
            self.computer.k = original_k

            # No resizing needed - eigenvectors are already correct size
            if self.debug:
                print(f"  ✓ CDC computed for {len(batch_results)} samples (no resizing)")

            # Merge into overall results
            all_results.update(batch_results)

        # Save individual CDC files next to each latent cache
        if self.debug:
            print(f"\n{'='*60}")
            print("Saving individual CDC files...")
            print(f"{'='*60}")

        files_saved = 0
        total_size = 0

        save_iter = tqdm(self.batcher.samples, desc="Saving CDC files", disable=self.debug) if not self.debug else self.batcher.samples

        for sample in save_iter:
            # Get CDC cache path with config hash and latent shape (for multi-resolution support)
            cdc_path = self.get_cdc_npz_path(sample.latents_npz_path, self.config_hash, sample.shape)

            # Get CDC results for this sample
            if sample.global_idx in all_results:
                eigvecs, eigvals = all_results[sample.global_idx]

                # Convert to numpy if needed
                if isinstance(eigvecs, torch.Tensor):
                    eigvecs = eigvecs.numpy()
                if isinstance(eigvals, torch.Tensor):
                    eigvals = eigvals.numpy()

                # Save metadata and CDC results
                np.savez(
                    cdc_path,
                    eigenvectors=eigvecs,
                    eigenvalues=eigvals,
                    shape=np.array(sample.shape),
                    k_neighbors=self.computer.k,
                    d_cdc=self.computer.d_cdc,
                    gamma=self.computer.gamma
                )

                files_saved += 1
                total_size += Path(cdc_path).stat().st_size

                logger.debug(f"Saved CDC file: {cdc_path}")

        total_size_mb = total_size / 1024 / 1024
        logger.info(f"Saved {files_saved} CDC files, total size: {total_size_mb:.2f} MB")

        return files_saved


class GammaBDataset:
    """
    Efficient loader for Γ_b matrices during training
    Loads from individual CDC cache files next to latent caches
    """

    def __init__(self, device: str = 'cuda', config_hash: Optional[str] = None):
        """
        Initialize CDC dataset loader

        Args:
            device: Device for loading tensors
            config_hash: Optional config hash to use for CDC file lookup.
                        If None, uses default naming without hash.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config_hash = config_hash
        if config_hash:
            logger.info(f"CDC loader initialized (hash: {config_hash})")
        else:
            logger.info("CDC loader initialized (no hash, backward compatibility mode)")

    @torch.no_grad()
    def get_gamma_b_sqrt(
        self,
        latents_npz_paths: List[str],
        device: Optional[str] = None,
        latent_shape: Optional[Tuple[int, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Γ_b^(1/2) components for a batch of latents

        Args:
            latents_npz_paths: List of latent cache paths (e.g., ["image_0512x0768_flux.npz", ...])
            device: Device to load to (defaults to self.device)
            latent_shape: Latent shape (C, H, W) to identify which CDC file to load
                         Required for multi-resolution training to avoid loading wrong CDC

        Returns:
            eigenvectors: (B, d_cdc, d) - NOTE: d may vary per sample!
            eigenvalues: (B, d_cdc)

        Note:
            For multi-resolution training, latent_shape MUST be provided to load the correct
            CDC file. Without it, the wrong CDC file may be loaded, causing dimension mismatch.
        """
        if device is None:
            device = self.device

        eigenvectors_list = []
        eigenvalues_list = []

        for latents_npz_path in latents_npz_paths:
            # Get CDC cache path with config hash and latent shape (for multi-resolution support)
            cdc_path = CDCPreprocessor.get_cdc_npz_path(latents_npz_path, self.config_hash, latent_shape)

            # Load CDC data
            if not Path(cdc_path).exists():
                raise FileNotFoundError(
                    f"CDC cache file not found: {cdc_path}. "
                    f"Make sure to run CDC preprocessing before training."
                )

            data = np.load(cdc_path)
            eigvecs = torch.from_numpy(data['eigenvectors']).to(device).float()
            eigvals = torch.from_numpy(data['eigenvalues']).to(device).float()

            eigenvectors_list.append(eigvecs)
            eigenvalues_list.append(eigvals)

        # Stack - all should have same d_cdc and d within a batch (enforced by bucketing)
        # Check if all eigenvectors have the same dimension
        dims = [ev.shape[1] for ev in eigenvectors_list]
        if len(set(dims)) > 1:
            # Dimension mismatch! This shouldn't happen with proper bucketing
            # but can occur if batch contains mixed sizes
            raise RuntimeError(
                f"CDC eigenvector dimension mismatch in batch: {set(dims)}. "
                f"Latent paths: {latents_npz_paths}. "
                f"This means the training batch contains images of different sizes, "
                f"which violates CDC's requirement for uniform latent dimensions per batch. "
                f"Check that your dataloader buckets are configured correctly."
            )

        eigenvectors = torch.stack(eigenvectors_list, dim=0)
        eigenvalues = torch.stack(eigenvalues_list, dim=0)

        return eigenvectors, eigenvalues

    def compute_sigma_t_x(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        x: torch.Tensor,
        t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Σ_t @ x where Σ_t ≈ (1-t) I + t Γ_b^(1/2)

        Args:
            eigenvectors: (B, d_cdc, d)
            eigenvalues: (B, d_cdc)
            x: (B, d) or (B, C, H, W) - will be flattened if needed
            t: (B,) or scalar time

        Returns:
            result: Same shape as input x

        Note:
            Gradients flow through this function for backprop during training.
        """
        # Store original shape to restore later
        orig_shape = x.shape

        # Flatten x if it's 4D
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.reshape(B, -1)  # (B, C*H*W)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t.dim() == 0:
            t = t.expand(x.shape[0])

        t = t.view(-1, 1)

        # Early return for t=0 to avoid numerical errors
        if not t.requires_grad and torch.allclose(t, torch.zeros_like(t), atol=1e-8):
            return x.reshape(orig_shape)

        # Check if CDC is disabled (all eigenvalues are zero)
        # This happens for buckets with < k_neighbors samples
        if torch.allclose(eigenvalues, torch.zeros_like(eigenvalues), atol=1e-8):
            # Fallback to standard Gaussian noise (no CDC correction)
            return x.reshape(orig_shape)

        # Γ_b^(1/2) @ x using low-rank representation
        Vt_x = torch.einsum('bkd,bd->bk', eigenvectors, x)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=1e-10))
        sqrt_lambda_Vt_x = sqrt_eigenvalues * Vt_x
        gamma_sqrt_x = torch.einsum('bkd,bk->bd', eigenvectors, sqrt_lambda_Vt_x)

        # Σ_t @ x
        result = (1 - t) * x + t * gamma_sqrt_x

        # Restore original shape
        result = result.reshape(orig_shape)

        return result
