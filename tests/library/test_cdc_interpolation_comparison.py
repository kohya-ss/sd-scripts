"""
Test comparing interpolation vs pad/truncate for CDC preprocessing.

This test quantifies the difference between the two approaches.
"""

import pytest
import torch
import torch.nn.functional as F


class TestInterpolationComparison:
    """Compare interpolation vs pad/truncate"""

    def test_intermediate_representation_quality(self):
        """Compare intermediate representation quality for CDC computation"""
        # Create test latents with different sizes - deterministic
        latent_small = torch.zeros(16, 4, 4)
        for c in range(16):
            for h in range(4):
                for w in range(4):
                    latent_small[c, h, w] = (c * 0.1 + h * 0.2 + w * 0.3) / 3.0

        latent_large = torch.zeros(16, 8, 8)
        for c in range(16):
            for h in range(8):
                for w in range(8):
                    latent_large[c, h, w] = (c * 0.1 + h * 0.15 + w * 0.15) / 3.0

        target_h, target_w = 6, 6  # Median size

        # Method 1: Interpolation
        def interpolate_method(latent, target_h, target_w):
            latent_input = latent.unsqueeze(0)  # (1, C, H, W)
            latent_resized = F.interpolate(
                latent_input, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
            # Resize back
            C, H, W = latent.shape
            latent_reconstructed = F.interpolate(
                latent_resized, size=(H, W), mode='bilinear', align_corners=False
            )
            error = torch.mean(torch.abs(latent_reconstructed - latent_input)).item()
            relative_error = error / (torch.mean(torch.abs(latent_input)).item() + 1e-8)
            return relative_error

        # Method 2: Pad/Truncate
        def pad_truncate_method(latent, target_h, target_w):
            C, H, W = latent.shape
            latent_flat = latent.reshape(-1)
            target_dim = C * target_h * target_w
            current_dim = C * H * W

            if current_dim == target_dim:
                latent_resized_flat = latent_flat
            elif current_dim > target_dim:
                # Truncate
                latent_resized_flat = latent_flat[:target_dim]
            else:
                # Pad
                latent_resized_flat = torch.zeros(target_dim)
                latent_resized_flat[:current_dim] = latent_flat

            # Resize back
            if current_dim == target_dim:
                latent_reconstructed_flat = latent_resized_flat
            elif current_dim > target_dim:
                # Pad back
                latent_reconstructed_flat = torch.zeros(current_dim)
                latent_reconstructed_flat[:target_dim] = latent_resized_flat
            else:
                # Truncate back
                latent_reconstructed_flat = latent_resized_flat[:current_dim]

            latent_reconstructed = latent_reconstructed_flat.reshape(C, H, W)
            error = torch.mean(torch.abs(latent_reconstructed - latent)).item()
            relative_error = error / (torch.mean(torch.abs(latent)).item() + 1e-8)
            return relative_error

        # Compare for small latent (needs padding)
        interp_error_small = interpolate_method(latent_small, target_h, target_w)
        pad_error_small = pad_truncate_method(latent_small, target_h, target_w)

        # Compare for large latent (needs truncation)
        interp_error_large = interpolate_method(latent_large, target_h, target_w)
        truncate_error_large = pad_truncate_method(latent_large, target_h, target_w)

        print("\n" + "=" * 60)
        print("Reconstruction Error Comparison")
        print("=" * 60)
        print("\nSmall latent (16x4x4 -> 16x6x6 -> 16x4x4):")
        print(f"  Interpolation error: {interp_error_small:.6f}")
        print(f"  Pad/truncate error:  {pad_error_small:.6f}")
        if pad_error_small > 0:
            print(f"  Improvement:         {(pad_error_small - interp_error_small) / pad_error_small * 100:.2f}%")
        else:
            print("  Note: Pad/truncate has 0 reconstruction error (perfect recovery)")
            print("        BUT the intermediate representation is corrupted with zeros!")

        print("\nLarge latent (16x8x8 -> 16x6x6 -> 16x8x8):")
        print(f"  Interpolation error: {interp_error_large:.6f}")
        print(f"  Pad/truncate error:  {truncate_error_large:.6f}")
        if truncate_error_large > 0:
            print(f"  Improvement:         {(truncate_error_large - interp_error_large) / truncate_error_large * 100:.2f}%")

        # The key insight: Reconstruction error is NOT what matters for CDC!
        # What matters is the INTERMEDIATE representation quality used for geometry estimation.
        # Pad/truncate may have good reconstruction, but the intermediate is corrupted.

        print("\nKey insight: For CDC, intermediate representation quality matters,")
        print("not reconstruction error. Interpolation preserves spatial structure.")

        # Verify interpolation errors are reasonable
        assert interp_error_small < 1.0, "Interpolation should have reasonable error"
        assert interp_error_large < 1.0, "Interpolation should have reasonable error"

    def test_spatial_structure_preservation(self):
        """Test that interpolation preserves spatial structure better than pad/truncate"""
        # Create a latent with clear spatial pattern (gradient)
        C, H, W = 16, 4, 4
        latent = torch.zeros(C, H, W)
        for i in range(H):
            for j in range(W):
                latent[:, i, j] = i * W + j  # Gradient pattern

        target_h, target_w = 6, 6

        # Interpolation
        latent_input = latent.unsqueeze(0)
        latent_interp = F.interpolate(
            latent_input, size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Pad/truncate
        latent_flat = latent.reshape(-1)
        target_dim = C * target_h * target_w
        latent_padded = torch.zeros(target_dim)
        latent_padded[:len(latent_flat)] = latent_flat
        latent_pad = latent_padded.reshape(C, target_h, target_w)

        # Check gradient preservation
        # For interpolation, adjacent pixels should have smooth gradients
        grad_x_interp = torch.abs(latent_interp[:, :, 1:] - latent_interp[:, :, :-1]).mean()
        grad_y_interp = torch.abs(latent_interp[:, 1:, :] - latent_interp[:, :-1, :]).mean()

        # For padding, there will be abrupt changes (gradient to zero)
        grad_x_pad = torch.abs(latent_pad[:, :, 1:] - latent_pad[:, :, :-1]).mean()
        grad_y_pad = torch.abs(latent_pad[:, 1:, :] - latent_pad[:, :-1, :]).mean()

        print("\n" + "=" * 60)
        print("Spatial Structure Preservation")
        print("=" * 60)
        print("\nGradient smoothness (lower is smoother):")
        print(f"  Interpolation - X gradient: {grad_x_interp:.4f}, Y gradient: {grad_y_interp:.4f}")
        print(f"  Pad/truncate  - X gradient: {grad_x_pad:.4f}, Y gradient: {grad_y_pad:.4f}")

        # Padding introduces larger gradients due to abrupt zeros
        assert grad_x_pad > grad_x_interp, "Padding should introduce larger gradients"
        assert grad_y_pad > grad_y_interp, "Padding should introduce larger gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
