import pytest
import torch

from library.loss import conditional_loss, reduce_per_sample_loss


def test_reduce_per_sample_loss_keeps_timestep_weights_paired_with_samples():
    sample_losses = torch.tensor([1.0, 4.0])
    elementwise_loss = sample_losses.view(2, 1, 1, 1).expand(2, 1, 2, 2)
    weighting = torch.tensor([4.0, 16.0]).view(2, 1, 1, 1)
    loss_weights = torch.tensor([1.0, 3.0])

    per_sample_loss = reduce_per_sample_loss(elementwise_loss, weighting, loss_weights)
    swapped_per_sample_loss = reduce_per_sample_loss(
        elementwise_loss, weighting.flip(0), loss_weights
    )

    assert per_sample_loss.shape == (2,)
    torch.testing.assert_close(per_sample_loss, torch.tensor([4.0, 192.0]))
    torch.testing.assert_close(per_sample_loss.mean(), torch.tensor(98.0))
    torch.testing.assert_close(swapped_per_sample_loss.mean(), torch.tensor(32.0))

    # This is the old, incorrect order: (B,) * (B, 1, 1, 1) creates (B, 1, 1, B).
    cartesian_loss = (
        elementwise_loss.mean((1, 2, 3)) * weighting * loss_weights
    ).mean()
    torch.testing.assert_close(cartesian_loss, torch.tensor(65.0))


def test_reduce_per_sample_loss_preserves_per_sample_gradients():
    sample_losses = torch.tensor([1.0, 4.0], requires_grad=True)
    elementwise_loss = sample_losses.view(2, 1, 1, 1).expand(2, 1, 2, 2)
    weighting = torch.tensor([4.0, 16.0]).view(2, 1, 1, 1)
    loss_weights = torch.tensor([1.0, 3.0])

    loss = reduce_per_sample_loss(elementwise_loss, weighting, loss_weights).mean()
    (gradient,) = torch.autograd.grad(loss, sample_losses)

    torch.testing.assert_close(gradient, torch.tensor([2.0, 24.0]))


def test_reduce_per_sample_loss_matches_elementwise_weighted_mse_reference():
    model_pred = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4) / 10
    target = torch.flip(model_pred, dims=(0,))
    weighting = torch.tensor([0.75, 1.25]).view(2, 1, 1, 1)

    elementwise_loss = conditional_loss(model_pred, target, "l2", "none")
    actual = reduce_per_sample_loss(elementwise_loss, weighting)

    # This is the pre-refactor SD3 and current FLUX ordering: weight first, then reduce.
    expected = (weighting * (model_pred - target).square()).reshape(2, -1).mean(1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("loss_shape", "weighting_shape"),
    [
        ((2, 3, 4, 5), (2, 1, 1, 1)),
        ((2, 3, 2, 4, 5), (2, 1, 1, 1, 1)),
        ((2, 3, 2, 4, 5), (2,)),
    ],
)
def test_reduce_per_sample_loss_supports_4d_and_5d_losses(loss_shape, weighting_shape):
    elementwise_loss = torch.arange(
        torch.tensor(loss_shape).prod(), dtype=torch.float32
    ).reshape(loss_shape)
    weighting = torch.tensor([0.5, 2.0]).reshape(weighting_shape)
    loss_weights = torch.tensor([1.5, 0.25])

    actual = reduce_per_sample_loss(elementwise_loss, weighting, loss_weights)
    expected = elementwise_loss.flatten(1).mean(1) * weighting.flatten() * loss_weights

    assert actual.shape == (loss_shape[0],)
    torch.testing.assert_close(actual, expected)


def test_reduce_per_sample_loss_accepts_masked_conditional_loss():
    model_pred = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4) / 10
    target = torch.zeros_like(model_pred)
    mask = torch.tensor([1.0, 0.0, 0.5, 1.0]).view(1, 1, 1, 4)
    weighting = torch.tensor([0.75, 1.25]).view(2, 1, 1, 1)
    loss_weights = torch.tensor([1.0, 0.5])

    elementwise_loss = conditional_loss(model_pred, target, "l2", "none") * mask
    actual = reduce_per_sample_loss(elementwise_loss, weighting, loss_weights)
    expected = (elementwise_loss * weighting).flatten(1).mean(1) * loss_weights

    torch.testing.assert_close(actual, expected)


def test_reduce_per_sample_loss_preserves_batch_one_and_uniform_weighting():
    elementwise_loss = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    weighting = torch.ones(1, 1, 1, 1)
    loss_weights = torch.tensor([0.25])

    actual = reduce_per_sample_loss(elementwise_loss, weighting, loss_weights)
    expected = elementwise_loss.mean().reshape(1) * loss_weights

    assert actual.shape == (1,)
    torch.testing.assert_close(actual, expected)


def test_reduce_per_sample_loss_supports_no_timestep_weighting():
    elementwise_loss = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    loss_weights = torch.tensor([0.5, 2.0])

    actual = reduce_per_sample_loss(elementwise_loss, None, loss_weights)
    expected = elementwise_loss.flatten(1).mean(1) * loss_weights

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)


def test_reduce_per_sample_loss_rejects_an_already_reduced_loss():
    with pytest.raises(ValueError, match="at least one non-batch axis"):
        reduce_per_sample_loss(torch.ones(2), torch.ones(2), torch.ones(2))


@pytest.mark.parametrize(
    "weighting",
    [
        torch.tensor(1.0),
        torch.ones(3, 1, 1, 1),
        torch.ones(2, 2, 1, 1),
    ],
)
def test_reduce_per_sample_loss_rejects_invalid_timestep_weight_shapes(weighting):
    with pytest.raises(ValueError, match="weighting must"):
        reduce_per_sample_loss(torch.ones(2, 3, 4, 5), weighting, torch.ones(2))


def test_reduce_per_sample_loss_rejects_non_vector_dataset_weights():
    with pytest.raises(ValueError, match=r"loss_weights must have shape \(2,\)"):
        reduce_per_sample_loss(torch.ones(2, 3, 4, 5), torch.ones(2), torch.ones(2, 1))
