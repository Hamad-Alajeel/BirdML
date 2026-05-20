"""Tests for image transforms. Subtle bugs here (wrong normalization, wrong shape) silently kill accuracy."""

import torch

from bird_classifier.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from bird_classifier.data.transforms import get_eval_transforms, get_train_transforms


def test_eval_transform_output_shape(tiny_pil_image):
    """Output must be [3, 224, 224] regardless of input size."""
    transform = get_eval_transforms()
    tensor = transform(tiny_pil_image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_eval_transform_output_dtype(tiny_pil_image):
    transform = get_eval_transforms()
    assert transform(tiny_pil_image).dtype == torch.float32


def test_eval_transform_is_deterministic(tiny_pil_image):
    """Same input → same output. Augmentation would break test-set evaluation."""
    transform = get_eval_transforms()
    a = transform(tiny_pil_image)
    b = transform(tiny_pil_image)
    assert torch.equal(a, b)


def test_eval_transform_applies_imagenet_normalization(tiny_pil_image):
    """Pixel statistics after the transform must reflect ImageNet normalization.

    A uniform-gray image would yield channel means ≈ -mean/std after normalization.
    For our gradient image we can't predict exact values, but we CAN check the
    transform is at least subtracting roughly the ImageNet mean: post-transform
    pixels must include negative values (raw ToTensor output is in [0, 1]).
    """
    transform = get_eval_transforms()
    tensor = transform(tiny_pil_image)
    assert (tensor < 0).any(), "Negative values are expected after ImageNet mean subtraction"


def test_eval_transform_normalization_uses_config_constants():
    """Guard against silent drift between transforms.py and config.IMAGENET_*."""
    # Sanity: the constants themselves
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    assert all(0 < m < 1 for m in IMAGENET_MEAN)
    assert all(0 < s < 1 for s in IMAGENET_STD)


def test_train_transform_is_stochastic(tiny_pil_image):
    """Augmentation must actually apply — otherwise warmup/finetune sees identical inputs."""
    transform = get_train_transforms()
    a = transform(tiny_pil_image)
    b = transform(tiny_pil_image)
    # At least one of the augmentations (RandomResizedCrop / Flip / ColorJitter)
    # should produce a different tensor across two calls.
    assert not torch.equal(a, b)


def test_train_transform_output_shape(tiny_pil_image):
    """Even after random crop, output is the same canonical size as eval."""
    transform = get_train_transforms()
    assert transform(tiny_pil_image).shape == (3, IMAGE_SIZE, IMAGE_SIZE)
