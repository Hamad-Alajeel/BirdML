"""Image transforms for training, evaluation, and inference."""

from torchvision import transforms

from bird_classifier.config import (
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    CROP_SCALE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_BEFORE_CROP,
)


def get_train_transforms() -> transforms.Compose:
    """Return image transforms used for training."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Return image transforms used for validation, test, and inference."""

    return transforms.Compose(
        [
            transforms.Resize(RESIZE_BEFORE_CROP),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
