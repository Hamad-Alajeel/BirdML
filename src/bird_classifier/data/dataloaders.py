"""DataLoader factory functions for the bird classifier dataset."""

from datasets import DatasetDict
from torch.utils.data import DataLoader

from bird_classifier.config import DEFAULT_HYPERPARAMS
from bird_classifier.data.dataset import BirdDataset, build_label_mapping
from bird_classifier.data.transforms import get_eval_transforms, get_train_transforms


def build_dataloaders(
    dataset: DatasetDict,
    label_to_idx: dict[int, int] | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Args:
        dataset: Hugging Face DatasetDict with train/validation/test splits.
        label_to_idx: Mapping from original sparse labels to dense labels.
        batch_size: Batch size. If None, uses config default.
        num_workers: Number of DataLoader workers.

    Returns:
        train_loader, val_loader, test_loader
    """
    if label_to_idx is None:
        label_to_idx = build_label_mapping(dataset)

    if batch_size is None:
        batch_size = DEFAULT_HYPERPARAMS["batch_size"]

    train_dataset = BirdDataset(
        hf_split=dataset["train"],
        transform=get_train_transforms(),
        label_to_idx=label_to_idx,
    )

    val_dataset = BirdDataset(
        hf_split=dataset["validation"],
        transform=get_eval_transforms(),
        label_to_idx=label_to_idx,
    )

    test_dataset = BirdDataset(
        hf_split=dataset["test"],
        transform=get_eval_transforms(),
        label_to_idx=label_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader