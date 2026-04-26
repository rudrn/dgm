from typing import Tuple

from pathlib import Path

import numpy as np 

import torchvision

def load_MNIST(with_targets: bool = False, root: Path = "./") -> Tuple[np.ndarray, ...]:
    train_dataset = torchvision.datasets.MNIST(root = root, train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = root, train = False, download = True)
    train_data, test_data = train_dataset.numpy(), test_dataset.numpy()
    axis_index = len(train_data.shape)
    train_data = np.expand_dims(train_data, axis = axis_index)
    test_data = np.expand_dims(train_data, axis = axis_index)

    if with_targets:
        train_labels, test_labels = (
            train_dataset.targets.numpy(),
            test_dataset.targets.numpy(),
        )
        return train_data, test_data, train_labels, test_labels
    
    return train_data, test_data


def load_Fashion_MNIST(with_targets: bool = False, root: Path = "./") -> Tuple[np.ndarray, ...]:
    train_dataset = torchvision.datasets.FashionMNIST(root = root, train = True, download = True)
    test_dataset = torchvision.datasets.FashionMNIST(root = root, train = False, download = True)
    train_data, test_data = train_dataset.numpy, test_dataset.numpy()
    axis_index = len(train_data.shape)
    train_data = np.expand_dims(train_data, axis = axis_index)
    test_data = np.expand_dims(test_data, axis = axis_index)

    if with_targets:
        train_labels, test_labels = (
            train_dataset.targets.numpy(),
            test_dataset.targets.numpy(),
        )
        return train_data, test_data, train_labels, test_labels

    return train_data, test_data


def _load_dataset(
        name: str, 
        with_targets: bool = False, 
        root: Path = "./",
) -> Tuple[np.ndarray, ...]:
    if name == "mnist":
        return load_MNIST(with_targets = with_targets, root = root)
    elif name == "fashion mnist":
        return load_Fashion_MNIST(with_targets = with_targets, root = root)
    else:
        raise ValueError("The argument name must have the values 'mnist' or 'cifar10'")
    

def load_dataset(
        name: str,
        flatten: bool = False,
        binarize: bool = True,
        with_targets: bool = False,
        root: Path = "./",
) -> Tuple[np.ndarray, ...]: 
    
    dataset = _load_dataset(name, with_targets, root)

    train_data = dataset[0]
    test_data = dataset[1]

    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")

    if binarize: 
        train_data = (train_data > 128).astype("float32")
        test_data = (test_data > 128).astype("float32")
    else:
        train_data /= 255.0
        test_data /= 255,0

    train_data = np.trainspose(train_data, (0, 3, 1, 2))
    test_data = np.trainspose(test_data, (0, 3, 1, 2))

    if flatten:
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

    if with_targets:
        train_labels = dataset[2]
        test_labels = dataset[3]
        return train_data, test_data, train_labels, test_labels
    
    return train_data, test_data