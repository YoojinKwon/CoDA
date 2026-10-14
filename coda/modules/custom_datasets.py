# import libraries
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from PIL import Image



class TinyImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        if split == 'train':
            root = os.path.join(root, 'train')  
        elif split == 'val':
            root = os.path.join(root, 'val/images')
        else:
            raise ValueError("split should be either 'train' or 'val'")
        
        class_names = sorted(os.listdir(root))
        for idx, cls in enumerate(class_names):
            if split == 'train':
                class_dir = os.path.join(root, cls, 'images')
            elif split == 'val':
                class_dir = os.path.join(root, cls)
            else:
                raise ValueError("split should be either 'train' or 'val'")
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[cls] = idx
            for fname in os.listdir(class_dir):
                if fname.endswith('.JPEG'):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Refer to https://github.com/tanimutomo/cifar10-c-eval/blob/b4b30257ddbd737c64bc3d2e961a2c47238b5249/src/dataset.py#L16
class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, severity_level=0):
        super(CIFAR10C, self).__init__(
            root, transform=transform, target_transform=target_transform)
        data_path = os.path.join(root)
        target_path = os.path.join(
            root.rsplit('/', 1)[0], 
            'labels.npy')
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        if severity_level == 5: # Only use the most severe level
            self.data = self.data[-10000:]
            self.targets = self.targets[-10000:]

    def __getitem__(self, index):
        imgs, targets = self.data[index], self.targets[index]
        imgs = Image.fromarray(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return imgs, targets

    def __len__(self):
        return len(self.data)

class CustomDatasetFolder(datasets.DatasetFolder):
    def __init__(self, *dataset_args, **kwargs):
        super().__init__(*dataset_args, **kwargs)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = np.load(path) # [3, 224, 224] array
        sample = Image.fromarray(sample.transpose((1, 2, 0)), 'RGB') # [224, 224, 3] PIL

        if self.transform is not None:
            sample = self.transform(sample) # ToTensor
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target



class CustomDatasetFolder_CIFAR10_r0_r8(Dataset):
    def __init__(self, root, train=True, transform=None):
            """
            Args:
                root (str): Root directory where the CIFAR-10 R0-R8 dataset is stored.
                train (bool): If True, loads the training data; otherwise, loads the test data.
                transform (callable, optional): A function/transform to apply to the images.
            """
            self.root = f'{root}/cifar-10-batches-py-r0-r8'
            self.train = train
            self.transform = transform
            self.data = []
            self.targets = []

            # Load data
            if train:
                for batch_idx in range(1, 6):  # CIFAR-10 training batches
                    batch_file = os.path.join(self.root, f"data_batch_{batch_idx}")
                    data, labels = self._load_batch(batch_file)
                    self.data.append(data)
                    self.targets.extend(labels)
            else:
                test_file = os.path.join(self.root, "test_batch")
                data, labels = self._load_batch(test_file)
                self.data.append(data)
                self.targets.extend(labels)

            # Concatenate all batch data
            self.data = np.concatenate(self.data)

    def _load_batch(self, file_path):
        """Load a single batch."""
        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            data = batch[b"data"]
            labels = batch[b"labels"]
            # Reshape and transpose data to (N, H, W, C)
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        image, label = self.data[idx], self.targets[idx]

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

class CustomDatasetFolder_CIFAR100(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir (str): Directory containing the dataset.
            split (str): Either 'train' or 'test' to load the respective dataset.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = f'{data_dir}/cifar-100-python'
        self.split = split
        self.transform = transform

        # Load data from the respective pickle file
        file_path = os.path.join(self.data_dir, self.split)
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        print(data_dict.keys())
        
        self.data = data_dict[b'data']  # Assuming data is in 'data' key
        self.labels = data_dict[b'fine_labels']  # Assuming CIFAR-100 structure

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Reshape the image data to (3, 32, 32) as CIFAR is stored in a flat array
        image = np.reshape(image, (3, 32, 32)).transpose(1, 2, 0)
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, label
