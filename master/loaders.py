import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms.functional as F

class BaseModelDataset(Dataset):
    """
    Dataloader to fine-tune Resnet50 (multiclass classification)
    """
    def __init__(self, image_data, labels, transform=None):
        self.image_data = torch.tensor(image_data, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.labels[idx]

        image = image.repeat(3,1,1)

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class SupconDataset(Dataset):
    """
    Dataloader to fine-tune Resnet50 (multiclass classification)
    """
    def __init__(self, image_data, labels, transform=None):
        self.image_data = torch.tensor(image_data, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.labels[idx]

        image = image.repeat(3,1,1)

        if self.transform:
            augmented_image = self.transform(image)
            return image, augmented_image, label
        
        return image, label
    

class TripletDataset(Dataset):
    """
    Dataloader to train triplet network
    """
    def __init__(self, image_data, labels, transform=None, num_triplets=None):
        self.image_data = torch.tensor(image_data, dtype=torch.float32).repeat(1, 3, 1, 1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.num_triplets = num_triplets if num_triplets else self.image_data.size(0)

        # Generate triplets
        self.triplets = self._generate_triplets(num_triplets)

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]

        anchor = self.image_data[anchor_idx]
        positive = self.image_data[positive_idx]
        negative = self.image_data[negative_idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    
    def _generate_triplets(self, num_triplets):
        triplets = []
        unique_labels = torch.unique(self.labels).tolist()

        for _ in range(num_triplets):
            label1, label2 = random.sample(unique_labels, 2)
            positive_indices = torch.where(self.labels == label1)[0].tolist()
            negative_indices = torch.where(self.labels == label2)[0].tolist()

            anchor, positive = random.sample(positive_indices, 2)
            negative = random.choice(negative_indices)
            triplets.append([anchor, positive, negative])
        
        return triplets

class PairDataset(Dataset):
    def __init__(self, image_data, labels, transform=None, num_pairs=1000):
        self.image_data = torch.tensor(image_data, dtype=torch.float32).repeat(1, 3, 1, 1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.pairs = self._generate_pairs(num_pairs)

    def _generate_pairs(self, num_pairs):
        pairs = []
        unique_labels = torch.unique(self.labels).tolist()

        for _ in range(num_pairs):
            label = random.choice(unique_labels)
            positive_indices = torch.where(self.labels == label)[0].tolist()
            negative_indices = torch.where(self.labels != label)[0].tolist()

            if len(positive_indices) > 1:
                anchor, positive = random.sample(positive_indices, 2)
                pairs.append((anchor, positive, 1))

            if negative_indices:
                anchor = random.choice(positive_indices)
                negative = random.choice(negative_indices)
                pairs.append((anchor, negative, 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_idx, pair_idx, label = self.pairs[idx]
        anchor = self.image_data[anchor_idx]
        pair = self.image_data[pair_idx]

        if self.transform:
            anchor = self.transform(anchor)
            pair = self.transform(pair)

        return anchor, pair, torch.tensor(label, dtype=torch.float32)

class RandomRotation:
    def __init__(self, degrees, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = F.rotate(img, angle)
        return img

class RandomGaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = F.gaussian_blur(img, self.kernel_size, self.sigma)
        return img

class RandomNoise:
    def __init__(self, mean=0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Clamp values to [0, 1] range
        return img

