import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from loaders import BaseModelDataset, TripletDataset, RandomRotation, RandomGaussianBlur, RandomNoise, SupconDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import argparse


def get_resnet50_data(data_size="small"):

    if data_size == "small": # 50 samples for each class
        train, test = np.load("./data/5k_data/train_sep.npz"), np.load("./data/5k_data/test_sep.npz")
    else: # 100 samples for each class
        train, test = np.load("./data/10k_data/train_sep.npz"), np.load("./data/10k_data/test_sep.npz")

    transform = transforms.Compose([
        RandomRotation(degrees=30, p=0.2),
        RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=1),
        RandomNoise(mean=0, std=0.1, p=1),
    ])

    train_dataset = BaseModelDataset(train["data"], train["labels"], transform=transform)
    test_dataset = BaseModelDataset(test["data"], test["labels"], transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader

def get_supcon_data(data_size="small"):

    if data_size == "small": # 50 samples for each class
        train, test = np.load("./data/5k_data/train_sep.npz"), np.load("./data/5k_data/test_sep.npz")
    else: # 100 samples for each class
        train, test = np.load("./data/10k_data/train_sep.npz"), np.load("./data/10k_data/test_sep.npz")
    
    transform = transforms.Compose([
        RandomRotation(degrees=30, p=0.2),
        RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=1),
        RandomNoise(mean=0, std=0.1, p=1),
    ])

    train_dataset = SupconDataset(train["data"], train["labels"], transform=transform)
    test_dataset = SupconDataset(test["data"], test["labels"], transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader


def train_resnet50_classifier(device, train_loader, test_loader):

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1], nn.Flatten(), nn.Linear(2048,100)).to(device)

    optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_model = None
    patience = 5
    patience_cnt = 0

    for e in range(100):
        resnet50.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
        
        resnet50.eval()
        running_test_loss = 0.0
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet50(inputs)
                _, preds = torch.max(outputs, 1)
                eval_total += labels.shape[0]
                eval_correct += (labels == preds).sum().item()
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        avg_test_loss = running_test_loss / len(test_loader)

        interim_encoder = torch.nn.Sequential(*list(resnet50.children())[:-1])
        interim_acc = eval_register(interim_encoder, device)
        eval_acc = eval_correct / eval_total

        print(f"Epoch: {e} || Train loss: {avg_train_loss:.5f} || Test loss: {avg_test_loss:.5f} || Eval Accuracy: {eval_acc:.2f} || Unseen Accuracy: {interim_acc:.2f}")

        if best_acc < interim_acc:
            best_acc = interim_acc
            best_model = copy.deepcopy(resnet50.state_dict())
            patience_cnt = 0
        
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                print("Early stopping")
                break
                
    resnet50.load_state_dict(best_model)
    encoder = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return encoder


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Implementation of Supervised Contrastive Learning loss
        
        Args:
            temperature: Scaling parameter for cosine similarity
            base_temperature: Baseline temperature parameter
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: Hidden vector of shape [batch_size, n_views, ...].
            labels: Ground truth of shape [batch_size].
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Reshape features to [batch_size * n_views, ...]
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        features = features.view(features.shape[0], features.shape[1], -1)
        n_views = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Expand labels to match features
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat(n_views, 1)
        
        # Compute similarity matrix
        features = nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # Get mask for positive pairs
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits / self.temperature)
        log_prob = logits / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos = mask.clone()
        mask_pos[torch.eye(mask_pos.shape[0], dtype=torch.bool).to(device)] = 0
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    

def supcon_finetune(encoder, device, train_loader, test_loader):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-6)
    criterion = SupConLoss()
    best_acc = 0.0
    best_model = None
    patience = 7
    patience_counter = 0

    
    interim_acc = eval_register(encoder, device)
    print(f"Accuracy before contrastive fine-tune: {interim_acc:.2f}")

    for e in range(100):
        encoder.train()
        running_train_loss = 0.0

        for anchors, augmented, labels in train_loader:
            anchors = anchors.to(device)
            augmented = augmented.to(device)
            labels = labels.to(device)

            inputs = torch.cat([anchors, augmented], dim=0) 
            labels = torch.cat([labels, labels], dim=0) 

            optimizer.zero_grad()

            embeddings = encoder(inputs)
            loss = criterion(embeddings, labels)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        encoder.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for anchors, augmented, labels in test_loader:
                anchors = anchors.to(device)
                augmented = augmented.to(device)
                labels = labels.to(device)

                inputs = torch.cat([anchors, augmented])
                labels = torch.cat([labels, labels])

                embeddings = encoder(inputs)
                loss = criterion(embeddings, labels)
                running_test_loss += loss.item()

        interim_acc = eval_register(encoder, device)
        
        avg_train_loss = running_train_loss / len(train_loader)
        avg_test_loss = running_test_loss / len(test_loader)

        print(f"Epoch: {e} || Train loss: {avg_train_loss:.5f} || Test loss: {avg_test_loss:.5f} || Unseen Accuracy: {interim_acc:.2f}")

        if best_acc < interim_acc:
            best_acc = interim_acc
            best_model = copy.deepcopy(encoder.state_dict())
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        
    encoder.load_state_dict(best_model)
    return encoder


def eval_register(encoder, device):
    test = np.load("./data/register_sep.npz")
    references = torch.tensor(test["reference"], dtype=torch.float32).to(device)
    references = references.repeat(1, 3, 1, 1)

    register = torch.tensor(test["register"], dtype=torch.float32).to(device)
    register = register.repeat(1, 3, 1, 1)

    encoder.eval()

    with torch.no_grad():
        database = encoder(references)
        query = encoder(register)

    dists = torch.cdist(database, query, p=2).cpu().numpy()    
    min_indides = np.argmin(dists, axis=0)
    acc = np.mean(min_indides == np.arange(len(min_indides))).item()

    return acc

def main(args):

    data_size = args.data_size
    print("Resnet50 training started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_train_loader, resnet_test_loader = get_resnet50_data(data_size)
    encoder = train_resnet50_classifier(device, resnet_train_loader, resnet_test_loader)
    best_base_acc = eval_register(encoder, device)
    print(f"Best base accuracy: {best_base_acc:.2f}")

    print("Contrastive fine-tune started")
    supcon_train_loader, supcon_test_loader = get_supcon_data(data_size)
    finetuned_encoder = supcon_finetune(encoder, device, supcon_train_loader, supcon_test_loader)
    best_cont_acc = eval_register(finetuned_encoder, device)
    print(f"Best contrastive accuracy: {best_cont_acc:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=str, default="small", help="small (5k) or large (10k)")
    args = parser.parse_args()
    main(args)