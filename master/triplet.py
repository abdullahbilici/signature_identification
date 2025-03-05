import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from loaders import BaseModelDataset, TripletDataset, RandomRotation, RandomGaussianBlur, RandomNoise   
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


def get_triplet_data(data_size="small"):

    if data_size == "small": # 50 samples for each class
        train, test = np.load("./data/5k_data/train_sep.npz"), np.load("./data/5k_data/test_sep.npz")
    else: # 100 samples for each class
        train, test = np.load("./data/10k_data/train_sep.npz"), np.load("./data/10k_data/test_sep.npz")
    
    transform = transforms.Compose([
        RandomRotation(degrees=30, p=0.2),
        RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=1),
        RandomNoise(mean=0, std=0.1, p=1),
    ])

    train_dataset = TripletDataset(train["data"], train["labels"], transform=transform, num_triplets=500)
    test_dataset = TripletDataset(test["data"], test["labels"], transform=transform, num_triplets=100)

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

def contrastive_finetune(encoder, device, train_loader, test_loader):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-6)
    criterion = triplet_loss
    margin = 8

    best_acc = 0.0
    best_model = None
    patience = 7
    patience_counter = 0

    interim_acc = eval_register(encoder, device)
    print(f"Accuracy before contrastive fine-tune: {interim_acc:.2f}")

    for e in range(100):
        encoder.train()
        running_train_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            anchor_emb = encoder(anchor)
            positive_emb = encoder(positive)
            negative_emb = encoder(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb, margin)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
        
        encoder.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for anchor, positive, negative in test_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_emb = encoder(anchor)
                positive_emb = encoder(positive)
                negative_emb = encoder(negative)

                loss = criterion(anchor_emb, positive_emb, negative_emb, margin)
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

def triplet_loss(anchor, positive, negative, margin: float = 1.0):
    """
    Computes Triplet Loss
    """
    #print(anchor.shape)
    distance_positive = nn.functional.pairwise_distance(anchor, positive)
    distance_negative = nn.functional.pairwise_distance(anchor, negative)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0)
    return torch.mean(loss)
    
def main(args):

    data_size = args.data_size
    print("Resnet50 training started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_train_loader, resnet_test_loader = get_resnet50_data(data_size)
    encoder = train_resnet50_classifier(device, resnet_train_loader, resnet_test_loader)
    best_base_acc = eval_register(encoder, device)
    print(f"Best base accuracy: {best_base_acc:.2f}")

    print("Contrastive fine-tune started")
    triplet_train_loader, triplet_test_loader = get_triplet_data(data_size)
    finetuned_encoder = contrastive_finetune(encoder, device, triplet_train_loader, triplet_test_loader)
    best_cont_acc = eval_register(finetuned_encoder, device)
    print(f"Best contrastive accuracy: {best_cont_acc:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=str, default="small", help="small (5k) or large (10k)")
    args = parser.parse_args()

    main(args)