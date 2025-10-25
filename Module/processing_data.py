from torch.utils.data import Dataset, DataLoader, random_split
import torch

class processed_dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # HuggingFace Tokenizer 출력은 딕셔너리이므로 텐서로 변환하여 반환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
    
def divide_into_TrainAndTest(total_dataset, train_ratio):
    total_num = len(total_dataset)
    train_num = int(total_num * train_ratio)
    test_num = total_num - train_num
    
    train_dataset, test_dataset = random_split(total_dataset, [train_num, test_num])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )
    return train_loader, test_loader
    