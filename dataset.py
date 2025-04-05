from torch.utils.data import Dataset
from datasets import load_from_disk

class EmotionDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk(
            r"E:\github_rep\emotion-classify-task-model\data\ChnSentiCorp"
        )
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集异常")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label


if __name__ == "__main__":
    dataset = EmotionDataset("validation")
    for data in dataset:
        print(data)
