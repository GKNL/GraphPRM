from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(test_set_path, *args, **kwargs):
    env_dir = Path(__file__).parent
    test_ds = JsonlMathDataset(env_dir / test_set_path)
    train_ds = JsonlMathDataset(env_dir / "dataset/train.jsonl")
    return train_ds, test_ds


class JsonlMathDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader: 
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["problem"], "answer": x["answer"]}
        # if "solution" not in x:
        #     return {"question": x["problem"], "answer": x["answer"]}
        # else:
        #     return {"question": x["problem"], "answer": x["solution"]}
