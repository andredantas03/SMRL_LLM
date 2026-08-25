#from datasets import load_dataset, load_from_disk, DatasetDict
#from datasets import Dataset as DS
import os
import yaml
#from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, Dataset, DataLoader, Subset
import numpy as np
import torch
from lightning import LightningDataModule


def _seeded_subset(dataset: Dataset, num_samples: int, seed: int) -> Subset:
    """Deterministic subsample of ``dataset`` (without replacement)."""
    n = len(dataset)
    k = min(num_samples, n)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:k].tolist()
    return Subset(dataset, indices)

class DataModule(LightningDataModule):
    def tokenize_and_save_causal_dataset(
        input_path: str,
        output_path: str,
        model_name: str = "gpt2",
        text_column: str = "text",
        ):
        
        dataset = load_from_disk(input_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)   
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_fn(batch):
            return tokenizer(batch[text_column], truncation=False)

        if isinstance(dataset, DatasetDict):
            remove_columns = {
                split: dataset[split].column_names
                for split in dataset.keys()
            }
            tokenized_dataset = DatasetDict({
                split: dataset[split].map(
                    tokenize_fn,
                    batched=True,
                    remove_columns=remove_columns[split]
                )
                for split in dataset.keys()
            })
        else:
            tokenized_dataset = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names
            )

        eos = tokenizer.encode(tokenizer.eos_token)[0]

        

        if 'train' in tokenized_dataset:
            
            all_tokens_train = []
            for seq in tokenized_dataset["train"]["input_ids"]:
                all_tokens_train.extend(seq)
                all_tokens_train.append(eos)
            all_tokens_train = np.array(all_tokens_train)
            
        if 'validation' in tokenized_dataset:
            all_tokens_validation = []
            for seq in tokenized_dataset["validation"]["input_ids"]:
                all_tokens_validation.extend(seq)
                all_tokens_validation.append(eos)
            all_tokens_validation = np.array(all_tokens_validation)    

        if 'test' in tokenized_dataset:
            all_tokens_test = []
            for seq in tokenized_dataset["test"]["input_ids"]:
                all_tokens_test.extend(seq)
                all_tokens_test.append(eos)
            all_tokens_test = np.array(all_tokens_test)
        
        
        
        np.save(f"{output_path}dataset.npy", all_tokens_train)
        np.save(f"{output_path}dataset.npy", all_tokens_validation)
        np.save(f"{output_path}dataset.npy", all_tokens_test)
        return True
    def load_tokenized_dataset(input_path: str):
        return load_from_disk(input_path)
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...
    
    def setup(self, stage):
        ...
    def train_dataloader(self):
        # any iterable or collection of iterables
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        # any iterable or collection of iterables
        return DataLoader(self.val_dataset_1)

    def test_dataloader(self):
        # any iterable or collection of iterables
        return DataLoader(self.test_dataset)




def load_config(config_path: str = "configs.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)




def build_dataloaders(
    tokenized_dataset_path,
    context_length,
    batch_size: int,
    max_steps: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    eval_batch_size: int | None = None,
    limit_val_batches: int | None = None,
    val_sample_seed: int = 42,
):
    eval_bs = eval_batch_size if eval_batch_size is not None else batch_size
    dataset_path = {
            'train': tokenized_dataset_path.get('train_dataset_path'),
            'valid': tokenized_dataset_path.get('valid_dataset_path'),
            'test' : tokenized_dataset_path.get('test_dataset_path')
    }

    dataloaders = {}
    
    if dataset_path["train"] != None:
        train_dataset = LMDataset_it(
            input_path=dataset_path["train"],
            context_length=context_length,
            batch_size=batch_size,
            max_steps=max_steps,
            forever=False
            )
        
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=pin_memory,
            persistent_workers=True,
            shuffle=False,
            )
        
    if dataset_path["valid"] != None:
        validation_dataset = LMDataset(
            input_path=dataset_path["valid"],
            context_length=context_length,
            )
        if limit_val_batches is not None and limit_val_batches > 0:
            num_val_samples = int(limit_val_batches) * int(eval_bs)
            validation_dataset = _seeded_subset(
                validation_dataset,
                num_samples=num_val_samples,
                seed=val_sample_seed,
            )
        
        dataloaders["valid"] = DataLoader(
            validation_dataset,
            batch_size=eval_bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            shuffle=False,
            )
    
    if dataset_path["test"] != None:
        test_dataset = LMDataset(
            input_path=dataset_path["test"],
            context_length=context_length,
            )
        
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            shuffle=False,
            )  

    return dataloaders

def _classification_split_paths(processed_dataset_path, split: str):
    prefix = processed_dataset_path.get(f"{split}_prefix")
    if prefix:
        return f"{prefix}-docs.npy", f"{prefix}-offsets.npy", f"{prefix}-labels.npy"
    docs = processed_dataset_path.get(f"{split}_docs_path")
    offsets = processed_dataset_path.get(f"{split}_offsets_path")
    labels = processed_dataset_path.get(f"{split}_labels_path")
    if docs and offsets and labels:
        return docs, offsets, labels
    return None

def build_classification_dataloaders(
    processed_dataset_path,
    context_length,
    batch_size: int,
    shuffle_train: bool = True,
    eval_batch_size: int | None = None,
    pad_id: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    eval_bs = eval_batch_size if eval_batch_size is not None else batch_size
    
    dataloaders = {}

    for split, bs, shuffle in (
        ("train", batch_size, shuffle_train),
        ("valid", eval_bs, False),
        ("test", eval_bs, False),
    ):
        paths = _classification_split_paths(processed_dataset_path, split)
        print(paths)
        if paths is None:
            continue
        docs_path, offsets_path, labels_path = paths
        dataset = ClassificationDataset(
            docs_path=docs_path,
            offsets_path=offsets_path,
            labels_path=labels_path,
            max_length=context_length,
            pad_id=pad_id,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
    return dataloaders








class ClassificationDataset(Dataset):
    def __init__(self, docs_path, offsets_path, labels_path, max_length, pad_id=0):
        self.docs = np.load(docs_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)
        self.labels = np.load(labels_path)
        self.max_length = max_length
        self.pad_id = pad_id
        if len(self.offsets) != len(self.labels) + 1:
            raise ValueError(
                f"offsets length {len(self.offsets)} != n_labels+1 ({len(self.labels) + 1})"
            )

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        ids = np.asarray(self.docs[start:end], dtype=np.int64)

        if ids.size > self.max_length:
            ids = ids[: self.max_length]

        n = int(ids.size)
        padded = np.full(self.max_length, self.pad_id, dtype=np.int64)
        attn = np.zeros(self.max_length, dtype=np.int64)
        padded[:n] = ids
        attn[:n] = 1

        return {
            "input_ids": torch.from_numpy(padded),
            "attention_mask": torch.from_numpy(attn),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class LMDataset(Dataset):
    def __init__(self, input_path, context_length, stride=None):
        self.ds = np.load(input_path,mmap_mode='r')
        self.context_length = context_length
        self.stride = stride or context_length
        self.max_i = self.ds.shape[0] - (self.context_length + 1)
        if self.max_i < 0:
            raise ValueError("Sequência menor que context_length+1")

        self.starts = list(range(0, self.max_i + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        inp = self.ds[i : i + self.context_length ]
        tgt = self.ds[i + 1 : i + self.context_length + 1]
        return {
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "attention_mask": torch.ones((self.context_length),dtype=torch.long),
                "labels": torch.tensor(tgt, dtype=torch.long),
            }

class LMDataset_it(Dataset):
    def __init__(
        self,
        input_path,
        batch_size,
        context_length,
        max_steps,
        device: str = "cpu",
        forever: bool = True,
    ):
        self.input_path = input_path
        self.ds = None
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.max_steps = max_steps
        
        
    def _lazy_init(self):
        if self.ds is None:
            self.ds = np.load(self.input_path, mmap_mode='r')
            self.max_i = self.ds.shape[0] - (self.context_length + 1)
            if self.max_i < 0:
                raise ValueError("Sequência menor que context_length+1")
            
            self.list_ids = np.random.randint(0,self.max_i,(self.max_steps))
            
    
    def __len__(self):
        return self.max_steps
    
    def __getitem__(self, idx):
        self._lazy_init()
        batch_input_ids = []
        batch_labels = []

        for _ in range(self.batch_size):
            id = self.list_ids[idx]
            input_ids = self.ds[id:id+self.context_length]
            labels = self.ds[id+1:id+self.context_length+1]
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        
        return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.ones((self.batch_size, self.context_length), dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }
    
