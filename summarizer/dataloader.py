from pydoc import doc
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import json
import os
from .utils import get_logger


class SumDataset(Dataset):
    """Dataset for Summarization

    Attributes:
        sep_token: token to seperate utterances
        ids: id of each example
        inputs: document of each sample
        summaries: summary of each example
        input_ids: document input id tokens of each example
        input_attention_masks: document attention masks of each example
        summary_input_ids: summary input id tokens of each example
        summary_attention_masks: summary attention masks of each example
    """
    def __init__(
        self,
        mode: str,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        input_max_seq_len: int,
        summary_max_seq_len: int
    ):
        """
        필요한 변수들 선언

        Args:
            mode : train, val, test
            path : dataset path (ex. ../xsum)
            tokenizer: tokenizer to tokenize summary string
            input_max_seq_len: max sequence length of input document
            summary_max_seq_len: max sequence length of summary
            summary_min_seq_len : min sequence length of summary
            use_summary: whether to use summary data or not (should be False for inference)
        """
        super().__init__()

        # path 생성
        dir_path = path + '/' + mode
        self.mode = mode
        self.dir_path = dir_path

        self.length = len(os.listdir(self.dir_path))
        self.tokenizer = tokenizer
        self.input_max_seq_len = input_max_seq_len
        self.summary_max_seq_len = summary_max_seq_len

        
    def __len__(self) -> int:
        return self.length


    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        """
        index번째 data를 return함
            - batch_encode_plus : https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html
        """
        with open(os.path.join(self.dir_path, "%d.json"%index), "r") as f:
            data = json.load(f)

        # input document
        article = data["article"]
        src_txt = self.tokenizer.sep_token.join(article)        # sep token으로 join
        src = self.tokenizer.batch_encode_plus([src_txt], padding="max_length", max_length=self.input_max_seq_len, return_tensors="pt", truncation=True)
        src_input_ids = src["input_ids"].squeeze(0)
        src_attention_mask = src["attention_mask"].squeeze(0)

        # summary
        abstract = data["abstract"]
        summary_txt = " ".join(abstract)   # sep token으로 join
        summary = self.tokenizer.batch_encode_plus([summary_txt], padding="max_length", max_length=self.summary_max_seq_len, return_tensors="pt", truncation=True)
        summary_input_ids = summary["input_ids"].squeeze(0)
        summary_attention_mask = summary["attention_mask"].squeeze(0)

        item = {
            "input_ids": src_input_ids, 
            "attention_mask": src_attention_mask,
            "decoder_input_ids": summary_input_ids,
            "decoder_attention_mask": summary_attention_mask
        }

        return item
