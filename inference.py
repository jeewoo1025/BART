import argparse
import csv
from glob import glob
import gc
import os
import json
from compare_mt.rouge.rouge_scorer import RougeScorer

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

from summarizer.dataloader import SumDataset
from summarizer.utils import get_logger


abs_path = "/workspace/BART/outputs/default_ver5/models/"
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def compute_rouge(target, tmp):
    score = all_scorer.score(target, tmp)
    return score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeLsum'].fmeasure


def load_json(path, index):
    path += '/test'
    with open(os.path.join(path, "%d.json"%index), "r") as f:
        data = json.load(f)

    return ' '.join(data['article']), data['abstract'][0]


def main(args: argparse.Namespace):
    logger = get_logger("inference")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    # logger.info("Load Tokenizer from facebook/bart-large")
    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    logger.info(f'[+] Load Test Dataset from "{args.dataset_path}"')
    test_dataset = SumDataset(
        mode='test',
        path=args.dataset_path,
        tokenizer=tokenizer,
        input_max_seq_len=args.input_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # load fine-tuning model 
    logger.info(f'[+] Load Model from "{args.pretrained_ckpt_path}"')
    model = BartForConditionalGeneration.from_pretrained(abs_path + args.pretrained_ckpt_path, local_files_only=True)
    # logger.info(f'[+] Load Model from facebook/bart-large-xsum')
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")

    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    total_summary_tokens = []
    for batch in tqdm(test_dataloader):
        input_doc = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # [BatchSize, SummarySeqLen]
        summary_tokens = model.generate(
            input_doc,
            attention_mask=attention_mask,
            decoder_start_token_id=tokenizer.bos_token_id,
            max_length=args.summary_max_seq_len,
            min_length=args.summary_min_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            use_cache=True,
            no_repeat_ngram_size=3
        )

        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    logger.info("[+] Start Decoding")
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]

    logger.info(f'[+] Save output to "{args.output_path}"')
    
    r1_list, r2_list, rL_list = [], [], []
    with open(args.output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(["document", "target", "predict"])

        for index, tmp in enumerate(decoded):
            doc, tgt = load_json(args.dataset_path, index)  # path, index

            # cal rouge
            r1, r2, rL = compute_rouge(tgt, tmp)      # target summary, decoded
            r1_list.append(r1)
            r2_list.append(r2)
            rL_list.append(rL)
            # logger.info('%d test json - r1 : {.%4f}, r2 : {.%4f}, r3 : {.%4f}'%(index, r1, r2, rL))

            writer.writerow([doc, tgt, tmp])

    
    logger.info('r1 : {.%4f}'%(sum(r1_list)/len(r1_list)))
    logger.info('r2 : {.%4f}'%(sum(r2_list)/len(r2_list)))
    logger.info('rL : {.%4f}'%(sum(rL_list)/len(rL_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="inference")

    parser.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
    parser.add_argument("--tokenizer", type=str, default="facebook/bart-large", help="huggingface pretrained tokenizer path")
    parser.add_argument("--dataset-path", type=str, required=True, help="finetuning dataset path")
    parser.add_argument("--output-path", type=str, required=True, help="output csv file path")
    parser.add_argument("--batch-size", type=int, default=16, help="inference batch size")
    parser.add_argument("--input-max-seq-len", type=int, default=1024, help=" max sequence length")
    parser.add_argument("--summary-max-seq-len", type=int, default=62, help="summary max sequence length")
    parser.add_argument("--summary-min-seq-len", type=int, default=11, help="summary min sequence length")
    parser.add_argument("--num-beams", type=int, default=4, help="beam size")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="beam search length penalty")
    parser.add_argument("--device", type=str, default="cuda", help="inference device")

    args = parser.parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()

    main(args)
