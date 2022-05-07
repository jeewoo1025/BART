import argparse
import os
from glob import glob
import gc
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from summarizer.dataloader import SumDataset
from summarizer.utils import get_logger
from summarizer.default import DefaultModule


def main(args: argparse.Namespace):
    logger = get_logger("train")
    
    os.makedirs(args.output_dir)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ============= Argument ================ ")
    for k,v in vars(args).items():
        logger.info(f"{k:25} : {v}")
    
    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Train Dataset from "{args.dataset_path}"')
    train_dataset = SumDataset(
        mode='train',
        path=args.dataset_path,
        tokenizer=tokenizer,
        input_max_seq_len=args.input_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len
    )

    logger.info(f'[+] Load Valid Dataset from "{args.dataset_path}"')
    valid_dataset = SumDataset(
        mode='val',
        path=args.dataset_path,
        tokenizer=tokenizer,
        input_max_seq_len=args.input_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len
    )

    # num_workers = 4*GPU갯수
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4)

    total_steps = len(train_dataloader)*args.epochs
    logger.info(f'[+] total steps "{total_steps}"')

    # load model - facebook/bart-large
    if args.pretrained_ckpt_path:
        logger.info(f'[+] Load fine-tuned Model from "{args.pretrained_ckpt_path}"')
        abs_path = "/workspace/BART/outputs/default_ver5/models/"
        model = BartForConditionalGeneration.from_pretrained(abs_path + args.pretrained_ckpt_path, local_files_only=True)
    else:
        logger.info(f'[+] Load pretrained BART model from "{args.tokenizer}"')
        model = BartForConditionalGeneration.from_pretrained(args.tokenizer)
    
    logger.info(f"[+] Use method")
    model_dir = os.path.join(args.output_dir, "models")
    lightning_module = DefaultModule(
        model, total_steps, args.max_learning_rate, args.warmup_step, model_save_dir=model_dir
    )

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    if args.wandb_project:
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir
            )
        )

    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=args.gpus      # gpu 갯수
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
    g.add_argument("--tokenizer", type=str, default="facebook/bart-large", help="huggingface pretrained tokenizer")
    g.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
    g.add_argument("--batch-size", type=int, default=128, help="training batch size")
    g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size")
    g.add_argument("--accumulate-grad-batches", type=int, default=5, help=" the number of gradident accumulation steps")
    g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
    g.add_argument("--max-learning-rate", type=float, default=2e-3, help="max learning rate")
    g.add_argument("--warmup-step", type=int, default=10000, help="warmup step")
    g.add_argument("--input-max-seq-len", type=int, default=1024, help="input max sequence length")
    g.add_argument("--summary-max-seq-len", type=int, default=62, help="summary max sequence length")
    g.add_argument("--gpus", type=int, default=1, help="the number of gpus")
    g.add_argument("--all-dropout", type=float, help="override all dropout")
    g.add_argument("--logging-interval", type=int, default=500, help="logging interval")
    g.add_argument("--evaluate-interval", type=int, default=1000, help="validation interval")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--dataset-path", type=str, help="finetuning dataset path")

    g = parser.add_argument_group("Wandb Options")
    g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
    g.add_argument("--wandb-entity", type=str, default="jeewoo25", help="wanDB entity name")
    g.add_argument("--wandb-project", type=str, help="wanDB project name")

    g = parser.add_argument_group("Method Specific Parameter")
    g.add_argument("--rdrop-alpha", type=float, default=0.7, help="rdrop alpha parameter (only used with `rdrop` method)")
    g.add_argument("--r3f-lambda", type=float, default=1.0, help="r3f lambda parameter (only used with `r3f` method)")
    g.add_argument("--rl-alpha", type=float, default=0.9984, help="rl alpha parameter (only used with `rl` method)")
    g.add_argument("--masking-rate", type=float, default=0.3, help="pretrain parameter (only used with `pretrain` method)")

    args = parser.parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()

    main(args)
