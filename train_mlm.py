import os
import os.path as osp
import math
import time
import logging
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
import evaluate as eval_lib
from transformers import default_data_collator
from transformers import AutoTokenizer, get_scheduler
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from utils.utils import get_num_params, get_experiment_name, get_latency, AverageMeter, save_object, LatencyReport, \
    CudaMemoryTracker, preprocess_function_mlm, get_ignore_list_glue, set_seeds

import peftnet as pn
import pandas as pd

wandb.login()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:.3f}'.format)

base = {
    "train": "train",
    "validation": "validation",
    "test": "validation"
}

task_to_split = {
    "mnli": {
        "train": "train",
        "validation": "validation_matched",
        "test": "validation_matched"
    },
    "qnli": base.copy(),
    "stsb": base.copy(),
    "cola": base.copy(),
    "rte": base.copy(),
    "mrpc": base.copy(),
    "sst2": base.copy(),
    "qqp": base.copy(),
    "wnli": base.copy(),
    "axb": base.copy(),
    "axg": base.copy(),
    "boolq": base.copy(),
    "cb": base.copy(),
    "copa": base.copy(),
    "multirc": base.copy(),
    "record": base.copy(),
    "wic": base.copy(),
    "wsc.fixed": base.copy(),
}


def get_dataloaders(args, tokenizer):
    # Load dataset
    assert args['dataset']['name'] in ["glue", "super_glue"], "Dataset not supported"
    assert args['dataset']['task_name'] in task_to_split.keys(), "Task not supported"

    train_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['train'],
        cache_dir=args['dataset']['cache']
    )
    test_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['test'],
        cache_dir=args['dataset']['cache']
    )
    valid_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['validation'],
        cache_dir=args['dataset']['cache']
    )

    # Filter for faster training (debug)
    num_train_pts, _ = train_dataset.shape
    train_dataset = train_dataset.select(range(int(num_train_pts * args['train']['fraction'])))

    # Apply tokenizer to dataset
    train_tokenized = train_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']
        ),
        # batched=True,
        # num_proc=4,
    )

    valid_tokenized = valid_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        # batched=True
    )

    test_tokenized = test_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        # batched=True
    )

    # Only include tokenized ids
    train_tokenized_reduced = train_tokenized.remove_columns(train_dataset.column_names)
    valid_tokenized_reduced = valid_tokenized.remove_columns(valid_dataset.column_names)
    test_tokenized_reduced = test_tokenized.remove_columns(test_dataset.column_names)

    # use default data collator
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_tokenized_reduced, shuffle=True, batch_size=args['train']['batch_size'], collate_fn=data_collator,
        pin_memory=True, num_workers=1
    )
    valid_dataloader = DataLoader(
        valid_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )

    # tweak for now since we can't evaluate on test (it's private there's no labels)
    # return train_dataloader, valid_dataloader, test_dataloader, test_dataset
    return train_dataloader, valid_dataloader, None, None


def evaluate(model, device, eval_dataloader, task="cola"):
    try:
        metric = eval_lib.load('super_glue', task)
    except KeyError:
        metric = eval_lib.load('glue', task)

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if task == "stsb":
                predictions.extend(outputs.logits.squeeze(-1).tolist())
            else:
                predictions.extend(torch.argmax(outputs.logits, dim=-1).tolist())
            references.extend(batch['labels'].tolist())

    # import pdb; pdb.set_trace()

    score = metric.compute(predictions=predictions, references=references)
    # scale score to 0-100 (score is a dict, multiply values by 100)
    score = {k: v * 100 for k, v in score.items()}
    return score


def factorize(args, model, lr_scheduler, optimizer, steps_counter, num_training_steps):
    # Mask gradients
    with torch.no_grad():
        logging.info("\n=> *** Factorizing model at step {} with factorize level {} ***\n".format(
                steps_counter, args['fnmodel']['factorize_level']
            )
        )
        model = model.module.factorize() if isinstance(model, nn.DataParallel) else model.factorize()

        # New optimizer
        opt_cls = optimizer.__class__
        del optimizer

        if "adam" in args['train']['optimizer']['name']:
            optimizer = opt_cls(
                model.parameters(),
                lr=args["train"]["lr"],
                **args['train']['optimizer']['params']
            )
        # Catch all exceptions
        else:
            optimizer = opt_cls(
                model.parameters(),
                lr=args["train"]["lr"]
            )

        # New scheduler
        if lr_scheduler is not None:
            del lr_scheduler

            # Scheduler
            n_warmup_steps = math.ceil(num_training_steps * args['train']['scheduler']['warmup_ratio'])
            lr_scheduler = get_scheduler(
                name=args['train']['scheduler']['name'],
                optimizer=optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=n_warmup_steps,
                **args['train']['scheduler']['params']
            ) if args['train']['scheduler']['name'] != "none" else None

            for i in range(steps_counter):
                lr_scheduler.step()

        return model, optimizer, lr_scheduler


def get_num_trainable_params(model):
    n_trainable_params = 0
    for name, param in model.named_parameters():
        n_trainable_params += param.numel() if param.requires_grad else 0
    return n_trainable_params


def printmodel(model):
    for name, param in model.named_parameters():
        print("Name: {} | Shape: {} | Requires grad: {}".format(name, param.shape, param.requires_grad))


def train_epoch(
        model,
        device,
        train_dataloader,
        optimizer,
        lr_scheduler,
        epoch,
        print_freq=10,
        report_latency=True,
        steps_counter=0,
        writer=None
):
    loss_average_meter = AverageMeter()
    # ppl_average_meter = AverageMeter()
    latency_report = LatencyReport()

    cuda_memory_tracker = CudaMemoryTracker()

    cuda_memory_tracker.track("[train_epoch] Initial")
    model.train()
    model.to(device)
    cuda_memory_tracker.track("[train_epoch] After model to device")

    curr_step = lambda epoch, i_step: epoch * len(train_dataloader) + i_step

    # Get trainable parameters
    n_trainable_params = get_num_trainable_params(model)

    for i_step, batch in enumerate(train_dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}
        cuda_memory_tracker.track("[train_epoch] After batch to device")

        # Masking FactorizedNet gradients
        latency_report.start()

        # get outputs from model, passing in labels as well for loss
        outputs = model(**batch)

        cuda_memory_tracker.track("[train_epoch] After forward")
        latency_report.stop(name="forward")

        if len(outputs.loss.shape) > 0:
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss
        latency_report.stop(name="loss.mean()")

        # import pdb; pdb.set_trace()
        loss.backward()
        cuda_memory_tracker.track("[train_epoch] After loss.backward()")
        latency_report.stop(name="loss.backward()")

        optimizer.step()
        cuda_memory_tracker.track("[train_epoch] After optimizer.step()")
        latency_report.stop(name="optimizer.step()")

        if lr_scheduler is not None:
            lr_scheduler.step()

        optimizer.zero_grad()
        cuda_memory_tracker.track("[train_epoch] After optimizer.zero_grad()")

        if i_step % print_freq == 0:
            logging.info(
                "[Epoch {:4d} Step {:4d}/{:4d}] | loss {:5.2f} | trainable: {:,} | lr: {:.6f}  ".format(
                    epoch, i_step, len(train_dataloader),
                    loss.item(),
                    n_trainable_params, optimizer.param_groups[0]['lr'],
                )
            )
            logging.info("Latency Report: {}".format("" if not report_latency else " | " + latency_report.report()))
            logging.info("Memory Report: {}\n".format(cuda_memory_tracker.report()))

        loss_average_meter.add(loss.item())
        steps_counter += 1

    model_fn = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(model_fn, pn.RosaNet) or isinstance(model_fn, pn.LoraNet):
        df = model_fn.get_report()
        logging.info("\n{}".format(df))
        logging.info(model_fn)

    if writer is not None:  # todo: fix this
        for i, (k, v) in enumerate(cuda_memory_tracker.memory_allocated.items()):
            writer.add_scalar("train_epoch/memory_allocated", curr_step(epoch, i))

        for i, (k, v) in enumerate(cuda_memory_tracker.memory_reserved.items()):
            writer.add_scalar("train_epoch/memory_reserved", curr_step(epoch, i))

    return {"loss": loss_average_meter.value}, optimizer, steps_counter


def train(
        args,
        cmodel,
        optimizer,
        lr_scheduler,
        train_dataloader,
        valid_dataloader,
        device, output_path,
        writer,
        curr_epoch=1,
        best_valid_metrics=None,
        cuda_memory_tracker=None,
        test_dataloader=None,
):
    # Get runtime metrics
    cuda_memory_tracker = CudaMemoryTracker() if cuda_memory_tracker is None else cuda_memory_tracker

    # Train loop
    steps_counter = 0  # will track number of gradient steps
    for i_epoch in range(curr_epoch, args["train"]["epochs"] + 1):

        cuda_memory_tracker.track("[train] Loop start")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        epoch_start_time = time.time()
        if (args['fnmodel']['name'] == "rosa" and
                args['fnmodel']['factorize_level'] == "epoch"
                and (i_epoch % args['fnmodel']['factorize_freq'] == 0) and i_epoch > 0):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            num_training_steps = args['train']['epochs'] * len(train_dataloader)
            cmodel, optimizer, lr_scheduler = factorize(
                args, cmodel, lr_scheduler, optimizer, steps_counter, num_training_steps
            )
            cuda_memory_tracker.track("[train] After epoch level sample trainable")

            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            factorize_end_time = time.time()
            factorize_elapsed = (start.elapsed_time(end) / 1000)
            print("Factorize time: {:5.2f}".format(factorize_elapsed))

        # Train
        cuda_memory_tracker.track("[train] Before train epoch")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        train_metrics, optimizer, steps_counter = train_epoch(
            model=cmodel,
            device=device,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=i_epoch,
            print_freq=args["logging"]["print_freq"],
            writer=writer,
            steps_counter=steps_counter
        )

        train_end_time = time.time()

        # Evaluate
        valid_metrics = evaluate(cmodel, device, valid_dataloader, task=args['dataset']['task_name'])
        valid_end_time = time.time()

        # Test
        logging.info("=> Computing test metrics...")
        test_metrics = evaluate(
            cmodel, device, test_dataloader, task=args['dataset']['task_name']
        ) if test_dataloader is not None else None

        # Log metrics
        epoch_str = \
            ("=> [Epoch {:4d}/{:4d} factorize(s): {:5.2f} train (s): {:5.2f} valid (s): {:5.2f}] | ".format(
                i_epoch, args["train"]["epochs"],
                (factorize_end_time - epoch_start_time),
                (train_end_time - epoch_start_time),
                (valid_end_time - epoch_start_time)
            )
             + " | ".join([f"Train {k}: {v:.2f}" for k, v in train_metrics.items()]) + " | "
             + " | ".join([f"Valid {k}: {v:.2f}" for k, v in valid_metrics.items()]) + " | "
             + (" | ".join(
                [f"Test {k}: {v:.2f}" for k, v in test_metrics.items()]) if test_metrics is not None else ""
                )
             )

        logging.info(epoch_str)
        logging.info("Memory Report: {}".format(cuda_memory_tracker.report()))

        # Ckpt object
        try:
            model_state_dict = cmodel.module.state_dict()
        except AttributeError:
            model_state_dict = cmodel.state_dict()
        ckpt = {
            'epoch': i_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model_state_dict,
            'torchrandom_state': torch.get_rng_state(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'config': args
        }

        # Save model checkpoint
        # set `metric_key` depending on the glue task
        # cola uses `matthews_correlation`
        # stsb uses `spearmanr`
        # all others use `accuracy`
        metric_key = 'accuracy'
        if args['dataset']['task_name'] == 'cola':
            metric_key = 'matthews_correlation'
        elif args['dataset']['task_name'] == 'stsb':
            metric_key = 'spearmanr'

        if best_valid_metrics is None or valid_metrics[metric_key] > best_valid_metrics[metric_key]:
            best_valid_metrics = valid_metrics
            torch.save(ckpt, osp.join(output_path, "model_best.pth"))
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        elif i_epoch % 1 == 0 or i_epoch == args["train"]["epochs"]:
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        # Log to tensorboard
        for m in valid_metrics.keys():
            if m is not None:
                writer.add_scalar("valid/{}".format(m), valid_metrics[m], i_epoch)
                wandb.log({"valid/{}".format(m): valid_metrics[m]}, step=i_epoch)

        for m in train_metrics.keys():
            if m is not None:
                writer.add_scalar("train/{}".format(m), train_metrics[m], i_epoch)
                wandb.log({"train/{}".format(m): train_metrics[m]}, step=i_epoch)

        if test_metrics is not None:
            for m in test_metrics.keys():
                if m is not None:
                    writer.add_scalar("test/{}".format(m), test_metrics[m], i_epoch)
                    wandb.log({"test/{}".format(m): train_metrics[m]}, step=i_epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], i_epoch)
        wandb.log({"train/lr": optimizer.param_groups[0]['lr']}, step=i_epoch)

        # Get trainable parameters
        n_trainable_params = 0
        for name, param in cmodel.named_parameters():
            n_trainable_params += param.numel() if param.requires_grad else 0
        writer.add_scalar("train/trainable_params", n_trainable_params, i_epoch)
        wandb.log({"train/trainable_params": n_trainable_params}, step=i_epoch)

        # Log best valid metrics
        logging.info("=> Best valid metrics: " + " | ".join(
            [f"{k}: {v:.2f}" for k, v in best_valid_metrics.items()]
        ))

        logging.info("END of Epoch\n=========\n")


@hydra.main(version_base=None, config_path="configs/conf_mlm", config_name="mlm")
def main(cfg: DictConfig):
    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))

    for t in range(max(1, args["runs"])):

        # Set diff seeds for each run
        if args['seed'] > 0:
            set_seeds(int(args['seed'] + t))

        exp_name = get_experiment_name(
            {"train": args["train"], "model": args["model"], "fnmodel": args["fnmodel"], "trial": t, "seed": args['seed']}
        )
        folder_name = "_".join(["{}{}".format(k, v) for k, v in exp_name.items()])

        dct_latest, dct_best = None, None

        output_path = osp.join(args['output']['path'], args['dataset']['name'], args['dataset']['task_name'],
                               folder_name)
        if not osp.exists(output_path):
            os.makedirs(output_path)
            save_object(args, osp.join(output_path, 'args.pkl'))
            print("=> Running Experiment: `{}`".format(folder_name))

        elif not osp.exists(osp.join(output_path, 'model_latest.pth')):
            print("=> Running Experiment: `{}`".format(folder_name))

        else:  # Experiment already exists
            dct_latest = torch.load(osp.join(output_path, 'model_latest.pth'))
            dct_best = torch.load(osp.join(output_path, 'model_best.pth'))
            if dct_latest['epoch'] >= args['train']['epochs']:
                print("=> Experiment `{}` already exists. (Latest @ epoch {})".format(
                    folder_name, dct_latest['epoch']
                ))
                continue

            else:
                print("=> Running Experiment: `{}`".format(folder_name))

        writer = SummaryWriter(log_dir=output_path)
        run = wandb.init(
            mode="disabled",
            project="rosa-mlm",
            name=folder_name,
            config=args,
        )

        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        logging.getLogger().addHandler(logging.FileHandler(osp.join(output_path, "logging.txt")))

        cuda_memory_tracker = CudaMemoryTracker()
        cuda_memory_tracker.track('[main] Initial')

        tokenizer = AutoTokenizer.from_pretrained(
            args['model']['name'],
            cache_dir=args['dataset']['cache'],
            use_fast=True
        )

        # train_dataloader, valid_dataloader, valid_dataset, test_dataset = get_dataloaders(args, tokenizer)
        train_dataloader, valid_dataloader, test_dataloader, test_dataset = get_dataloaders(args, tokenizer)

        is_regression = args['dataset']['task_name'] == 'stsb'

        if is_regression:
            num_labels = 1
        else:
            num_labels = len(train_dataloader.dataset.unique('labels'))

        config = AutoConfig.from_pretrained(
            args['model']['name'],
            cache_dir=args['dataset']['cache'],
            num_labels=num_labels,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args['model']['name'],
            config=config,
            cache_dir=args['dataset']['cache']
        )

        # import pdb; pdb.set_trace()
        logging.info("Model:\n{}".format(model))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cuda_memory_tracker.track('[main] Created base model loaded to cpu')

        logging.info("=> Computing baseline latency...")
        model.to(device)
        torch.cuda.empty_cache()
        cuda_memory_tracker.track('[main] Moved baseline model loaded to gpu')

        # Factorize model either using ROSA or LORA
        logging.info("=> Using {} model ...".format(args['fnmodel']['name'].lower()))
        ignore_list = get_ignore_list_glue(model) if args['fnmodel']['ignore_list'] else None
        cmodel = {
            "rosa": pn.RosaNet,
            "lora": pn.LoraNet,
            "ia3": pn.IA3Net,
            "none": lambda x, **kwargs: x
        }[args['fnmodel']['name'].lower()](model, ignore_list=ignore_list, **args['fnmodel']['params'])
        logging.info("Factorized Model:\n{}".format(cmodel))
        model_fn = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(model_fn, pn.RosaNet) or isinstance(model_fn, pn.LoraNet):
            df = model_fn.get_report()
            logging.info("\n{}".format(df))
            logging.info(model_fn)

        cuda_memory_tracker.track('[main] Created factorized model loaded to cpu')

        opt = {
            "adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD
        }[args['train']['optimizer']['name']]

        cuda_memory_tracker.track('[main] Created optimizer on cpu')

        # Resume training
        if dct_latest is not None:
            cmodel.load_state_dict(dct_latest['model_state_dict'])
            cmodel.to(device)

            torch.cuda.empty_cache()
            cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

            if "adam" in args['train']['optimizer']['name']:
                optimizer = opt(
                    cmodel.parameters(),
                    lr=args["train"]["lr"],
                    **args['train']['optimizer']['params']
                )
            # Catch all exceptions
            else:
                optimizer = opt(
                    cmodel.parameters(),
                    lr=args["train"]["lr"]
                )

            cuda_memory_tracker.track('[main] Optimizer passed network parameters')

            optimizer.load_state_dict(dct_latest['optimizer_state_dict'])
            curr_epoch = dct_latest['epoch'] + 1
            curr_best_valid_metrics = dct_best['valid_metrics']
            logging.info("=> Resuming training from from epoch {}".format(dct_latest['epoch']))

        else:
            curr_epoch = 0
            curr_best_valid_metrics = None
            cmodel.to(device)
            torch.cuda.empty_cache()
            cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

            if "adam" in args['train']['optimizer']['name']:
                optimizer = opt(
                    cmodel.parameters(),
                    lr=args["train"]["lr"],
                    **args['train']['optimizer']['params']
                )
            # Catch all exceptions
            else:
                optimizer = opt(
                    cmodel.parameters(),
                    lr=args["train"]["lr"]
                )

            cuda_memory_tracker.track('[main] Optimizer passed network parameters')
            logging.info("=> Starting training from scratch ...")

        # Scheduler
        n_training_steps = args['train']['epochs'] * len(train_dataloader)
        n_warmup_steps = math.ceil(n_training_steps * args['train']['scheduler']['warmup_ratio'])
        lr_scheduler = get_scheduler(
            name=args['train']['scheduler']['name'],
            optimizer=optimizer,
            num_training_steps=n_training_steps,
            num_warmup_steps=n_warmup_steps,
            **args['train']['scheduler']['params']
        ) if args['train']['scheduler']['name'] != "none" else None

        # Parallelize the model
        if torch.cuda.device_count() >= 1:
            logging.info("=> Using {} GPU(s)".format(torch.cuda.device_count()))
            cmodel = nn.DataParallel(cmodel)

        # Training
        logging.info(cuda_memory_tracker.report())
        train(
            args=args,
            cmodel=cmodel,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            device=device,
            output_path=output_path,
            writer=writer,
            curr_epoch=curr_epoch,
            best_valid_metrics=curr_best_valid_metrics,
            cuda_memory_tracker=cuda_memory_tracker,
            test_dataloader=None,
        )

        print("=> Experiment: `{}` Succeeded".format(folder_name))


if __name__ == '__main__':
    main()
