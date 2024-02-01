import peftnet as pn
import numpy as np
# import transformers
from transformers import default_data_collator
from transformers import AutoTokenizer, get_scheduler
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
import torch
import torch.nn as nn

from utils.utils import get_ignore_list_glue


def get_latency(model, iters=10):
    """Get latency mean/std of a model on a given input size."""

    latency = []
    for _ in range(iters):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model = model.module.factorize() if isinstance(model, nn.DataParallel) else model.factorize()
        end.record()

        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000
        latency.append(elapsed_time)
        print(f"Factorize elapsed time (ms): {elapsed_time}")

    return np.mean(latency), np.std(latency)


def main():


    hf_model_names = ["roberta-base", "gpt2"]

    for model_name in hf_model_names:
        config = AutoConfig.from_pretrained(
            "roberta-base",
            num_labels=1,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            config=config
        )

        ignore_list = get_ignore_list_glue(model)

        peft_model = pn.RosaNet(model, rank=1, ignore_list=ignore_list)
        if torch.cuda.is_available():
            print("Using GPU")
            peft_model = peft_model.to(torch.device('cuda:0'))
        mean, std = get_latency(peft_model, iters=10)

        print(f"Model: {model_name}")
        print(f"Factorize time (ms): {mean} +/- {std}\n")




if __name__ == '__main__':
    main()