import pickle
import random
import time
from csv import writer

import re
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import transformers
import yaml
from enum import Enum
import numpy as np


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def test_latency(model, device='cpu', inp=torch.randn(32, 3, 224, 224), iters=100):
    """Test latency of the argument model."""

    if not isinstance(device, torch.device):
        assert device in ['cpu', 'cuda:0']

    torch_device = torch.device(device)
    model = model.to(torch_device)
    dummy_input = inp.to(torch_device)
    latency = np.zeros((iters, 1))

    with torch.no_grad():
        if device == 'cpu':
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            for rep in range(iters):
                start = time.time()
                _ = model(dummy_input)
                elapsed = time.time() - start
                latency[rep] = elapsed

        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            with torch.no_grad():
                for rep in range(iters):
                    starter.record()
                    _ = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    latency[rep] = elapsed

    return np.mean(latency), np.std(latency)


def get_params(rmodel):
    n_params = 0
    for name, param in rmodel.named_parameters():
        n_params += param.numel()
    return n_params


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def saveckpt(model, epoch, optimizer):
    pass


def get_yaml_dict(yaml_path="conf_clm.yaml"):
    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_experiment_name(configs, abbrevs=None, mode='dict'):
    assert mode in ['str', 'dict'], f"Invalid mode: {mode}. Must be either 'str' or 'dict'."
    if abbrevs is None:
        abbrevs = {}

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs)
        else:
            i = 1
            while i <= len(key):
                if key[:i] not in abbrevs:
                    abbrevs[key[:i]] = str(value).replace(" ", "").replace(",", "_").replace("[", "").replace("]", "")
                    break
                i += 1

                if i == len(key) + 1:
                    raise ValueError("Could not find a suitable abbreviation for key: {}".format(key))

    if mode == 'str':
        return '_'.join(['{}{}'.format(k, v) for k, v in abbrevs.items()])


    return abbrevs


def get_latency(model, device='cpu', inp=torch.randn(32, 3, 224, 224), iters=2):
    """Test latency of the argument model."""

    if not isinstance(device, torch.device):
        assert device in ['cpu', 'cuda:0']

    torch_device = torch.device(device)
    model = model.to(torch_device)
    dummy_input = inp.to(torch_device)
    latency = np.zeros((iters, 1))

    with torch.no_grad():
        if device == 'cpu':
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            for rep in range(iters):
                start = time.time()
                _ = model(dummy_input)
                elapsed = time.time() - start
                latency[rep] = elapsed

        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            with torch.no_grad():
                for rep in range(iters):
                    starter.record()
                    _ = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    latency[rep] = elapsed

    return np.mean(latency), np.std(latency)


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.pts = 0

    def add(self, val, n=1):
        self.sum += val
        self.pts += n

    @property
    def value(self):
        if self.pts == 0:
            return 0
        return self.sum / self.pts


def write2csv(row: list, output_path: str, write_mode='a'):
    with open(output_path, write_mode) as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()


class LatencyReport:
    def __init__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.latencies = dict()
        self.stoppers = list()

    def report(self):
        # str = "Latency report:\n"
        strs = ["{} {:4d} ms".format(name, int(latency)) for name, latency in self.latencies.items()]
        strs = " | ".join(strs)
        return strs

    def start(self):
        self.starter.record()
        self.stoppers.append(torch.cuda.Event(enable_timing=True))

    def stop(self, name="Unk"):
        self.stoppers[-1].record()
        torch.cuda.synchronize()
        self.latencies[name] = self.starter.elapsed_time(self.stoppers[-1])
        self.stoppers.append(torch.cuda.Event(enable_timing=True))


class CudaMemoryTracker:
    def __init__(self):
        self.memory_allocated = {
            # "start": torch.cuda.memory_allocated(),
        }

        self.memory_reserved = {
            # "start": torch.cuda.memory_reserved(),
        }

    def track(self, name="Unk"):
        self.memory_allocated[name] = torch.cuda.memory_allocated()
        self.memory_reserved[name] = torch.cuda.memory_reserved()

    def report(self):
        strs = ["{} {:4d} MB".format(name, int(mem / (1024 * 1024))) for name, mem in self.memory_allocated.items()]
        strs = " | ".join(strs)
        return strs


def preprocess_function(examples, tokenizer, dataset_name="eli5", max_length=512):
    """Concat all questions/answers into one text and tokenize them afterwards."""

    if dataset_name == "eli5":
        return tokenizer([" ".join(x) for x in examples["answers.text"]])
    elif dataset_name == "e2e_nlg":
        output = tokenizer(
            ["Input: {} Output: {} {}".format(x, y, tokenizer.eos_token) for x, y in
             zip(examples['meaning_representation'], examples['human_reference'])],
            max_length=max_length,
            truncation=True,
        )
        output["labels"] = output["input_ids"].copy()
        return output
    else:
        raise NotImplementedError


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "axb": ("sentence1", "sentence2"),
    "axg": ("premise", "hypothesis"),
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1", "choice2", "question"),  # TODO: figure out how to set this up
    "multirc": ("paragraph", "question", "answer"),  # TODO: figure out how to set this up
    "record": ("passage", "query", "answer"),  # TODO: figure out how to set this up
    "wic": ("word", "sentence1", "sentence2"),
    "wsc.fixed": ("text", "span1_text", "span2_text"),
}


def preprocess_function_mlm(example, tokenizer, task_name="cola", max_length=512):
    # tokenize the texts according to the keys for each glue task
    text_keys = task_to_keys[task_name]

    if task_name == "wic":
        # take word spans (start1, start2, end1, end2)
        # and surround those words in the sentences with
        # special tokens
        # then tokenize the sentences
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]

        start1, start2, end1, end2 = example["start1"], example["start2"], example["end1"], example["end2"]

        sentence1 = sentence1[:start1] + "<t>" + sentence1[start1:end1] + "</t>" + sentence1[end1:]
        sentence2 = sentence2[:start2] + "<t>" + sentence2[start2:end2] + "</t>" + sentence2[end2:]

        output = tokenizer(sentence1, sentence2, max_length=max_length, truncation=True, padding="max_length")
    # tokenize the texts, passing two arguments to the tokenizer
    # if the task has two inputs. otherwise just one
    elif text_keys[1] is not None:
        # pad to max length
        output = tokenizer(example[text_keys[0]], example[text_keys[1]], max_length=max_length, truncation=True,
                           padding="max_length")
    else:
        output = tokenizer(example[text_keys[0]], max_length=max_length, truncation=True, padding="max_length")

    # output["labels"] is just "label" for mlm task
    output["labels"] = example["label"]

    return output


def group_texts_old(examples, block_size=128):
    # Concatenate all texts across batches. {ids: [List_1, .., List_N]} => [*List_1, ..., *List_N]
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    result = {
        column_name: [column_vals[i: i + block_size] for i in range(0, total_length, block_size)]
        for column_name, column_vals in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_function_old(examples, tokenizer, dataset_name="eli5", max_length=512):
    """Concat all questions/answers into one text and tokenize them afterwards."""

    if dataset_name == "eli5":
        return tokenizer([" ".join(x) for x in examples["answers.text"]])
    elif dataset_name == "e2e_nlg":
        output = tokenizer(
            [" ".join([x, y]) for x, y in zip(examples['meaning_representation'], examples['human_reference'])],
            max_length=max_length,
            truncation=True,
        )
        return output
    else:
        raise NotImplementedError


def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print("Found nan in {}".format(name))
            return True
    return False


def get_ignore_list_e2e(model):
    ignore_list = []
    for name, layer in model.named_modules():
        if 'attention' not in name:
            ignore_list.append(name)
    return ignore_list


def get_ignore_list_glue(model):
    ignore_list = []
    for name, layer in model.named_modules():
        if 'attention' not in name:
            ignore_list.append(name)
    return ignore_list


def set_seeds(seed=42):
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Necessary for reproducibility on CUDA. Might impact performance.
    torch.backends.cudnn.benchmark = False  # If set to True, it can introduce randomness for certain operations on CUDA


def refactorize(model: nn.Module, optimizer: torch.optim):
    """Refactorize model and update optimizer and scheduler accordingly

    Args:
        model: peft model to refactorize
        optimizer: optimizer to update

    Returns:
        model, optimizer
    """

    # (Re)factorize model
    model = model.module.factorize() if isinstance(model, nn.DataParallel) else model.factorize()

    # New optimizer
    opt_cls = optimizer.__class__
    lr = optimizer.param_groups[0]['lr']
    del optimizer
    optimizer = opt_cls(
        model.parameters(),
        lr=lr
    )

    return model, optimizer


def heatmap_cumsum_singular_vals(model, regular_expression=".*attention", out_path="figures-old/cumsum_singular_vals.png"):
    """Plot heatmap of cumulative sum of singular values in each layer of the model matching the regular expression."""
    cumsum_singular_vals = {}

    if torch.cuda.is_available():
        print("Using CUDA.")
        model.to("cuda:0")

    for name, module in model.named_modules():
        if bool(re.match(regular_expression, name)):
            if isinstance(module, nn.Linear) or isinstance(module, transformers.modeling_utils.Conv1D):
                print("Decomposing `{}`: {}".format(name, module))
                _, s, _ = torch.svd(module.weight.data)
                s_norm = s / torch.sum(s)
                cumsum_singular_vals[name] = torch.cumsum(s_norm, dim=0).cpu().numpy()

    # Check if any layers matched
    if not cumsum_singular_vals:
        print("No layers matched the regular expression.")
        return

    # Convert the dictionary to a 2D NumPy array
    max_length = max(len(vals) for vals in cumsum_singular_vals.values())
    array_data = np.zeros((len(cumsum_singular_vals), max_length))
    row_labels = []

    for i, (name, vals) in enumerate(cumsum_singular_vals.items()):
        array_data[i, :len(vals)] = vals
        row_labels.append(name)

    # Plot
    plt.figure(figsize=(15, 8))
    plt.imshow(array_data, interpolation='none', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.yticks(np.arange(len(row_labels)), labels=[])
    plt.xlabel('Singular Value Index')
    plt.ylabel('Layer')
    plt.title('Cumulative Sum of Singular Values in Model Layers')
    plt.savefig(out_path)



