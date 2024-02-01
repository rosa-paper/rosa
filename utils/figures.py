import os
import os.path as osp
import argparse
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
import matplotlib.pyplot as plt
import random
import numpy as np

markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

rank_color_map = dict()


def rosa_vs_lora_plot_name_func(s):
    name = {"rosa": "ROSA", "factorized": "ROSA", "lora": "LoRA", "none": "Baseline", "ia3": "IA3"}[s.split("_name")[1].split("_")[0]]
    rank = s.split("_r")[1].split("_")[0]
    lr = s.split("_l")[1].split("_")[0]
    return name + f" (r={rank}, lr={lr})"


def rosa_ablation_name_func(s):
    name = {"rosa": "ROSA", "factorized": "ROSA", "lora": "LoRA", "none": "Baseline", "ia3": "IA3"}[s.split("_name")[1].split("_")[0]]
    rank = s.split("_r")[1].split("_")[0]
    lr = s.split("_l")[1].split("_")[0]
    fact_method = s.split("_facto")[1].split("_")[0]
    fact = s.split("_fact")[1].split("_")[0]
    freq = s.split("_fa")[1].split("_")[0]
    return name + f" (r={rank}, lr={lr}, fact={fact}, freq={freq}, fmethod={fact_method})"


def default_name_func(s):
    name = {"rosa": "ROSA", "factorized": "ROSA", "lora": "LoRA", "none": "Baseline", "ia3": "IA3"}[s.split("_name")[1].split("_")[0]]
    rank = s.split("_r")[1].split("_")[0]
    lr = s.split("_l")[1].split("_")[0]
    fact_method = s.split("_facto")[1].split("_")[0]
    fact = s.split("_fact")[1].split("_")[0]
    freq = s.split("_fa")[1].split("_")[0]
    return name + f" (r={rank}, lr={lr}, fact={fact}, freq={freq}, fmethod={fact_method})"


def rosa_vs_lora_plot_color_func(s, cmap_name="tab20", n_pts=64):

    if "namenone" in s:
        return "black"

    rank_str = s.split("_r")[1].split("_")[0]
    if rank_str in rank_color_map:
        return rank_color_map[rank_str]
    else:

        # cmap = plt.get_cmap("Reds")  # "Blues", "tab20", "tab20b", "tab20c", "Reds", "YlOrRd"
        cmap = plt.get_cmap("hsv")
        values = np.linspace(0, 1, n_pts)
        colors = cmap(values)[::-1]
        random_color = colors[len(rank_color_map.items()) + 1 % len(colors)]

        # colors = list(plt.cm.colors.CSS4_COLORS.keys())
        # random_color = random.choice(colors)
        # rank_color_map[rank_str] = random_color
        rank_color_map[rank_str] = random_color
        return random_color


def random_matplotlib_color(s):
    # Get the list of available matplotlib colors
    colors = list(plt.cm.colors.CSS4_COLORS.keys())

    # Choose a random color
    random_color = random.choice(colors)

    return random_color


experiments = {
    # "rosa_vs_lora_bleu_sm": {
    #     "xlabel": "EPOCH",
    #     "ylabel": "BLEU",
    #     "xticks": list(range(1, 6, 1)),
    #     "scalar_name": "test/bleu",
    #     "plot_name_func": default_name_func,
    #     "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--" }[s.split("_name")[1].split("_")[0]],
    #     # "plot_marker_func": lambda s: random.choice(markers),
    #     "plot_color_func": rosa_vs_lora_plot_color_func,
    #     "regex": r'^(?=.*gpt2_).*'
    # },
    # "rosa_vs_lora_bleu_md": {
    #     "xlabel": "EPOCH",
    #     "ylabel": "BLEU",
    #     "xticks": list(range(1, 6, 1)),
    #     "scalar_name": "test/bleu",
    #     "plot_name_func": default_name_func,
    #     "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--"}[
    #         s.split("_name")[1].split("_")[0]],
    #     # "plot_marker_func": lambda s: random.choice(markers),
    #     "plot_color_func": rosa_vs_lora_plot_color_func,
    #     "regex": r'^(?=.*gpt2-medium).*'
    # },
    "rosa_vs_lora_cola_base": {
        "xlabel": "EPOCH",
        "ylabel": "BLEU",
        "xticks": list(range(1, 6, 1)),
        "scalar_name": "valid/matthews_correlation",
        "plot_name_func": default_name_func,
        "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--"}[
            s.split("_name")[1].split("_")[0]],
        # "plot_marker_func": lambda s: random.choice(markers),
        "plot_color_func": rosa_vs_lora_plot_color_func,
        "regex": r'^(?=.*namroberta-base).*'
    },
    # "rosa_vs_lora_cola_base_ablation": {
    #     "xlabel": "EPOCH",
    #     "ylabel": "BLEU",
    #     "xticks": list(range(1, 6, 1)),
    #     "scalar_name": "valid/matthews_correlation",
    #     "plot_name_func": default_name_func,
    #     "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--"}[
    #         s.split("_name")[1].split("_")[0]],
    #     # "plot_marker_func": lambda s: random.choice(markers),
    #     "plot_color_func": rosa_vs_lora_plot_color_func,
    #     "regex": r'^(?=.*namroberta-base).*'
    # },
    # "rosa_vs_lora_cola_large": {
    #     "xlabel": "EPOCH",
    #     "ylabel": "BLEU",
    #     "xticks": list(range(1, 6, 1)),
    #     "scalar_name": "valid/matthews_correlation",
    #     "plot_name_func": default_name_func,
    #     "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--"}[
    #         s.split("_name")[1].split("_")[0]],
    #     # "plot_marker_func": lambda s: random.choice(markers),
    #     "plot_color_func": rosa_vs_lora_plot_color_func,
    #     "regex": r'^(?=.*namroberta-large).*'
    # },
    # "rosa_vs_lora_qnli_base": {
    #     "xlabel": "EPOCH",
    #     "ylabel": "BLEU",
    #     "xticks": list(range(1, 6, 1)),
    #     "scalar_name": "valid/accuracy",
    #     "plot_name_func": default_name_func,
    #     "plot_marker_func": lambda s: {"rosa": "o-", "lora": "x--", "none": "s-", "ia3": "s--"}[
    #         s.split("_name")[1].split("_")[0]],
    #     # "plot_marker_func": lambda s: random.choice(markers),
    #     "plot_color_func": rosa_vs_lora_plot_color_func,
    #     "regex": r'^(?=.*namroberta-base).*'
    # },
}


def match_string(s, regex):
    return bool(re.match(regex, s))


def aggregate_train_loss(directory):
    for filename in os.listdir(directory):
        if filename.startswith('events.out.tfevents'):
            filepath = os.path.join(directory, filename)
            ea = EventAccumulator(filepath)
            ea.Reload()  # load all data from the event file
            if 'train/loss' in ea.Tags()['scalars']:
                for summary in ea.Scalars('train/loss'):
                    print(f"File: {filename}, Step: {summary.step}, Train Loss: {summary.value}")
            else:
                print(f"'train/loss' scalar is not found in the event file: {filename}")


def aggregate_scalar(exp_directory, scalar_name):

    scalar_x, scalar_y = list(), list()
    for filename in os.listdir(exp_directory):
        if filename.startswith('events.out.tfevents'):
            filepath = os.path.join(exp_directory, filename)
            ea = EventAccumulator(filepath)
            ea.Reload()  # load all data from the event file
            if scalar_name in ea.Tags()['scalars']:
                for summary in ea.Scalars(scalar_name):
                    scalar_x.append(float(summary.step))
                    scalar_y.append(float(summary.value))
            else:
                # print(f"'{scalar_name}' scalar is not found in the event file: {filename}")
                pass
    return scalar_x, scalar_y


def main(dir_path, output_dir="outputs"):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    dataset = osp.basename(dir_path)

    min_epochs_constraint = 5

    for exp_name, exp_config in experiments.items():
        print("\nExperiment: {} (scalar: {})\n ".format(exp_name, exp_config['scalar_name']))
        plt.clf()
        n_exps = 0
        for filename in os.listdir(dir_path):
            if match_string(filename, exp_config['regex']):
                print("\t", filename)
                exp_path = os.path.join(dir_path, filename)
                x, y = aggregate_scalar(exp_path, scalar_name=exp_config['scalar_name'])
                # print("\t\tx: {}, y: {}".format(x, y))
                if len(x) >= min_epochs_constraint:
                    plt.plot(
                        x, y,
                        exp_config['plot_marker_func'](filename),
                        label=exp_config['plot_name_func'](filename),
                        color=exp_config['plot_color_func'](filename),
                        # alpha=0.7
                    )
                n_exps += 1

        if n_exps == 0:
            continue
        if 'xlim' in exp_config:
            plt.xlim(*exp_config['xlim'])
        if 'ylim' in exp_config:
            plt.ylim(*exp_config['ylim'])

        plt.xticks(exp_config['xticks'])
        plt.xlabel(exp_config['xlabel'])
        plt.ylabel(exp_config['ylabel'])

        # Get the handles and labels of the lines
        handles, labels = plt.gca().get_legend_handles_labels()

        # Sort the handles and labels based on a specific criterion
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])

        # Unpack the sorted handles and labels
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        # Create the legend with the sorted handles and labels
        # plt.legend(sorted_handles, sorted_labels)
        plt.legend(sorted_handles, sorted_labels, loc='upper right', borderaxespad=0.5)

        # plt.legend()
        output_filename = osp.join(output_dir, f"{exp_name}_{dataset}.png")
        plt.savefig(output_filename, dpi=300)

    # # Tables
    # best_func = max
    # # e10_l0.0002_b32_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r8_leepoch_factrandom_factoequal_uFalse_t0
    # for exp_name, exp_config in experiments.items():
    #     dct = {'name': [], "trainable_params": [], exp_config['scalar_name']: []}
    #     for filename in os.listdir(dir_path):
    #         if match_string(filename, exp_config['regex']):
    #             exp_path = os.path.join(dir_path, filename)
    #             x, y = aggregate_scalar(exp_path, scalar_name=exp_config['scalar_name'])
    #             _, t = aggregate_scalar(exp_path, scalar_name="train/trainable_params")
    #             # print("\tx: {}, y: {}".format(x, y))
    #             if len(x) >= min_epochs_constraint:
    #                 dct['name'].append(exp_config['plot_name_func'](filename))
    #                 dct[exp_config['scalar_name']].append(best_func(y))
    #                 dct["trainable_params"].append(max(t))

    # Tables
    best_func = max
    for exp_name, exp_config in experiments.items():
        rows = {}
        for filename in os.listdir(dir_path):
            if match_string(filename, exp_config['regex']):
                exp_path = os.path.join(dir_path, filename)
                x, y = aggregate_scalar(exp_path, scalar_name=exp_config['scalar_name'])
                _, t = aggregate_scalar(exp_path, scalar_name="train/trainable_params")
                # print("\tx: {}, y: {}".format(x, y))
                if len(x) >= min_epochs_constraint:

                    nme = filename.split("_t")[0]  # aggregate trial
                    if nme not in rows:
                        rows[nme] = {
                            "name": exp_config['plot_name_func'](filename),
                            "trainable_params": max(t),
                            exp_config['scalar_name']: [best_func(y)],
                            "num_runs": 1
                        }
                    else:
                        rows[nme][exp_config['scalar_name']].append(best_func(y))
                        rows[nme]["trainable_params"] = max(t)
                        rows[nme]["num_runs"] += 1

        # Compute mean and std
        for nme, dct in rows.items():
            dct[exp_config['scalar_name'] + " (median)"] = np.median(dct[exp_config['scalar_name']])
            dct[exp_config['scalar_name'] + " (std)"] = np.std(dct[exp_config['scalar_name']])

        # Transform to dataframe
        output_dct = {
            "name": [],
            "trainable_params": [],
            exp_config['scalar_name'] + " (median)": [],
            exp_config['scalar_name'] + " (std)": [],
            "num_runs": []
        }

        for nme, dct in rows.items():
            output_dct['name'].append(dct['name'])
            output_dct['trainable_params'].append(dct['trainable_params'])
            output_dct[exp_config['scalar_name'] + " (median)"].append(dct[exp_config['scalar_name'] + " (median)"])
            output_dct[exp_config['scalar_name'] + " (std)"].append(dct[exp_config['scalar_name'] + " (std)"])
            output_dct["num_runs"].append(dct["num_runs"])

        df = pd.DataFrame(output_dct)
        df['trainable_params'] = df['trainable_params'].map(lambda x: x / 1e6)
        df = df.sort_values(by=['name'], ascending=True)
        df['trainable_params'] = df['trainable_params'].round(3)
        df[exp_config['scalar_name'] + " (median)"] = df[exp_config['scalar_name'] + " (median)"].round(3)
        df[exp_config['scalar_name'] + " (std)"] = df[exp_config['scalar_name'] + " (std)"].round(3)
        df = df.applymap(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)
        latex_table = df.to_latex(index=False)

        print("\nExperiment: {}\n ".format(exp_name))
        print("{}\n".format(latex_table))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fn', type=str, required=True, help="Path to the directory containing the experiments"
    )
    args = parser.parse_args()
    main(args.fn)
