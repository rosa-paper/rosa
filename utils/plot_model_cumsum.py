from transformers import AutoModelForSequenceClassification
from utils import heatmap_cumsum_singular_vals


def main():

    model_name = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    heatmap_cumsum_singular_vals(model, out_path="figures-old/{}_cumsum_singular_vals.png".format(model_name))


if __name__ == '__main__':
    main()
