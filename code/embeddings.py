"""Add embeddings to an events file that has words.

"""

from glob import glob
from os import makedirs

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from util.path import Path

# short names for long model names
HFMODELS = {
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-2b": "facebook/opt-1.3b",
    "opt-3b": "facebook/opt-2.7b",
    "opt-7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
    "olmo-1b": "allenai/OLMo-1B",
    "olmo-7b": "allenai/OLMo-7B",
    "olmo-7b-chat": "allenai/OLMo-7B-Instruct",
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-1.1-2b-it",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-1.1-7b-it",
    "llama-7b": "models/llama/7b",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    # "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "neo-125m": "EleutherAI/gpt-neo-125m",
    "neo-1b": "EleutherAI/gpt-neo-1.3B",
    "neo-3b": "EleutherAI/gpt-neo-2.7B",
    # "gptj-6b": "EleutherAI/gpt-j-6b",
    # "gpt2-82m": "distilgpt2",
    "gpt2-124m": "gpt2",
    "gpt2-355m": "gpt2-medium",
    "gpt2-774m": "gpt2-large",
    "gpt2-2b": "gpt2-xl",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}


def get_model_metadata():
    records = []
    for modelname, hfmodelname in HFMODELS.items():
        config = AutoConfig.from_pretrained(hfmodelname, trust_remote_code=True)
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        # if modelname.startswith("gpt2"):
        #     breakpoint()
        if "max_position_embeddings" in config.to_dict():
            max_positions = config.max_position_embeddings
        elif "n_positions" in config.to_dict():
            max_positions = config.n_positions
        else:
            max_positions = config.max_sequence_length
        # params = config.num_parameters()
        records.append((modelname, n_layers, hidden_size, max_positions))

    df = pd.DataFrame(
        records, columns=["model", "num_layers", "hidden_size", "max_positions"]
    )
    print(df)
    df.to_csv("mats/models.csv", index=False)


def main(modelname: str, device: str = "cpu", layer: int = None):
    """Reimplemented to share model across iterations and for models with big context size."""

    hfmodelname = HFMODELS[modelname]

    # Find transcripts
    transpath = Path(root="data/stimuli", datatype="whisperx", conv="*", ext=".csv")
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    if not len(files):
        raise FileNotFoundError("No files found for: " + search_str)
    print(f"Found {len(files)} transcripts")

    tokenizer_args = dict(
        trust_remote_code=True, token="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"
    )
    if "gpt2" in hfmodelname or "opt" in hfmodelname:
        tokenizer_args["add_prefix_space"] = True
    model_args = dict(
        trust_remote_code=True, token="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"
    )
    if "Llama-2" in hfmodelname:
        # https://huggingface.co/docs/transformers/main/model_doc/llama2#usage-tips
        # Setting config.pretraining_tp to a value different than 1 will
        # activate the more accurate but slower computation of the linear
        # layers, which should better match the original logits.
        model_args["pretraining_tp"] = 0
        # model_args[torch_dtype] = torch.float16

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname, **tokenizer_args)
    model = AutoModelForCausalLM.from_pretrained(hfmodelname, **model_args)
    if layer is None:
        layer = model.config.num_hidden_layers // 2

    print(
        f"Model : {hfmodelname}"
        f"\nLayers: {model.config.num_hidden_layers} ({layer})"
        f"\nEmbDim: {model.config.hidden_size}"
        f"\nTokens: {tokenizer}"
        # f"\nCxtLen: {model.config.max_position_embeddings}"
    )
    model = model.eval()
    model = model.to(device)
    staticEmbeddingTable = model.lm_head.weight

    metrics = {}
    dirname = f"model-{modelname}_layer-{layer}"

    for tpath in tqdm(files):
        df = pd.read_csv(tpath)
        df.dropna(subset="word", inplace=True)

        # Tokenize input
        df.insert(0, "word_idx", df.index.values)
        # df["hftoken"] = df.word.apply(tokenizer.tokenize)
        # manually add space:
        df["hftoken"] = df.word.apply(lambda x: tokenizer.tokenize(" " + x))
        df = df.explode("hftoken", ignore_index=True)
        df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

        # Set up input
        # tokenids = [tokenizer.bos_token_id] + df.token_id.tolist()
        tokenids = [1] + df.token_id.tolist()
        batch = torch.tensor([tokenids], dtype=torch.long, device=device)

        # Static embedding lookup
        if layer == 0:
            with torch.no_grad():
                states = staticEmbeddingTable[batch][0, 1:].numpy(force=True)
        else:
            # Run through model
            with torch.no_grad():
                output = model(batch, labels=batch, output_hidden_states=True)
                states = output.hidden_states[layer][0, 1:].numpy(force=True)

                loss = output.loss
                logits = output.logits[0]

                logits_order = logits.argsort(descending=True, dim=-1)
                ranks = torch.eq(logits_order[:-1], batch[:, 1:].T).nonzero()[:, 1]

                probs = logits[:-1, :].softmax(-1)
                true_probs = probs[0, batch[0, 1:]]

                entropy = torch.distributions.Categorical(probs=probs).entropy()

            df["rank"] = ranks.numpy(force=True)
            df["true_prob"] = true_probs.numpy(force=True)
            df["entropy"] = entropy.numpy(force=True)

            metrics[tpath] = dict(
                top1_acc=(df["rank"] == 0).mean(), perplexity=loss.exp().item()
            )

        df["embedding"] = [e for e in states]
        epath = Path.frompath(tpath)
        epath.update(root="data/stimuli", datatype=dirname, suffix=None, ext="pkl")
        epath.mkdirs()
        df.to_pickle(epath)

    outdir = f"results/{modelname}"
    makedirs(outdir, exist_ok=True)
    summary_df = pd.DataFrame(metrics).T
    summary_df.to_csv(f"{outdir}/performance.csv")

    print(summary_df.describe())


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="gpt2-2b")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    args.device = torch.device("cpu")
    if torch.cuda.is_available() and not args.force_cpu:
        args.device = torch.device("cuda", args.cuda)
    else:
        print("WARNING: using cpu only")

    main(args.model, device=args.device, layer=args.layer)
    # get_model_metadata()
