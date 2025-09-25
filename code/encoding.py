"""
Run an encoding model on BOLD data.

https://gallantlab.org/voxelwise_tutorials/_auto_examples/shortclips/03_plot_wordnet_model.html
https://gallantlab.org/voxelwise_tutorials/_auto_examples/shortclips/06_plot_banded_ridge_model.html
"""

import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from glob import glob

import h5py
import numpy as np
import torch
from constants import CONV_TRS, RUNS, SUBS_STRANGERS, TR
from himalaya.backend import set_backend
from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer,
                                   MultipleKernelRidgeCV)
from himalaya.scoring import correlation_score_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.path import Path
from util.subject import (get_bold, get_conv, get_timinglog_boxcars,
                          get_transcript_features)

# from util.atlas import Atlas

SPACES = {
    "llm_split": {
        "word_stats": [
            "prod_screen",
            "comp_screen",
            "prod_word_onset",
            "prod_word_rate",
            "prod_phoneme_rate",
            "comp_word_onset",
            "comp_word_rate",
            "comp_phoneme_rate",
        ],
        "llm": ["prod_lexical_emb", "comp_lexical_emb"],
    },
    "llm_nosplit": {
        "word_stats": [
            "prod_screen",
            "comp_screen",
            "word_onset",
            "word_rate",
            "phoneme_rate",
        ],
        "llm": ["lexical_emb"],
    },
    "joint_split": {
        "task": ["prod_screen", "comp_screen"],
        "word_stats": [
            "prod_word_onset",
            "prod_word_rate",
            "prod_phoneme_rate",
            "comp_word_onset",
            "comp_word_rate",
            "comp_phoneme_rate",
        ],
        "spectral": ["prod_spectral_emb", "comp_spectral_emb"],
        "articulation": ["prod_phoneme_emb", "comp_phoneme_emb"],
        "llm": ["prod_lexical_emb", "comp_lexical_emb"],
    },
    "joint_split_nollm": {
        "task": ["prod_screen", "comp_screen"],
        "word_stats": [
            "prod_word_onset",
            "prod_word_rate",
            "prod_phoneme_rate",
            "comp_word_onset",
            "comp_word_rate",
            "comp_phoneme_rate",
        ],
        "spectral": ["prod_spectral_emb", "comp_spectral_emb"],
        "articulation": ["prod_phoneme_emb", "comp_phoneme_emb"],
    },
    "joint_nosplit": {
        "task": ["prod_screen", "comp_screen"],
        "word_stats": [
            "word_onset",
            "word_rate",
            "phoneme_rate",
        ],
        "spectral": ["spectral_emb"],
        "articulation": ["phoneme_emb"],
        "llm": ["lexical_emb"],
    },
}


class SplitDelayer(BaseEstimator, TransformerMixin):
    def __init__(self, delays=None):
        self.delays = delays

    def fit(self, X, y=None):
        # print("Delayer fit", X.shape)
        X = self._validate_data(X, dtype="numeric")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        # print("Delayer transform", X.shape)
        # split X, apply delayer to each, then restitch
        n_samples = X.shape[0]
        if n_samples % 120 != 0:
            raise ValueError(f"n_samples is not divisible by 120, {n_samples}")
        n_splits = n_samples // 120  # number of trials
        return np.vstack([self._delay(x) for x in np.split(X, n_splits, axis=0)])

    def _delay(self, X):
        # Borrowed from himalaya
        if self.delays is None:
            return X

        n_samples, n_features = X.shape

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)), dtype=X.dtype)
        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[: -abs(delay), beg:end] = X[abs(delay) :]

        return X_delayed


def get_regressors(sub_id: int, modelname: str, split: bool = True):
    """
    Will always return the following regressors:
        ['word_onset', 'word_rate', 'phoneme_rate', 'phoneme_emb', 'lexical_emb', 'spectral_emb', 'prod_screen', 'comp_screen']
    and split them into prod/comp if split=True (prod_screen and comp_screen are always split).
    """
    conv = get_conv(sub_id)
    dfemb = get_transcript_features(sub_id, modelname=modelname)
    dfphone = get_transcript_features(sub_id, modelname="articulatory")

    regressors = defaultdict(list)

    # For each trial, create a feature for each TR
    for (run, trial), subdf in dfemb.groupby(["run", "trial"]):
        word_rate = np.zeros((CONV_TRS, 1))
        phoneme_rate = np.zeros((CONV_TRS, 1))
        word_onset = np.zeros((CONV_TRS, 1))
        dim_lexemb = len(subdf.iloc[0].embedding)
        lexical_emb = np.zeros((CONV_TRS, dim_lexemb))

        subdf_phones = dfphone[(dfphone.run == run) & (dfphone.trial == trial)]
        dim_phoemb = len(subdf_phones.iloc[0].embedding)
        phone_emb = np.zeros((CONV_TRS, dim_phoemb))

        # Go through one TR at a time and find words that fall within this TR
        # average their embeddings, get num of words, etc
        for t in range(CONV_TRS):
            start_s = t * TR
            end_s = start_s + TR
            mask = (subdf.start <= end_s) & (subdf.start > start_s)
            if mask.any():
                word_onset[t] = 1
                word_rate[t] = mask.sum()
                lexical_emb[t] = np.vstack(subdf[mask].embedding).mean(
                    axis=0, keepdims=True
                )

            mask = (subdf_phones.start <= end_s) & (subdf_phones.start > start_s)
            if mask.any():
                phoneme_rate[t] = mask.sum()
                phone_emb[t] = np.vstack(subdf_phones[mask].embedding).mean(
                    axis=0, keepdims=True
                )

        regressors["word_onset"].append(word_onset)
        regressors["word_rate"].append(word_rate)
        regressors["phoneme_rate"].append(phoneme_rate)
        regressors["phoneme_emb"].append(phone_emb)
        regressors["lexical_emb"].append(lexical_emb)

        audpath = Path(root="data/stimuli", datatype="spectrogram", ext=".npy")
        audpath.update(conv=conv, run=run, trial=trial)
        filename = glob(audpath.starstr(["conv", "datatype"]))[0]
        audio_emb = np.load(filename)
        regressors["spectral_emb"].append(audio_emb)
        # print(audio_emb[-1].shape, audpath)

    # remove any uninformative dimensions for syntactic only
    if modelname == "syntactic":
        values = np.concatenate(regressors["lexical_emb"])
        missingMask = values.sum(0) > 0
        if not np.all(missingMask):
            print(
                "[WARNING] contains features with all 0s",
                missingMask.sum(),
                len(missingMask),
            )
            values = values[:, missingMask]
        regressors["lexical_emb"] = values

    # add additional nuisance variables
    prod_boxcar, _, _ = get_timinglog_boxcars(sub_id)
    prod_mask = prod_boxcar.astype(bool)

    if split:
        split_regressors = defaultdict(list)
        for regressor, values in regressors.items():
            if isinstance(values, list) and len(values) > 1:
                values = np.concatenate(values)
            else:
                values = np.asarray(values)

            prod_values = np.zeros_like(values)
            prod_values[prod_mask] = values[prod_mask]
            split_regressors[f"prod_{regressor}"] = prod_values

            comp_values = np.zeros_like(values)
            comp_values[~prod_mask] = values[~prod_mask]
            split_regressors[f"comp_{regressor}"] = comp_values

        regressors = split_regressors

    regressors["prod_screen"] = prod_boxcar.reshape(-1, 1)
    regressors["comp_screen"] = 1 - prod_boxcar.reshape(-1, 1)

    return regressors


def build_regressors(
    subject: int, modelname: str, spaces: dict = None, split: bool = True
):
    """Bundle regressors into feature spaces for banded ridge regression."""
    regressors = get_regressors(subject, modelname, split=split)

    X = []
    start = 0
    slices = {}
    for feature_space, items in spaces.items():
        x_features = []
        for item in items:
            values = regressors[item]
            if isinstance(values, list) and len(values) > 1:
                values = np.concatenate(values)
            else:
                values = np.asarray(values)
            x_features.append(values)
            # print(subject, modelname, feature_space, type(values), values.shape)
        x_features = np.hstack(x_features)
        X.append(x_features)
        slices[feature_space] = slice(start, start + x_features.shape[1])
        start += x_features.shape[1]

    X = np.hstack(X)
    return X, slices


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    """Build the pipeline"""

    # Set up modeling pipeline
    default_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SplitDelayer(delays=[0, 1, 2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )
    task_pipeline = make_pipeline(
        SplitDelayer(delays=[0, 1, 2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )
    space2pipe = {"task": task_pipeline}

    # Make kernelizer
    kernelizers_tuples = [
        (name, space2pipe.get(name, default_pipeline), slice_)
        for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=n_jobs)

    params = dict(
        alphas=alphas, progress_bar=verbose, diagonalize_method="svd", n_iter=100
    )
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    return pipeline


def encoding(
    sub_id: int,
    space: str,
    lang_model: str,
    cache: str,
    save_weights: bool,
    save_preds: bool,
    verbose: bool,
    n_jobs: int,
    alphas: np.ndarray,
) -> dict:

    spaces = SPACES[space]
    split = "nosplit" not in space
    X, features = build_regressors(sub_id, lang_model, spaces=spaces, split=split)
    X = X.astype(np.float32)
    feature_names = list(features.keys())
    slices = list(features.values())

    delayer = SplitDelayer(delays=[0, 1, 2, 3, 4, 5])
    pipeline = build_model(feature_names, slices, alphas, verbose, n_jobs)

    Y_bold = get_bold(sub_id, cache=cache)

    # glasser = Atlas.glasser2016()
    # Y_bold = glasser.vox_to_parc(Y_bold)

    results = defaultdict(list)
    run_ids = np.repeat(RUNS, CONV_TRS * 2)
    # kfold = PredefinedSplit(run_ids)
    # train on just 2 runs and test on 3 (to compare with story):
    kfold = PredefinedSplit(np.array([-1] * 480 + [0] * (1200 - 480)))
    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
        print(
            datetime.now(),
            f"F{k+1}",
            X_train.shape,
            Y_train.shape,
            X_test.shape,
            Y_test.shape,
        )

        # define prod/comp TR masks
        prod_mask = X_test[:, 0:1]
        prod_mask = delayer.fit_transform(prod_mask).any(-1)
        comp_mask = X_test[:, 1:2]
        comp_mask = delayer.fit_transform(comp_mask).any(-1)
        results["cv_prod_mask"].append(prod_mask)
        results["cv_comp_mask"].append(comp_mask)

        # Fit model and generate predictions
        pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])  # type: ignore
        pipeline.fit(X_train, Y_train)
        Y_preds = pipeline.predict(X_test, split=True)

        # inclusive model evaluation
        scores_prod = correlation_score_split(
            Y_test[prod_mask], Y_preds[:, prod_mask, :]
        )
        scores_comp = correlation_score_split(
            Y_test[comp_mask], Y_preds[:, comp_mask, :]
        )
        results["cv_scores_prod"].append(scores_prod.numpy(force=True))
        results["cv_scores_comp"].append(scores_comp.numpy(force=True))

        # exclusive model evaluation
        prod_only_mask = prod_mask & ~comp_mask
        comp_only_mask = comp_mask & ~prod_mask
        scores_prod = correlation_score_split(
            Y_test[prod_only_mask], Y_preds[:, prod_only_mask, :]
        )
        scores_comp = correlation_score_split(
            Y_test[comp_only_mask], Y_preds[:, comp_only_mask, :]
        )
        results["cv_scores_prod_exclusive"].append(scores_prod.numpy(force=True))
        results["cv_scores_comp_exclusive"].append(scores_comp.numpy(force=True))

        enc_model = pipeline["multiplekernelridgecv"]  # type: ignore
        results["cv_alphas"].append(enc_model.best_alphas_.numpy(force=True))
        if save_weights:
            Xfit = pipeline["columnkernelizer"].get_X_fit()
            weights = enc_model.get_primal_coef(Xfit)

            weights_prod = weights[-2]
            weights_prod_delay = weights_prod.reshape(-1, 6, weights_prod.shape[-1])
            weights_prod = weights_prod_delay.mean(1)
            # weights_prod = torch.linalg.vector_norm(weights_prod_delay, dim=0)

            weights_comp = weights[-1]
            weights_comp_delay = weights_comp.reshape(-1, 6, weights_comp.shape[-1])
            weights_comp = weights_comp_delay.mean(1)
            # weights_comp = torch.linalg.vector_norm(weights_comp_delay, dim=0)

            results["cv_weights_prod"].append(weights_prod)
            results["cv_weights_comp"].append(weights_comp)

        if save_preds:
            results["cv_preds"].append(Y_preds.numpy(force=True))

    # stack across folds
    result = {k: np.stack(v) for k, v in results.items()}
    return result


def main(subject: list[int], model: str, cache: str, suffix: str, cuda: int, **kwargs):
    if cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    for sub_id in subject:
        print(datetime.now(), "Start", sub_id)
        result = encoding(sub_id, space=model, cache=cache, **kwargs)

        desc = kwargs.get("lang_model")
        pklpath = Path(
            root=f"results/encoding_{cache}{suffix}",
            sub=f"{sub_id:03d}",
            datatype=model,
            desc=desc,
            ext=".hdf5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in result.items():
                f.create_dataset(name=key, data=value)

    print(datetime.now())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", nargs="+", type=int, default=SUBS_STRANGERS)
    parser.add_argument("-m", "--model", type=str, default="joint_split")
    parser.add_argument(
        "-lm", "--lang-model", type=str, default="model-gpt2-2b_layer-24"
    )
    parser.add_argument("-j", "--n-jobs", type=int, default=1)
    parser.add_argument("--cache", type=str, default="default_task")
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    _args = parser.parse_args()
    _args.alphas = np.logspace(0, 12, 13)

    if task_id := os.environ.get("SLURM_ARRAY_TASK_ID") and not len(_args.subject):
        portion = int(task_id) - 1
        subject_chunks = np.array_split(np.array(SUBS_STRANGERS), 4)
        subject_subset = subject_chunks[portion]
        print("\nUsing SLRUM ID", task_id, subject_subset)
        _args.subject = subject_subset

    main(**vars(_args))
