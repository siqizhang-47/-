"""Main runner for the MIRI-IES imputation benchmark.

Single experiment:
    CUDA_VISIBLE_DEVICES=0 python experiments/ies/run_ies.py \
        --config experiments/ies/configs/ies_default.yaml \
        --method miri --mechanism mcar --missing-rate 0.4 --seed 0

Full sweep (all methods x mechanisms x rates x seeds in the config):
    CUDA_VISIBLE_DEVICES=0 python experiments/ies/run_ies.py \
        --config experiments/ies/configs/ies_default.yaml --run-all
"""

from __future__ import annotations

# Pin GPU 0 *before* importing torch (see implementation plan section 3).
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import random
import time

import numpy as np
import yaml

# Make the repo root importable regardless of the current working directory,
# so ``from src.imputer import ...`` and ``experiments.ies.*`` both resolve.
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402  (imported after CUDA_VISIBLE_DEVICES is set)

from experiments.ies.prepare_ies import (  # noqa: E402
    load_ies_data,
    stratified_month_hour_split,
    standardize_by_train,
)
from experiments.ies.masks import make_test_mask  # noqa: E402
from experiments.ies.baselines import initialize_missing, run_selected_method  # noqa: E402
from experiments.ies import metrics as M  # noqa: E402


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "")


def run_single(config, method, mechanism, missing_rate, seed, out_root):
    """Run one (method, mechanism, rate, seed) experiment.

    Returns the result dict (also written to disk as JSON).  Baseline failures
    are captured and returned with ``status="failed"`` instead of raising, so a
    sweep is never aborted by a single broken dependency.
    """
    feature_cols = config["data"]["feature_cols"]
    raw_dir = os.path.join(out_root, "raw")
    imp_dir = os.path.join(out_root, "imputations")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(imp_dir, exist_ok=True)

    tag = f"{method}_{mechanism}_r{_rate_tag(missing_rate)}_seed{seed}"
    json_path = os.path.join(raw_dir, f"{tag}.json")

    set_all_seeds(seed)
    t0 = time.time()

    # 1. load data
    df_meta, df_x = load_ies_data(config)
    X_raw = df_x.to_numpy(dtype=np.float32)

    # 2. stratified split
    train_idx, val_idx, test_idx = stratified_month_hour_split(
        df_meta,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=seed,
    )

    # 3. standardize with train statistics only
    X_std, mean, std = standardize_by_train(X_raw, train_idx)

    # 4. build the test-set missing mask
    X_test = X_std[test_idx]
    M_test = make_test_mask(X_test, mechanism, missing_rate, seed, feature_cols, config)

    # 5. concat: complete train + masked test
    X_train = X_std[train_idx]
    M_train = np.ones_like(X_train, dtype=np.float32)
    X_concat_true = np.concatenate([X_train, X_test], axis=0)
    M_concat = np.concatenate([M_train, M_test], axis=0)

    # 6. initialize missing entries
    X_concat_init = initialize_missing(
        X_concat_true,
        M_concat,
        seed=seed,
        strategy=config["miri"]["init_strategy"],
        noise_std=config["miri"]["init_noise_std"],
    )

    n_train = len(train_idx)
    result = {
        "method": method,
        "mechanism": mechanism,
        "missing_rate": float(missing_rate),
        "seed": int(seed),
        "n_train": int(n_train),
        "n_test": int(len(test_idx)),
        "n_features": int(X_std.shape[1]),
        "missing_fraction_actual": float((M_test == 0).mean()),
    }

    # optional MIRI checkpoint path
    if config.get("output", {}).get("save_checkpoints", False):
        ckpt_dir = os.path.join(out_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        config["_checkpoint_path"] = os.path.join(ckpt_dir, f"{tag}.pt")

    # 7. impute
    try:
        X_concat_imp, extra = run_selected_method(
            method=method,
            X_init=X_concat_init,
            M=M_concat,
            X_true=X_concat_true,
            config=config,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - record and continue
        import traceback
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["runtime_sec"] = round(time.time() - t0, 3)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[FAILED] {tag}: {result['error']}")
        return result

    # 8. slice test rows
    X_test_imp = X_concat_imp[n_train:]

    # 9. metrics (standardized scale)
    mmd_max = config["metrics"]["mmd_max_samples"]
    sigma = config["metrics"].get("rbf_sigma")
    mae, rmse = M.masked_mae_rmse(X_test, X_test_imp, M_test)
    masked_mmd = M.masked_value_mmd(X_test, X_test_imp, M_test, max_samples=mmd_max, seed=seed, sigma=sigma)
    joint_mmd = M.rbf_mmd(X_test, X_test_imp, max_samples=mmd_max, seed=seed, sigma=sigma)
    corr_err = M.corr_error(X_test, X_test_imp)
    relation_errs = M.key_relation_errors(X_test, X_test_imp)
    per_var = M.per_variable_errors(X_test, X_test_imp, M_test, feature_cols)

    result["status"] = "ok"
    result["metrics"] = {
        "mae_std": mae,
        "rmse_std": rmse,
        "masked_mmd": masked_mmd,
        "joint_mmd": joint_mmd,
        "corr_error": corr_err,
        **relation_errs,
    }
    result["per_variable"] = per_var
    if "mmd_list" in extra:
        result["miri_mmd_list"] = extra["mmd_list"]
        result["miri_mi_list"] = extra["mi_list"]
    result["runtime_sec"] = round(time.time() - t0, 3)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 10. save arrays for plotting
    if config.get("output", {}).get("save_imputations", True):
        npz_path = os.path.join(imp_dir, f"{tag}.npz")
        np.savez(
            npz_path,
            X_test_true=X_test,
            X_test_imp=X_test_imp,
            M_test=M_test,
            mean=mean,
            std=std,
            feature_cols=np.array(feature_cols, dtype=object),
            method=method,
            mechanism=mechanism,
            missing_rate=missing_rate,
            seed=seed,
        )

    print(
        f"[OK] {tag}  MAE={mae:.4f} RMSE={rmse:.4f} "
        f"joint_mmd={joint_mmd:.5f} corr_err={corr_err:.4f} "
        f"({result['runtime_sec']}s)"
    )
    return result


def _rates_for(config, mechanism):
    return {
        "mcar": config["missing"]["mcar_rates"],
        "mar": config["missing"]["mar_rates"],
        "mnar": config["missing"]["mnar_rates"],
    }[mechanism]


def run_all(config, out_root, methods=None, mechanisms=None, seeds=None):
    methods = methods or config["experiment"]["methods"]
    mechanisms = mechanisms or config["missing"]["mechanisms"]
    seeds = seeds or config["experiment"]["seeds"]

    jobs = []
    for mechanism in mechanisms:
        for rate in _rates_for(config, mechanism):
            for method in methods:
                for seed in seeds:
                    jobs.append((method, mechanism, rate, seed))

    print(f"Total experiments to run: {len(jobs)}")
    for i, (method, mechanism, rate, seed) in enumerate(jobs, 1):
        tag = f"{method}_{mechanism}_r{_rate_tag(rate)}_seed{seed}"
        json_path = os.path.join(out_root, "raw", f"{tag}.json")
        if os.path.exists(json_path):
            print(f"[{i}/{len(jobs)}] skip existing {tag}")
            continue
        print(f"[{i}/{len(jobs)}] running {tag}")
        run_single(config, method, mechanism, rate, seed, out_root)


def build_argparser():
    p = argparse.ArgumentParser(description="MIRI-IES imputation benchmark runner")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None, help="override data.excel_path")
    p.add_argument("--method", default=None)
    p.add_argument("--mechanism", default=None, choices=["mcar", "mar", "mnar"])
    p.add_argument("--missing-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-all", action="store_true")
    # MIRI quick-test overrides
    p.add_argument("--miri-max-rounds", type=int, default=None)
    p.add_argument("--miri-max-epochs", type=int, default=None)
    p.add_argument("--miri-ode-steps", type=int, default=None)
    p.add_argument("--miri-batch-size", type=int, default=None)
    return p


def apply_overrides(config, args):
    if args.data:
        config["data"]["excel_path"] = args.data
    if args.miri_max_rounds is not None:
        config["miri"]["max_rounds"] = args.miri_max_rounds
    if args.miri_max_epochs is not None:
        config["miri"]["max_epochs"] = args.miri_max_epochs
    if args.miri_ode_steps is not None:
        config["miri"]["ode_steps"] = args.miri_ode_steps
    if args.miri_batch_size is not None:
        config["miri"]["batch_size"] = args.miri_batch_size
    return config


def main():
    args = build_argparser().parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    out_root = config["output"]["root"]
    os.makedirs(out_root, exist_ok=True)

    print(f"Device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    if args.run_all:
        run_all(config, out_root)
        return

    # single experiment
    missing = args.missing_rate
    if None in (args.method, args.mechanism, missing, args.seed):
        raise SystemExit(
            "For a single run, provide --method --mechanism --missing-rate --seed "
            "(or use --run-all)."
        )
    run_single(config, args.method, args.mechanism, missing, args.seed, out_root)


if __name__ == "__main__":
    main()
