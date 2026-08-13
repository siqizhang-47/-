"""Phase 3/4/6: run any adapter method / baseline / ablation on cached scenarios.

Methods (Phase 4 table):
    frozen        baseline 1: frozen CCDM (metrics straight from the cache)
    tafas         baseline 4: TAFAS port (shared translation, MSE)
    cosa          baseline 5: original COSA (buffer-mean context, MSE, translation)
    shift_only    baseline 6: proposed context+CRPS but s == 1
    scale_only    baseline 7: proposed context+CRPS but Delta == 0
    proposed      baseline 8: full location-scale adapter (CRPS)
    independent   ablation J: per-scenario independent fitting control

Examples:
    python experiments/run_adapter.py --method proposed
    python experiments/run_adapter.py --method proposed --override scale_mode=shared
    python experiments/run_adapter.py --method proposed --override loss=mse   # ablation I
    python experiments/run_adapter.py --method proposed --override K=14 adapt_every=7
    python experiments/run_adapter.py --method frozen --cache data/scenarios/wnorm_off
    python experiments/run_adapter.py --method proposed --eval-dependence     # Exp4
    python experiments/run_adapter.py --method proposed --tune                # lambda sweep mode
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, repo_root
from utils.seed import set_seed

METHODS = ["frozen", "tafas", "cosa", "shift_only", "scale_only", "proposed", "independent"]

ADAPTER_DEFAULTS = {
    "K": 7, "adapt_steps": 5, "adapt_every": 1, "adapter_lr": 1.0e-3,
    "lam_s": 0.1, "lam_delta": 1.0e-3, "grad_clip": 0.1,
    "delta_mode": "lowrank", "n_basis": 4, "scale_mode": "carrier",
    "s_min": 0.5, "s_max": 2.0, "d_hidden": 64, "loss": "crps",
    "cosa_buffer_context_size": 5,
}


def build_method(method, cfg):
    """Returns (adapter, loss_fn, ctx_fn)."""
    import torch  # noqa: F401
    from adapters.context import ResidualContext
    from adapters.cosa_original import COSAOriginalAdapter, cosa_loss
    from adapters.lsa import LSAdapter, lsa_loss
    from adapters.tafas_port import TAFASAdapter, tafas_loss

    d_ctx = ResidualContext(K=cfg.K).d_ctx

    if method == "frozen":
        return None, None, None

    if method == "cosa":
        n = cfg.cosa_buffer_context_size
        adapter = COSAOriginalAdapter(H=cfg.pred_len, C=cfg.num_target,
                                      buffer_context_size=n)
        return adapter, cosa_loss, lambda ctx: ctx.recent_target_means(n)

    if method == "tafas":
        adapter = TAFASAdapter(H=cfg.pred_len, C=cfg.num_target)
        return adapter, lambda a, s, y, c: tafas_loss(a, s, y, c), lambda ctx: None

    # LSA family
    objective = cfg.loss
    kwargs = dict(H=cfg.pred_len, C=cfg.num_target, d_ctx=d_ctx,
                  d_hidden=cfg.d_hidden, delta_mode=cfg.delta_mode,
                  n_basis=cfg.n_basis, scale_mode=cfg.scale_mode,
                  s_min=cfg.s_min, s_max=cfg.s_max)
    if method == "shift_only":
        kwargs["s_frozen"] = True
    elif method == "scale_only":
        kwargs["delta_frozen"] = True
    elif method == "independent":
        kwargs["independent_scenarios"] = True
        objective = "mse_per_scenario"
    adapter = LSAdapter(**kwargs)

    def loss_fn(a, scen, y, ctx):
        return lsa_loss(a, scen, y, ctx, w_c=loss_fn.w_c,
                        lam_s=cfg.lam_s, lam_delta=cfg.lam_delta,
                        objective=objective)
    import torch
    loss_fn.w_c = torch.ones(cfg.num_target)  # replaced by runner after prefill
    return adapter, loss_fn, lambda ctx: ctx.features()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = repo_root()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--config", default=str(root / "configs/backbone_energy.yaml"))
    parser.add_argument("--deploy-config", default=str(root / "configs/deploy.yaml"))
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--cache", default=None, help="scenario cache dir (default wnorm_on)")
    parser.add_argument("--out", default=None, help="results dir (default results/<name>)")
    parser.add_argument("--name", default=None, help="experiment name for results dir")
    parser.add_argument("--eval-dependence", action="store_true", help="Exp4 CME metrics")
    parser.add_argument("--tune", action="store_true",
                        help="lambda-sweep mode: refuses caches after 2020-06-30")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config, args.deploy_config,
                      args.extra_config, overrides=args.override)
    for key, val in ADAPTER_DEFAULTS.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.eval_dependence = args.eval_dependence
    set_seed(cfg.seed)

    from deployment.replay import ReplayRunner, ScenarioCache

    cache_dir = args.cache or str(root / cfg.scenario_dir / "wnorm_on")
    name = args.name or args.method
    out_dir = args.out or str(root / cfg.results_dir / name)

    adapter, loss_fn, ctx_fn = build_method(args.method, cfg)
    runner = ReplayRunner(cfg, args.method, adapter, loss_fn, ctx_fn,
                          cache_dir, tune_mode=args.tune)

    all_dates = ScenarioCache(cache_dir).dates()
    deploy_start = pd.Timestamp(cfg.deploy_start)
    end = min(pd.Timestamp(cfg.deploy_end),
              pd.Timestamp("2020-06-30")) if args.tune else pd.Timestamp(cfg.deploy_end)
    warmup_dates = [d for d in all_dates if d < deploy_start]
    deploy_dates = [d for d in all_dates if deploy_start <= d <= end]
    assert deploy_dates, "no deployment days in cache for the requested range"

    if args.method != "frozen":
        assert warmup_dates, ("adapter methods need warmup days in the cache "
                              "for cold start -- rerun run_deploy.py with warmup_days > 0")
    if warmup_dates:
        runner.prefill(warmup_dates)
        if loss_fn is not None and hasattr(loss_fn, "w_c"):
            loss_fn.w_c = runner.w_c
    runner.run(deploy_dates, desc=name)
    runner.save_results(out_dir)


if __name__ == "__main__":
    main()
