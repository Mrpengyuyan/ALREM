import argparse
import json
import os
from typing import Any, Dict, List

import yaml


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_runs(outputs_dir: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        params_path = os.path.join(run_dir, "params.json")
        config_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(metrics_path) or not os.path.exists(config_path):
            continue
        metrics = load_json(metrics_path)
        params = load_json(params_path) if os.path.exists(params_path) else {}
        cfg = load_yaml(config_path)
        runs.append({"name": name, "dir": run_dir, "metrics": metrics, "params": params, "cfg": cfg})
    return runs


def write_markdown_table(headers: List[str], rows: List[List[Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ALREM results.")
    parser.add_argument("--outputs_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = scan_runs(args.outputs_dir)

    main_rows: List[List[Any]] = []
    efficiency_rows: List[List[Any]] = []
    ablation_rows: List[List[Any]] = []

    for run in runs:
        cfg = run["cfg"]
        metrics = run["metrics"]
        params = run["params"]
        task = metrics.get("task", cfg.get("task", "unknown"))
        method = cfg.get("method", "unknown")

        if task == "mgsm":
            overall = metrics.get("overall", {})
            main_rows.append(
                [
                    run["name"],
                    task,
                    method,
                    overall.get("accuracy", 0.0),
                    "-",
                    overall.get("num_samples", 0),
                ]
            )
        else:
            overall = metrics.get("overall", {})
            main_rows.append(
                [
                    run["name"],
                    task,
                    method,
                    overall.get("chrf", 0.0),
                    overall.get("bleu", 0.0),
                    overall.get("num_samples", 0),
                ]
            )

        efficiency_rows.append(
            [
                run["name"],
                task,
                method,
                params.get("lora_params_total", 0),
                params.get("trainable_params_total", 0),
                params.get("alrem_target_params", 0),
                params.get("uniform_matched_params", 0),
                params.get("relative_error", 0.0),
            ]
        )

        ablation_rows.append(
            [
                run["name"],
                task,
                method,
                cfg.get("r_high", "-"),
                cfg.get("r_low", "-"),
                cfg.get("r_uniform", "-"),
                cfg.get("cut_ratio_early", cfg.get("early_end", "-")),
                cfg.get("cut_ratio_mid", cfg.get("mid_end", "-")),
                ",".join(cfg.get("target_modules", [])) if cfg.get("target_modules") else "-",
            ]
        )

    write_markdown_table(
        ["run", "task", "method", "primary", "secondary", "num_samples"],
        main_rows,
        os.path.join(args.out_dir, "main_results.md"),
    )
    write_markdown_table(
        [
            "run",
            "task",
            "method",
            "lora_params_total",
            "trainable_params_total",
            "alrem_target_params",
            "uniform_matched_params",
            "relative_error",
        ],
        efficiency_rows,
        os.path.join(args.out_dir, "efficiency.md"),
    )
    write_markdown_table(
        [
            "run",
            "task",
            "method",
            "r_high",
            "r_low",
            "r_uniform",
            "cut_ratio_early_or_end",
            "cut_ratio_mid_or_end",
            "target_modules",
        ],
        ablation_rows,
        os.path.join(args.out_dir, "ablations.md"),
    )


if __name__ == "__main__":
    main()
