#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算某个 W&B run 的指定 metric（默认 performance/iter_time）的均值/中位数等统计
用法示例：
  python wandb_iter_time_avg.py --entity your_team --project your_proj --run_id abc123 \
      --metric "performance/iter_time" --start_step 0 --end_step 100000 --trim 0.0
"""
import argparse
import math
import numpy as np
import wandb

def fetch_metric_series(entity, project, run_id, metric, start_step=None, end_step=None):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    values = []
    steps  = []

    # scan_history 更省内存，支持只拉取特定 keys
    for row in run.scan_history(keys=[metric, "_step"], page_size=2000):
        if metric not in row:
            continue
        val = row[metric]
        step = row.get("_step", None)

        # 过滤 NaN 或 None
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue

        # 步数过滤
        if start_step is not None and step is not None and step < start_step:
            continue
        if end_step is not None and step is not None and step >= end_step:
            continue

        values.append(float(val))
        steps.append(step)

    return np.array(values, dtype=float), np.array(steps, dtype=float)

def trimmed_stats(x, trim=0.0):
    """对称裁剪后统计（trim=0.05 表示去掉两端各5%）"""
    if x.size == 0:
        return None
    if trim > 0:
        lo = 100 * trim
        hi = 100 * (1 - trim)
        lo_v, hi_v = np.percentile(x, [lo, hi])
        x = x[(x >= lo_v) & (x <= hi_v)]
        if x.size == 0:
            return None
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True, help="W&B entity（团队/用户名）")
    p.add_argument("--project", required=True, help="W&B project 名")
    p.add_argument("--run_id", required=True, help="目标 run 的 ID（UI 地址最后那段）")
    p.add_argument("--metric", default="performance/iter_time", help="要统计的 metric 名")
    p.add_argument("--start_step", type=int, default=0, help="起始 _step（含）")
    p.add_argument("--end_step", type=int, default=4000, help="结束 _step（不含）")
    p.add_argument("--trim", type=float, default=0.0, help="对称裁剪比例(0~0.49)，如0.05去掉两端各5%")
    args = p.parse_args()

    values, steps = fetch_metric_series(
        args.entity, args.project, args.run_id, args.metric,
        start_step=args.start_step, end_step=args.end_step
    )

    stats = trimmed_stats(values, trim=args.trim)
    if not stats:
        print("没有取到任何有效数据（可能 metric 名不对或该区间没有数据）。")
        return

    print(f"Run: {args.entity}/{args.project}/{args.run_id}")
    print(f"Metric: {args.metric}")
    if args.start_step is not None or args.end_step is not None:
        print(f"Step range: [{args.start_step if args.start_step is not None else '-inf'}, "
              f"{args.end_step if args.end_step is not None else '+inf'})")
    if args.trim > 0:
        print(f"Trimmed: 两端各 {args.trim*100:.1f}%")

    print("\n统计：")
    print(f"- 样本数 (count): {stats['count']}")
    print(f"- 均值 (mean): {stats['mean']:.6f}")
    print(f"- 中位数 (median): {stats['median']:.6f}")
    print(f"- 标准差 (std): {stats['std']:.6f}")
    print(f"- 最小值 (min): {stats['min']:.6f}")
    print(f"- 最大值 (max): {stats['max']:.6f}")

if __name__ == "__main__":
    main()