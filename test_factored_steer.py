"""Equivalence + speed test: current dense D×D steering vs. rank-2 factored form.

The library's AngularSteeringOperator builds P = b1·b1ᵀ + b2·b2ᵀ as a dense D×D
matrix and runs `h @ P.T` every forward. P is rank-2 by construction, so the same
result can be obtained with two D×2 matmuls and (N, 2) intermediates instead of
(N, D).

This script:
  1. Implements the factored version (with the same caching strategy as the dense
     operator, so the speed comparison is apples-to-apples).
  2. Verifies they produce identical outputs across N, θ, adaptive_mode, dtype.
  3. Times both to confirm the throughput claim.

Run:  python test_factored_steer.py
"""
import math
import time

import numpy as np
import torch

from vllm_angular_steering import AngularSteeringOperator


class FactoredSteeringOperator:
    """Rank-2 factored equivalent of AngularSteeringOperator.

    Math derivation:
      Current:  steered = h - (h·P) + r·v_θ
                where P = b1·b1ᵀ + b2·b2ᵀ  (D×D, rank 2)
                      v_θ = cos(θ)·b1 + sin(θ)·b2

      Factored: Let Q = stack(b1, b2)                       (2, D)
                proj_2d = h @ Qᵀ                            (N, 2)
                r = ||proj_2d||
                steered = h + (r·[cos,sin] - proj_2d) @ Q   (one back-proj)

      Caches Q per (device, dtype) and the (cos, sin) tensor per (device, dtype,
      θ) — mirroring AngularSteeringOperator so the comparison is fair.
    """
    def __init__(self, b1_np, b2_np):
        b1 = torch.from_numpy(b1_np).float()
        b2 = torch.from_numpy(b2_np).float()
        # Orthonormalize, same recipe as the dense operator
        self.b1 = b1 / b1.norm()
        self.b2 = b2 - (b2 @ self.b1) * self.b1
        self.b2 = self.b2 / self.b2.norm()
        self._Q_cache = {}
        self._cs_cache = {}

    def _get_Q(self, device, dtype):
        key = (device, dtype)
        if key not in self._Q_cache:
            self._Q_cache[key] = torch.stack([
                self.b1.to(device=device, dtype=dtype),
                self.b2.to(device=device, dtype=dtype),
            ], dim=0)
        return self._Q_cache[key]

    def _get_cs(self, theta_deg, device, dtype):
        key = (device, dtype, theta_deg % 360)
        if key not in self._cs_cache:
            theta = math.radians(theta_deg % 360)
            self._cs_cache[key] = torch.tensor(
                [math.cos(theta), math.sin(theta)], dtype=dtype, device=device,
            )
        return self._cs_cache[key]

    def steer(self, h, target_degree, adaptive_mode=0):
        Q = self._get_Q(h.device, h.dtype)
        cs = self._get_cs(target_degree, h.device, h.dtype)

        proj_2d = h @ Q.T                       # (N, D) @ (D, 2) → (N, 2)
        r = proj_2d.norm(dim=-1, keepdim=True)  # (N, 1)
        delta_2d = r * cs - proj_2d             # (N, 2)
        steered = h + delta_2d @ Q              # (N, 2) @ (2, D) → (N, D)

        if adaptive_mode == 0:
            return steered
        if adaptive_mode == 1:
            mask = ((h @ Q[0]) > 0).unsqueeze(-1)
            return torch.where(mask, steered, h)
        raise ValueError(f"unknown adaptive_mode: {adaptive_mode}")


def equivalence_test(device):
    print(f"\n--- Equivalence check  ({device}) ---")
    print(f"{'dtype':>10} {'N':>6} {'theta':>6} {'mode':>5} "
          f"{'max_abs_err':>14} {'max_rel_err':>14}")

    torch.manual_seed(0)
    np.random.seed(0)
    D = 2560

    for dtype in (torch.float32, torch.bfloat16):
        b1_np = np.random.randn(D).astype(np.float32)
        b2_np = np.random.randn(D).astype(np.float32)
        op_dense = AngularSteeringOperator(b1_np, b2_np)
        op_fact = FactoredSteeringOperator(b1_np, b2_np)

        for N in (1, 16, 256, 1024):
            for theta in (0, 45, 90, 180, 270):
                for mode in (0, 1):
                    h = torch.randn(N, D, device=device, dtype=dtype)
                    out_dense = op_dense.steer(h, target_degree=theta, adaptive_mode=mode)
                    out_fact = op_fact.steer(h, target_degree=theta, adaptive_mode=mode)
                    diff = (out_dense - out_fact).abs()
                    max_abs = diff.max().item()
                    max_rel = (diff / (out_dense.abs() + 1e-6)).max().item()
                    print(f"{str(dtype).split('.')[-1]:>10} {N:>6} {theta:>6} "
                          f"{mode:>5} {max_abs:>14.2e} {max_rel:>14.2e}")


def speed_test(device):
    if device.type != "cuda":
        print("\n--- Speed test skipped (CPU) ---")
        return
    print(f"\n--- Speed benchmark  ({device})  D=2560, mode=0, θ=90 ---")
    print(f"{'N':>8} {'dense_μs':>12} {'fact_μs':>12} {'speedup':>10}")

    D, theta = 2560, 90
    np.random.seed(0)
    b1_np = np.random.randn(D).astype(np.float32)
    b2_np = np.random.randn(D).astype(np.float32)
    op_dense = AngularSteeringOperator(b1_np, b2_np)
    op_fact = FactoredSteeringOperator(b1_np, b2_np)
    dtype = torch.bfloat16

    for N in (1, 8, 32, 256, 1024, 4096):
        h = torch.randn(N, D, device=device, dtype=dtype)
        # warmup (also primes both operators' caches)
        for _ in range(50):
            op_dense.steer(h, theta, 0)
            op_fact.steer(h, theta, 0)
        torch.cuda.synchronize()

        iters = 1000 if N <= 256 else 200
        t0 = time.perf_counter()
        for _ in range(iters):
            op_dense.steer(h, theta, 0)
        torch.cuda.synchronize()
        dt_dense = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            op_fact.steer(h, theta, 0)
        torch.cuda.synchronize()
        dt_fact = (time.perf_counter() - t0) / iters * 1e6

        print(f"{N:>8} {dt_dense:>12.1f} {dt_fact:>12.1f} {dt_dense/dt_fact:>9.1f}x")


def memory_test(device):
    if device.type != "cuda":
        print("\n--- Memory test skipped (CPU) ---")
        return
    print(f"\n--- Memory measurement  ({device})  D=2560, bf16 ---")

    D = 2560
    np.random.seed(0)
    b1_np = np.random.randn(D).astype(np.float32)
    b2_np = np.random.randn(D).astype(np.float32)

    # 1. Persistent operator state: how much each cached operator keeps on device
    print("\n  Persistent operator state (cached on device after first call):")
    print(f"  {'op':>20} {'bytes':>14} {'human':>10}")

    def measure_state(make_op):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        op = make_op()
        # trigger device-cache population by calling once
        h = torch.randn(1, D, device=device, dtype=torch.bfloat16)
        op.steer(h, target_degree=90, adaptive_mode=0)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        # subtract the size of h itself (allocated by us in the measured window)
        h_bytes = h.numel() * h.element_size()
        return after - before - h_bytes

    dense_state = measure_state(lambda: AngularSteeringOperator(b1_np, b2_np))
    fact_state  = measure_state(lambda: FactoredSteeringOperator(b1_np, b2_np))
    print(f"  {'AngularSteeringOp':>20} {dense_state:>14,} {dense_state/2**20:>8.2f} MB")
    print(f"  {'FactoredSteeringOp':>20} {fact_state:>14,} {fact_state/2**10:>8.1f} KB")
    print(f"  ratio: {dense_state / max(fact_state, 1):.0f}×")

    # 2. Per-call peak activation memory across N
    print("\n  Peak activation memory per steer() call (intermediates):")
    print(f"  {'N':>6} {'dense_bytes':>14} {'dense':>12} {'fact_bytes':>14} {'fact':>12} {'ratio':>7}")

    op_dense = AngularSteeringOperator(b1_np, b2_np)
    op_fact  = FactoredSteeringOperator(b1_np, b2_np)
    # Prime device caches so we measure only per-call intermediates
    _ = op_dense.steer(torch.randn(1, D, device=device, dtype=torch.bfloat16), 90, 0)
    _ = op_fact.steer(torch.randn(1, D, device=device, dtype=torch.bfloat16), 90, 0)

    def measure_call(op, N):
        h = torch.randn(N, D, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = op.steer(h, target_degree=90, adaptive_mode=0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        # peak − base = bytes alive during the call, INCLUDING the returned `out`
        # subtract `out` size to get just the intermediates
        return peak - base - out.numel() * out.element_size()

    for N in (1, 256, 1024, 4096):
        d = measure_call(op_dense, N)
        f = measure_call(op_fact, N)
        ratio = d / max(f, 1)
        print(f"  {N:>6} {d:>14,} {d/2**20:>9.2f} MB {f:>14,} {f/2**20:>9.2f} MB {ratio:>6.1f}×")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    equivalence_test(device)
    speed_test(device)
    memory_test(device)


if __name__ == "__main__":
    main()
