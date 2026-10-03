"""Losses and routing diagnostics of the soft MoE (shared by training, Optuna and evaluation).

Routing statistics are computed from the per-entry routing weights w (N, 4) and the true
labels y (N,) of a deterministic manifest:

    R[c, k]   mean weight of branch k on inputs of true class c (rows sum to 1)
    usage_k   mean_c R[c, k]                      class-balanced share of branch k (1/4 if balanced)
    own_k     R[k, k]                             weight on the inputs the branch exists for
    off_k     mean_{c != k} R[c, k]               weight on inputs of the three other classes

With perfect routing R is the identity matrix: usage_k = 1/4, own_k = 1, off_k = 0.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.common.losses import ssim
from src.data.corruptions import CLASSES

N_BRANCHES = len(CLASSES)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def balance_loss(weights):
    """The brief's routing-balance term: sum_k (mean_batch(w_k) - 1/K)^2.

    It only sees the batch-mean weights. On an exactly class-balanced batch, perfect routing
    (one-hot on the true class) has mean 1/K for every branch and costs 0, so the term does
    not fight the cross-entropy; total collapse onto one branch costs (1 - 1/K)^2 + (K-1)/K^2
    = 0.75 for K = 4.
    """
    k = weights.shape[1]
    return ((weights.float().mean(dim=0) - 1.0 / k) ** 2).sum()


def moe_loss(output, target, weights, logits, labels, config):
    """L = l1 * L1 + ls * (1 - SSIM) + lc * CE + lb * L_balance. Returns (loss, detached parts).

    CE is applied to the raw gate logits by default (config["ce_input"] == "logits"): the
    Task 2 objective, which keeps G(x) a calibrated classifier and leaves the sharpness of the
    routing to tau. "routing" applies it to G(x) / tau, i.e. to the routing distribution w.
    """
    l1 = F.l1_loss(output.float(), target.float())
    s = ssim(output, target)
    ce_logits = logits if config["ce_input"] == "logits" else logits / config["tau"]
    ce = F.cross_entropy(ce_logits.float(), labels)
    bal = balance_loss(weights)
    loss = (config["lambda_l1"] * l1 + config["lambda_ssim"] * (1 - s)
            + config["lambda_ce"] * ce + config["lambda_balance"] * bal)
    return loss, {"l1": l1.detach(), "ssim": s.detach(), "ce": ce.detach(), "balance": bal.detach()}


# --------------------------------------------------------------------------- #
# Routing statistics
# --------------------------------------------------------------------------- #
def routing_entropy(weights):
    """Per-entry routing entropy normalised by log(4): 0 = one branch only, 1 = uniform."""
    w = torch.as_tensor(weights, dtype=torch.float64).clamp_min(1e-12)
    return -(w * w.log()).sum(dim=1) / math.log(w.shape[1])


def routing_matrix(weights, labels):
    """R[c, k]: mean weight of branch k over the entries of true class c (NaN row if absent)."""
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(labels)
    matrix = np.full((N_BRANCHES, N_BRANCHES), np.nan)
    for c in range(N_BRANCHES):
        if (y == c).any():
            matrix[c] = w[y == c].mean(axis=0)
    return matrix


def routing_stats(weights, labels):
    """Matrix, per-branch usage / own / off-class weights, entropy and routing accuracy."""
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(labels)
    matrix = routing_matrix(w, y)
    present = ~np.isnan(matrix[:, 0])
    usage = np.nanmean(matrix, axis=0)
    off = np.array([np.nanmean(np.delete(matrix[:, k], k)) if np.delete(present, k).any() else np.nan
                    for k in range(N_BRANCHES)])
    return {
        "matrix": matrix,
        "usage": usage,
        "own": np.diag(matrix),
        "off_class": off,
        "entropy": float(routing_entropy(w).mean()),
        "accuracy": float((w.argmax(axis=1) == y).mean()),
    }


def detect_collapse(stats, min_usage=0.05, max_off_class=0.5):
    """Routing-collapse criterion on a class-balanced validation set. Returns (collapsed, reason).

    A branch is
      dead      if its class-balanced usage is below `min_usage` (a fifth of its fair share
                1/4). Since usage_k >= own_k / 4, a dead branch gets less than 0.2 of the
                weight even on the inputs it was built for: the MoE has lost that expert.
      dominant  if it takes more than `max_off_class` of the weight, on average, on inputs of
                the other three classes: routing no longer follows the corruption type.
    Either condition is collapse. Total collapse onto one branch triggers both.
    """
    reasons = []
    for k, name in enumerate(CLASSES):
        if stats["usage"][k] < min_usage:
            reasons.append(f"dead branch {name} (usage {stats['usage'][k]:.3f} < {min_usage})")
        if stats["off_class"][k] > max_off_class:
            reasons.append(f"branch {name} dominates other classes (off-class weight "
                           f"{stats['off_class'][k]:.3f} > {max_off_class})")
    return bool(reasons), "; ".join(reasons)


def flat_routing_stats(stats, prefix="routing/"):
    """Scalars for W&B / Optuna attributes: usage, own and off-class weight per branch, R[c, k]."""
    out = {f"{prefix}entropy": stats["entropy"], f"{prefix}accuracy": stats["accuracy"]}
    for k, name in enumerate(CLASSES):
        out[f"{prefix}usage_{name}"] = float(stats["usage"][k])
        out[f"{prefix}own_{name}"] = float(stats["own"][k])
        out[f"{prefix}off_class_{name}"] = float(stats["off_class"][k])
    for c, true in enumerate(CLASSES):
        for k, name in enumerate(CLASSES):
            out[f"{prefix}R/{true}->{name}"] = float(stats["matrix"][c, k])
    return out


# --------------------------------------------------------------------------- #
# Hard routing (comparison only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def hard_route(x, route, experts):
    """Task 2-style hard routing: route 0 keeps the input (identity bypass), route k runs expert k only.

    `experts` maps "salt"/"blur"/"occlusion" to modules in eval mode (BatchNorm with running
    statistics, so running an expert on a sub-batch gives the same per-sample result).
    """
    output = x.clone()
    for k, name in enumerate(CLASSES[1:], start=1):
        idx = torch.nonzero(route == k).squeeze(1)
        if idx.numel():
            output[idx] = experts[name](x[idx]).to(output.dtype)
    return output
