"""
Batched torch implementation of TreeGrad ensembles.

Instead of looping over individual trees in Python (the legacy ``gbm_gen``
closures), all trees are padded to a common size and stacked into single
tensors so the whole ensemble forward pass runs as a handful of batched
ops. The math is identical to the legacy implementation:

    g(x) = sigmoid(-clamp(decision / tau, -32, 32))
    route_prob = exp(log(g + eps) @ route.T)      # product routing
    pred_tree  = route_prob @ leaf

Padded (fake) nodes/leaves contribute nothing because their entries in the
route matrix and leaf vector are zero.
"""

import warnings

import numpy as np
import torch
from torch import nn


def resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    return {"float32": torch.float32, "float64": torch.float64}[str(dtype)]


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class TorchTreeEnsemble(nn.Module):
    """
    Differentiable gradient-boosted tree ensemble on torch.

    Parameters
    ----------
    all_param : list
        Flat list of per-tree parameter triples ``[coef_data, inter, leaf]``
        as produced by ``tree_to_param`` / ``multi_tree_to_param`` /
        ``multiclass_trees_to_param``.
    all_route : list
        Per-tree routing matrices ``(n_leaves, 2 * n_splits)``, or a list of
        such lists (per class) for multiclass ensembles. Entries may be None
        only if that tree is unused.
    all_sparse_info : list
        Per-tree feature masks matching ``all_route``'s structure.
    num_classes : int
        2 for binary/regression (raw score output), >2 for multiclass
        (logits output).
    dtype : torch dtype or str
        float32 (default) or float64.
    device : str or torch.device
        "cpu", "cuda", "mps", ...
    """

    def __init__(
        self,
        all_param,
        all_route,
        all_sparse_info,
        num_classes=2,
        dtype=torch.float32,
        device="cpu",
        tau=0.01,
        eps=1e-11,
    ):
        super().__init__()
        params = list(all_param)
        if len(params) % 3 != 0:
            raise ValueError(
                "all_param must be a flat list of [coef, inter, leaf] triples"
            )
        num_trees = len(params) // 3

        nested = len(all_route) > 0 and isinstance(all_route[0], (list, tuple))
        if nested:
            num_classes_ = len(all_route)
            routes, infos = [], []
            counts = []
            for routes_c, infos_c in zip(all_route, all_sparse_info):
                routes.extend(routes_c)
                infos.extend(infos_c)
                counts.append(len(routes_c))
            if sum(counts) != num_trees:
                raise ValueError(
                    "number of trees ({}) does not match route structure ({})".format(
                        num_trees, sum(counts)
                    )
                )
            class_index = np.repeat(np.arange(num_classes_), counts)
        else:
            num_classes_ = num_classes
            routes, infos = list(all_route), list(all_sparse_info)
            class_index = np.zeros(num_trees, dtype=np.int64)

        coef_np, inter_np, leaf_np, mask_np, route_np = [], [], [], [], []
        for t in range(num_trees):
            coef_t, inter_t, leaf_t = params[3 * t : 3 * t + 3]
            route_t = routes[t]
            mask_t = infos[t]
            if route_t is None:
                raise ValueError(
                    "tree {} has no routing matrix; cannot build ensemble".format(t)
                )
            coef_np.append(np.asarray(coef_t, dtype=np.float64))
            inter_np.append(np.asarray(inter_t, dtype=np.float64))
            leaf_np.append(np.asarray(leaf_t, dtype=np.float64).reshape(-1))
            mask_np.append(np.asarray(mask_t, dtype=np.float64))
            route_np.append(np.asarray(route_t, dtype=np.float64))

        self.num_trees = num_trees
        self.num_classes = num_classes_
        self.eps = eps
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float64))

        n_feat = mask_np[0].shape[0]
        s_max = max(int(m.shape[1]) for m in mask_np)
        l_max = max(int(lf.shape[0]) for lf in leaf_np)

        def zeros(*shape):
            return torch.zeros(*shape, dtype=torch.float64)

        coef = zeros(num_trees, s_max)
        inter = zeros(num_trees, s_max)
        leaf = zeros(num_trees, l_max)
        mask = zeros(num_trees, n_feat, s_max)
        route = zeros(num_trees, l_max, 2 * s_max)
        for t in range(num_trees):
            s, n_leaves = coef_np[t].shape[0], leaf_np[t].shape[0]
            coef[t, :s] = torch.from_numpy(coef_np[t])
            inter[t, :s] = torch.from_numpy(inter_np[t])
            leaf[t, :n_leaves] = torch.from_numpy(leaf_np[t])
            mask[t, :, :s] = torch.from_numpy(mask_np[t])
            # legacy layout: [left splits | right splits] of width 2*s_t.
            # Padded layout widens each half to s_max, so the right half
            # must be re-offset from column s_t to column s_max.
            route_t = torch.from_numpy(route_np[t])
            route[t, :n_leaves, :s] = route_t[:, :s]
            route[t, :n_leaves, s_max : s_max + s] = route_t[:, s:]

        self.coef = nn.Parameter(coef)
        self.inter = nn.Parameter(inter)
        self.leaf = nn.Parameter(leaf)
        self.register_buffer("mask", mask)
        self.register_buffer("route", route)
        self.register_buffer(
            "class_index", torch.as_tensor(class_index, dtype=torch.long)
        )

        self.to(device=device, dtype=resolve_dtype(dtype))

    def set_tau(self, tau):
        with torch.no_grad():
            self.tau.fill_(float(tau))

    def forward(self, X):
        X = torch.as_tensor(X, dtype=self.coef.dtype, device=self.coef.device)
        # effective split weights: zeroed-out entries via feature mask
        weights = self.mask * self.coef.unsqueeze(1)  # (T, F, S)
        decisions_left = torch.einsum("bf,tfs->bts", X, weights)
        decisions = torch.cat([decisions_left, -decisions_left], dim=-1) + torch.cat(
            [self.inter, -self.inter], dim=-1
        ).unsqueeze(0)

        # legacy gumbel_softmax: 1 / (1 + exp(clamp(x / tau))) == sigmoid(-x/tau)
        z = torch.clamp(decisions / self.tau, -32, 32)
        gate = torch.sigmoid(-z)
        decision_soft = torch.log(gate + self.eps)

        route_probas = torch.exp(
            torch.einsum("bts,tsl->btl", decision_soft, self.route.transpose(1, 2))
        )  # (B, T, L)
        tree_out = torch.einsum("btl,tl->bt", route_probas, self.leaf)

        if self.num_classes > 2:
            batch = X.shape[0]
            logits = tree_out.new_zeros(batch, self.num_classes)
            logits.index_add_(1, self.class_index, tree_out)
            return logits
        return tree_out.sum(dim=-1)


def make_loss_fn(task, num_classes=2, loss=None, l1_reg=0.0):
    """
    Numerically stable loss factories.

    task="classification": binary uses BCEWithLogits, multiclass uses
    cross_entropy on raw logits. task="regression" supports "mse" (default)
    and "huber".
    """
    import torch.nn.functional as F

    if task == "classification":
        if num_classes > 2:

            def loss_fn(model, xb, yb):
                out = F.cross_entropy(model(xb), yb.long())
                return out + l1_reg * _l1(model)

        else:

            def loss_fn(model, xb, yb):
                out = F.binary_cross_entropy_with_logits(model(xb), yb.float())
                return out + l1_reg * _l1(model)

    elif task == "regression":
        reduction_loss = {
            "mse": F.mse_loss,
            "huber": F.smooth_l1_loss,
        }
        if loss not in reduction_loss:
            raise ValueError(
                "regression loss must be one of {}, got {}".format(
                    sorted(reduction_loss), loss
                )
            )

        def loss_fn(model, xb, yb):
            out = reduction_loss[loss](model(xb), yb.float())
            return out + l1_reg * _l1(model)

    else:
        raise ValueError(
            "task must be 'classification' or 'regression', got {}".format(task)
        )
    return loss_fn


def _l1(model):
    return model.coef.abs().sum() + model.inter.abs().sum()


def fit_ensemble(
    model,
    X,
    y,
    loss_fn,
    *,
    step_size=0.05,
    num_iters=1000,
    batch_size=32,
    tau_end=None,
    shuffle=True,
    lr_schedule=None,
    compile_mode=False,
    verbose=False,
):
    """
    Convenience wrapper: optimises ``loss_fn(model, xb, yb)`` with Adam
    (regularisation belongs in the loss itself; see :func:`make_loss_fn`).
    """
    device = model.coef.device
    dtype = model.coef.dtype
    X_t = torch.as_tensor(np.asarray(X), dtype=dtype, device=device)
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.floating):
        y_t = torch.as_tensor(y_arr, dtype=dtype, device=device)
    else:
        y_t = torch.as_tensor(y_arr, device=device)

    run_model = model
    if compile_mode:
        try:
            run_model = torch.compile(model)
        except Exception as exc:  # pragma: no cover - depends on torch env
            warnings.warn("torch.compile unavailable ({}); running eager".format(exc))
            run_model = model

    optimizer = torch.optim.Adam(
        [model.coef, model.inter, model.leaf], lr=float(step_size)
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
        if lr_schedule == "cosine"
        else None
    )

    n = X_t.shape[0]
    num_batches = max(1, int(np.ceil(n / batch_size)))
    start_tau = float(model.tau.item())
    stride = max(1, num_iters // 10)

    order = torch.arange(n, device=device)
    for it in range(num_iters):
        if shuffle and it % num_batches == 0:
            order = torch.randperm(n, device=device)
        pos = (it % num_batches) * batch_size
        idx = order[pos : pos + batch_size]
        if tau_end is not None:
            frac = (it + 1) / num_iters
            model.set_tau(start_tau + (float(tau_end) - start_tau) * frac)

        optimizer.zero_grad()
        loss = loss_fn(run_model, X_t[idx], y_t[idx])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if verbose and ((it + 1) % stride == 0 or it == 0):
            print(
                "Iteration {} / {} (loss {:.6f})".format(it + 1, num_iters, float(loss))
            )

    return run_model
