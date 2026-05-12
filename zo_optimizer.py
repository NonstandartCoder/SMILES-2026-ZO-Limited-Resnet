"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-2,  # Adam usually needs a higher LR for ZO
            eps: float = 1e-3,
            perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode

        # We start by tuning only the final layer (the head)
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam state
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0  # Time step
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        return {n: named[n] for n in self.layer_names}

    def _estimate_grad(
            self,
            loss_fn: Callable[[], float],
            params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        grads: dict[str, torch.Tensor] = {}

        # SPSA: Sample ONE random perturbation for ALL parameters
        # This is the key to efficiency!
        u_dict = {}
        for name, param in params.items():
            if self.perturbation_mode == "gaussian":
                u = torch.randn_like(param)
            else:
                u = torch.bernoulli(torch.full_like(param, 0.5)) * 2 - 1
            u_dict[name] = u

        with torch.no_grad():
            # Apply perturbation: theta + eps * u
            for name, param in params.items():
                param.data.add_(u_dict[name], alpha=self.eps)
            f_plus = loss_fn()

            # Apply perturbation: theta - eps * u (subtract 2*eps)
            for name, param in params.items():
                param.data.sub_(u_dict[name], alpha=2.0 * self.eps)
            f_minus = loss_fn()

            # Restore original: theta
            for name, param in params.items():
                param.data.add_(u_dict[name], alpha=self.eps)

            # Gradient Estimate: (f+ - f-) / (2 * eps) * u
            # Note: For SPSA, we multiply by the direction vector
            diff = (f_plus - f_minus) / (2.0 * self.eps)
            for name in params:
                grads[name] = diff * u_dict[name]

        return grads

    def _update_params(
            self,
            params: dict[str, nn.Parameter],
            grads: dict[str, torch.Tensor],
    ) -> None:
        self.t += 1
        with torch.no_grad():
            for name, param in params.items():
                # Initialize Adam states if not present
                if name not in self.m:
                    self.m[name] = torch.zeros_like(param)
                    self.v[name] = torch.zeros_like(param)

                # Update moments
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads[name] ** 2)

                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                # Apply update
                param.data.sub_(self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps))

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)
