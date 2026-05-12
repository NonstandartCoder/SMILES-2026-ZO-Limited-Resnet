from typing import Callable
import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        eps: float = 5e-4,
        K: int = 16,
        reg_lambda: float = 0.1,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.K = K
        self.reg_lambda = reg_lambda

        self.expand_at_step = 32
        self.has_expanded = False

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        self.all_layer_names = self.layer_names + self._get_bn_names()

        self.m = {}
        self.v = {}
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8

        self.initial_params = {}
        self._capture_initial(self.layer_names)

    def _get_bn_names(self) -> list[str]:
        bn_names = []
        for name, param in self.model.named_parameters():
            if ("bn" in name or "downsample.1" in name) and (
                name.endswith(".weight") or name.endswith(".bias")
            ):
                bn_names.append(name)
        return bn_names

    def _capture_initial(self, layer_names: list[str]):
        named = dict(self.model.named_parameters())
        for n in layer_names:
            if n not in self.initial_params:
                self.initial_params[n] = named[n].data.clone()

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        return {n: named[n] for n in self.layer_names}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        grads: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in params.items()
        }

        for _ in range(self.K):
            u_dict = {}
            for name, param in params.items():
                u = torch.randn_like(param)
                u = u / (u.norm() / u.numel() ** 0.5)
                u_dict[name] = u

            with torch.no_grad():
                for name, param in params.items():
                    param.data.add_(u_dict[name], alpha=self.eps)
                f_plus = loss_fn()

                for name, param in params.items():
                    param.data.sub_(u_dict[name], alpha=2.0 * self.eps)
                f_minus = loss_fn()

                for name, param in params.items():
                    param.data.add_(u_dict[name], alpha=self.eps)

                diff = (f_plus - f_minus) / (2.0 * self.eps)
                for name in params:
                    grads[name] += diff * u_dict[name]

        for name in grads:
            grads[name] /= self.K
        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        self.t += 1
        with torch.no_grad():
            for name, param in params.items():
                if name not in self.m:
                    self.m[name] = torch.zeros_like(param)
                    self.v[name] = torch.zeros_like(param)

                self.m[name] = (
                    self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
                )
                self.v[name] = (
                    self.beta2 * self.v[name]
                    + (1 - self.beta2) * (grads[name] ** 2)
                )

                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                param.data.sub_(
                    self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps)
                )

    def step(self, loss_fn: Callable[[], float]) -> float:
        if (
            not self.has_expanded
            and self.expand_at_step is not None
            and self.t >= self.expand_at_step
        ):
            self.layer_names = self.all_layer_names
            self._capture_initial(self.all_layer_names)
            self.has_expanded = True

        params = self._active_params()

        def reg_loss_fn():
            ce = loss_fn()
            reg = 0.0
            for name, p in params.items():
                reg += torch.sum((p - self.initial_params[name]) ** 2)
            return ce + self.reg_lambda * reg.item()

        with torch.no_grad():
            loss_before = reg_loss_fn()

        grads = self._estimate_grad(reg_loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)