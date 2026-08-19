import torch
import lightning as L

from shared.tools.functions.loss import perplexity_from_loss


class GradientNormLoggingMixin:
    """Log global L2 gradient norm (pre-clip), clip ratio, and LM perplexity."""

    def log_loss_and_ppl(self, loss, loss_key: str, *, on_step: bool, on_epoch: bool, prog_bar: bool = False):
        prefix = getattr(self, "log_prefix", "")
        loss_key = f"{prefix}{loss_key}"
        self.log(loss_key, loss, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)
        ppl_key = loss_key.replace("_loss", "_ppl")
        self.log(
            ppl_key,
            perplexity_from_loss(loss),
            prog_bar=False,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    def on_fit_start(self):
        super().on_fit_start()
        self._grad_norm_max = 0.0
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._grad_clip_count = 0

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            gradient_clip_val,
            norm_type=2.0,
        )
        grad_norm_val = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        clip_ratio = min(1.0, gradient_clip_val / (grad_norm_val + 1e-6))

        self._grad_norm_max = max(self._grad_norm_max, grad_norm_val)
        self._grad_norm_sum += grad_norm_val
        self._grad_norm_count += 1
        if clip_ratio < 1.0:
            self._grad_clip_count += 1

        if self.trainer is not None and self.global_step % self.trainer.log_every_n_steps == 0:
            self.log(
                "grad_norm/global_l2_pre_clip",
                grad_norm_val,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                "grad_norm/clip_ratio",
                clip_ratio,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def on_fit_end(self):
        if self._grad_norm_count == 0:
            super().on_fit_end()
            return

        mean_norm = self._grad_norm_sum / self._grad_norm_count
        clip_fraction = self._grad_clip_count / self._grad_norm_count

        logger = self.logger
        if logger is not None and hasattr(logger, "experiment"):
            summary = logger.experiment.summary
            summary["grad_norm_max_pre_clip"] = self._grad_norm_max
            summary["grad_norm_mean_pre_clip"] = mean_norm
            summary["grad_clip_fraction"] = clip_fraction
            logger.experiment.log({
                "grad_norm/mean_pre_clip": mean_norm,
                "trainer/global_step": self.trainer.global_step,
            })

        super().on_fit_end()
