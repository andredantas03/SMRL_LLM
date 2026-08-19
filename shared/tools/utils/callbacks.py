import time

from lightning.pytorch.callbacks import Callback
import torch


class WandbReportCallback(Callback):
    def __init__(self, n_params: int):
        self.n_params = n_params
        self.start_time = None
        self._logged_n_params = False

    def _log_to_wandb(self, trainer, metrics: dict) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        logger.experiment.log({**metrics, "trainer/global_step": trainer.global_step})

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        self._logged_n_params = False

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        run = logger.experiment
        run.summary["n_params"] = self.n_params
        self._log_to_wandb(trainer, {"n_params": float(self.n_params)})

        if hasattr(run, "define_metric"):
            run.define_metric("epoch", hidden=True)
            run.define_metric("lr*", hidden=True)
            run.define_metric("learning_rate", hidden=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.start_time is None:
            return

        step = trainer.global_step
        if step == 0 and not self._logged_n_params:
            self._log_to_wandb(trainer, {"n_params": float(self.n_params)})
            self._logged_n_params = True

        if step % trainer.log_every_n_steps != 0:
            return

        elapsed = time.time() - self.start_time
        pl_module.log(
            "train_time_sec",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def on_fit_end(self, trainer, pl_module):
        if self.start_time is None:
            return

        total_sec = time.time() - self.start_time
        self._log_to_wandb(trainer, {"train_time_total_sec": total_sec})

        logger = pl_module.logger
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.summary["train_time_total_sec"] = total_sec


class VRAMCallback(Callback):
    def __init__(self):
        self.peak_gpu_used_gb = 0.0

    def _gpu_vram_used_gb(self, device=None):
        device = device if device is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / 1e9

    def on_fit_start(self, trainer, pl_module):
        self.peak_gpu_used_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return
        step = trainer.global_step
        log_every = trainer.log_every_n_steps
        if step % log_every != 0:
            return
        if step % 50 != 0:
            return

        gpu_used_gb = self._gpu_vram_used_gb()
        self.peak_gpu_used_gb = max(self.peak_gpu_used_gb, gpu_used_gb)

        pl_module.log("vram/allocated_gb", torch.cuda.memory_allocated() / 1e9, prog_bar=False)
        pl_module.log("vram/reserved_gb", torch.cuda.memory_reserved() / 1e9)
        pl_module.log("vram/peak_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
        pl_module.log("vram/gpu_used_gb", gpu_used_gb, prog_bar=False)
        pl_module.log("vram/gpu_peak_used_gb", self.peak_gpu_used_gb, prog_bar=False)

    def on_fit_end(self, trainer, pl_module):
        if not torch.cuda.is_available() or pl_module.logger is None:
            return

        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_gpu_used_gb = max(self.peak_gpu_used_gb, self._gpu_vram_used_gb())

        pl_module.logger.experiment.summary["vram_peak_allocated_gb"] = peak_allocated_gb
        pl_module.logger.experiment.summary["vram_gpu_peak_gb"] = peak_gpu_used_gb
        pl_module.logger.experiment.log({
            "vram/peak_allocated_final_gb": peak_allocated_gb,
            "vram/gpu_peak_final_gb": peak_gpu_used_gb,
        })