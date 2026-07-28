from pathlib import Path
import logging

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from stable_pretraining import data as dt
from stable_worldmodel.wm.utils import save_pretrained

logger = logging.getLogger(__name__)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer for DataLoader workers."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific dataset column."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCkptCallback(Callback):
    """Save object checkpoints for stable_worldmodel evaluation."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        keep_last_n: int = 0,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.keep_last_n = max(0, int(keep_last_n))
        self.epoch_offset = max(0, int(epoch_offset))

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1 + self.epoch_offset
            is_final = trainer.current_epoch + 1 == trainer.max_epochs
            if epoch % self.epoch_interval == 0 or is_final:
                self._save(pl_module.model, epoch)
                if self.keep_last_n > 0:
                    self._prune_old_epochs(epoch)

    def _checkpoint_dir(self) -> Path:
        import stable_worldmodel as swm

        return Path(swm.data.utils.get_cache_dir("checkpoints"), self.run_name)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f"weights_epoch_{epoch}.pt",
        )

    def _prune_old_epochs(self, current_epoch: int) -> None:
        ckpt_dir = self._checkpoint_dir()
        if not ckpt_dir.is_dir():
            return
        epochs = sorted(
            int(p.stem.split("_")[-1])
            for p in ckpt_dir.glob("weights_epoch_*.pt")
        )
        if len(epochs) <= self.keep_last_n:
            return
        for epoch in epochs[: len(epochs) - self.keep_last_n]:
            path = ckpt_dir / f"weights_epoch_{epoch}.pt"
            if path.exists():
                path.unlink()
                logger.info("Pruned old checkpoint %s", path.name)


class PlanningEvalCallback(Callback):
    """Run PushT planning evaluation at the end of selected training epochs."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.planning_cfg = cfg.planning_eval
        self.dataset = None
        self.process = None
        self.eval_cfg = None

    def _should_run(self, trainer) -> bool:
        if not bool(self.planning_cfg.get("enabled", False)):
            return False
        every_n = int(self.planning_cfg.get("every_n_epochs", 1))
        epoch = trainer.current_epoch + 1
        return epoch % every_n == 0 or epoch == trainer.max_epochs

    def _build_eval_cfg(self):
        from planning_eval import default_planning_cost_mode

        planning_cost = OmegaConf.to_container(self.planning_cfg.planning_cost, resolve=True)
        if planning_cost.get("mode") == "auto":
            model_mode = OmegaConf.select(self.cfg, "model.planning_cost_mode", default=None)
            if model_mode and str(model_mode).lower() not in ("auto", "mse", ""):
                planning_cost["mode"] = str(model_mode).lower()
            else:
                planning_cost["mode"] = default_planning_cost_mode(
                    self.cfg.loss.prediction.name,
                    energy=str(self.cfg.loss.prediction.get("energy", "dot_product")),
                )

        eval_cfg = OmegaConf.create(
            {
                "eval": {
                    "num_eval": self.planning_cfg.num_eval,
                    "goal_offset_steps": self.planning_cfg.goal_offset_steps,
                    "eval_budget": self.planning_cfg.eval_budget,
                    "img_size": self.planning_cfg.img_size,
                    "dataset_name": self.planning_cfg.dataset_name,
                    "seed": self.planning_cfg.seed,
                    "manifest_path": self.planning_cfg.get("manifest_path"),
                    "save_video": self.planning_cfg.get("save_video"),
                    "callables": OmegaConf.to_container(
                        self.planning_cfg.get("callables"), resolve=True
                    ),
                },
                "dataset": {
                    "keys_to_cache": OmegaConf.to_container(
                        self.planning_cfg.keys_to_cache, resolve=True
                    ),
                },
                "world": OmegaConf.to_container(self.planning_cfg.world, resolve=True),
                "plan_config": OmegaConf.to_container(
                    self.planning_cfg.plan_config, resolve=True
                ),
                "planning_cost": planning_cost,
                "solver": OmegaConf.to_container(self.planning_cfg.solver, resolve=True),
            }
        )
        return eval_cfg

    def _ensure_dataset(self):
        if self.dataset is not None:
            return
        from planning_eval import build_processors, get_dataset

        self.eval_cfg = self._build_eval_cfg()
        self.dataset = get_dataset(
            self.eval_cfg.eval.dataset_name,
            self.eval_cfg.dataset.keys_to_cache,
        )
        self.process = build_processors(
            self.dataset, self.eval_cfg.dataset.keys_to_cache
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self._should_run(trainer):
            return

        self._ensure_dataset()
        epoch = trainer.current_epoch + 1

        from planning_eval import append_results_file, run_planning_eval

        result = run_planning_eval(
            pl_module.model,
            self.eval_cfg,
            dataset=self.dataset,
            process=self.process,
        )
        success_rate = result["success_rate"]
        msg = (
            f"[planning_eval] epoch={epoch} success_rate={success_rate:.2f}% "
            f"time={result['evaluation_time']:.1f}s"
        )
        logger.info(msg)
        print(msg, flush=True)

        pl_module.log(
            "planning/success_rate",
            torch.tensor(success_rate, device=pl_module.device),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        pl_module.log(
            "planning/eval_time_s",
            torch.tensor(result["evaluation_time"], device=pl_module.device),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "planning/success_rate": success_rate,
                    "planning/eval_time_s": result["evaluation_time"],
                    "epoch": trainer.current_epoch,
                },
                step=trainer.global_step,
            )

        import stable_worldmodel as swm

        results_path = (
            Path(swm.data.utils.get_cache_dir("checkpoints"))
            / self.cfg.output_model_name
            / "planning_eval"
            / f"epoch_{epoch:03d}.txt"
        )
        append_results_file(results_path, self.eval_cfg, result)
