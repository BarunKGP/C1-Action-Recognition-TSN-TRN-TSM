import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from systems import EpicActionRecogintionDataModule, EpicActionRecognitionSystem

LOG = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="tsn_rgb")
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    system = EpicActionRecognitionSystem(cfg)
    LOG.info("EpicActionRecognitionSystem initialized")
    if not cfg.get("log_graph", True):
        # MTRN can't be traced due to the model stochasticity so causes a JIT tracer
        # error, we allow you to prevent the tracer from running to log the graph when
        # the summary writer is created
        try:
            delattr(system, "example_input_array")
        except AttributeError:
            pass
    data_module = EpicActionRecogintionDataModule(cfg)
    LOG.info("EpicActionRecognitionDataModule initialized")
    checkpoint_callback = ModelCheckpoint(save_top_k=None, monitor=None)
    # with ipdb.launch_ipdb_on_exception():
    trainer = Trainer(
        callbacks=[],
        checkpoint_callback=checkpoint_callback,
        max_steps=5,
        **cfg.trainer
    )
    LOG.info("Starting training....")
    trainer.fit(system, datamodule=data_module)
    LOG.info("Training completed!")


if __name__ == "__main__":
    main()
