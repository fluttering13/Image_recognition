from functools import partial, update_wrapper
from types import MethodType
from typing import Any

from pathlib import Path
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType

from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



from anomalib.models.fastflow.lightning_model import Fastflow

from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    MetricVisualizerCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode

root=Path.cwd() / "datasets"
task = TaskType.SEGMENTATION
# dataset_download_info = DownloadInfo(
#     name="cubes.zip",
#     url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/cubes.zip",
#     hash="e6e067f9e0979a4d190dd2cb1db227d7",
# )

# download_and_extract(root=root, info=dataset_download_info)
def configure_optimizers(lightning_module: LightningModule, optimizer: Optimizer) -> Any:  # pylint: disable=W0613,W0621
    """Override to customize the LightningModule.configure_optimizers` method."""
    return optimizer    



if __name__=='__main__':
    datamodule = Folder(
        root=root / "cubes",
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        task=TaskType.CLASSIFICATION,
    )

    datamodule.setup()  # Split the data to train/val/test/prediction sets.
    datamodule.prepare_data()  # Create train/val/test/predic dataloaders

    model = Fastflow(input_size=(256, 256), backbone="resnet18", flow_steps=8)
    optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)

    fn = partial(configure_optimizers, optimizer=optimizer)
    update_wrapper(fn, configure_optimizers)  # necessary for `is_overridden`
    model.configure_optimizers = MethodType(fn, model)

    
    callbacks = [
        MetricsConfigurationCallback(
            task=task,
            image_metrics=["AUROC"],
            pixel_metrics=["AUROC"],
        ),
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
        ),
        EarlyStopping(
            monitor="pixel_AUROC",
            mode="max",
            patience=3,
        ),
        PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
        ),
        ImageVisualizerCallback(mode="full", task=task, image_save_path="./results/images"),
        MetricVisualizerCallback(mode="full", task=task, image_save_path="./results/images"),]

    trainer = Trainer(
        callbacks=callbacks,
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        max_epochs=100,
        logger=False,
    )

    trainer.fit(datamodule=datamodule, model=model)
    trainer.test(datamodule=datamodule, model=model)
    # trainer = Trainer(
    #     callbacks=callbacks,
    #     accelerator="auto",
    #     auto_scale_batch_size=False,
    #     check_val_every_n_epoch=1,
    #     devices=1,
    #     gpus=None,
    #     max_epochs=1,
    #     num_sanity_val_steps=0,
    #     val_check_interval=1.0,
    # )


    # trainer.fit(model=model, datamodule=datamodule)

    # # Validation
    # test_results = trainer.test(model=model, datamodule=datamodule)
