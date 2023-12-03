from anomalib.data.utils import DownloadInfo, download_and_extract

from pathlib import Path

from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType

from anomalib.models import Padim

from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode

root=Path.cwd() / "datasets"

# dataset_download_info = DownloadInfo(
#     name="cubes.zip",
#     url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/cubes.zip",
#     hash="e6e067f9e0979a4d190dd2cb1db227d7",
# )

# download_and_extract(root=root, info=dataset_download_info)


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

    model = Padim(
        input_size=(256, 256),
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )

    callbacks = [
        MetricsConfigurationCallback(
            task=TaskType.CLASSIFICATION,
            image_metrics=["AUROC"],
        ),
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
        ),
        PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
        ),
        MinMaxNormalizationCallback(),
        ExportCallback(
            input_size=(256, 256),
            dirpath=str(Path.cwd()),
            filename="model",
            export_mode=ExportMode.OPENVINO,
        ),
    ]

    trainer = Trainer(
        callbacks=callbacks,
        accelerator="auto",
        auto_scale_batch_size=False,
        check_val_every_n_epoch=1,
        devices=1,
        gpus=None,
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )


    trainer.fit(model=model, datamodule=datamodule)

    # Validation
    test_results = trainer.test(model=model, datamodule=datamodule)
