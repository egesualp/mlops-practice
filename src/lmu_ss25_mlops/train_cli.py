from lightning.pytorch.cli import LightningCLI
from model_light import MyAwesomeButLightModel

if __name__ == "__main__":
    cli = LightningCLI(
        MyAwesomeButLightModel,
        save_config_overwrite=True
    )