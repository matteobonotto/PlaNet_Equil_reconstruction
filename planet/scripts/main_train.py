import lightning as L
from argparse import ArgumentParser, Namespace
import yaml
from planet.train import LightningPlaNet, DataModule


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('config', help="path to config file")
    args, _ = parser.parse_known_args()
    return args






def load_config(path: str) -> PlaNetConfig:
    config_dict = yaml.safe_load(open(path, "r"))
    return PlaNetConfig.from_dict(config_dict=config_dict)




if __name__ == "__main__":

    args = parse_arguments()


    ### instantiate model and datamodule
    model = LightningPlaNet(is_physics_informed=is_physics_informed)
    datamodule = DataModule(dataset_path=dataset_path)

    ### train the model
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="mps",
        devices=1,
    )
    trainer.fit(model=model, datamodule=datamodule)

    ### save model + scaler for inference
