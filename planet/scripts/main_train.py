import lightning as L
from planet.train import LightningPlaNet, DataModule

if __name__ == "__main__":

    ### instantiate model and datamodule
    model = LightningPlaNet()
    datamodule = DataModule(dataset_path="planet_data_sample.h5")

    ### train the model
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="mps",
        devices=1,
    )
    trainer.fit(model=model, datamodule=datamodule)

    ### save model + scaler for inference
