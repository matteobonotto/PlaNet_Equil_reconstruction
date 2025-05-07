from planet.config import PlaNetConfig


def test_config():
    config = PlaNetConfig()

    config_2 = PlaNetConfig.from_dict({
        'batch_size': 128,
        "log_to_wandb": True
    })


test_config()