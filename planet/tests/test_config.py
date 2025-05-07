import yaml

from planet.config import PlaNetConfig


def test_config():

    config = PlaNetConfig(
        batch_size=128,
        branch_in_dim=10000,
    )

    cfg = yaml.safe_load(open("planet/tests/data/config.yml"))
    config = PlaNetConfig.from_dict(cfg)
