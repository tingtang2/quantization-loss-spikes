import hydra
from train import train_and_evaluate
from configs import resolver_setup
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)
    print(OmegaConf.to_yaml(c))
    train_and_evaluate(c)


if __name__ == '__main__':
    main()
