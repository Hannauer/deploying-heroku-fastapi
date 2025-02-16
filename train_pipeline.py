import logging
import hydra
from starter.ml.data import clean_data
from starter.train_model import load_data, split_data, train_save_model
from starter.model_validation import model_validate
from omegaconf import DictConfig

_steps = [
    "clean_data",
    "train_model",
    "check_score"
]


@hydra.main(config_name="config.yaml")
def go(config: DictConfig):
    """
    Run pipeline stages
    """
    logging.basicConfig(level=logging.INFO)

    root_path = hydra.utils.get_original_cwd()
    print(root_path)
    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    #cat_features = config['data']['cat_features']

    if "clean_data" in active_steps:
        logging.info("Cleaning and saving raw_data")
        clean_data(root_path)

    df = load_data(root_path+'/data/census_clean.csv')
    df_train, df_test = split_data(df)

    if "train_model" in active_steps:
        logging.info("Train model step  has started...")
        train_save_model(df_train, root_path)

    if "check_score" in active_steps:
        logging.info("Score check procedure started")
        model_validate(df_test, root_path)


if __name__ == "__main__":
    """
    Main entrypoint
    """
    go()