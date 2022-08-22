import importlib
import os
from datetime import datetime

import deepchecks
import lightning as L
from deepchecks.tabular import Dataset
from lightning.app.storage import Path, Payload


class GetDataWork(L.LightningWork):

    """This component is responsible to download some data and store them with a PayLoad."""

    def __init__(self):
        super().__init__()
        self.df_train = None
        self.df_test = None

    def run(self, config: dict):
        print(f"Starting {config['dataset']} data collection...")
        df_train, df_test = eval(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}.load_data(data_format='Dataframe')"
        )
        self.df_train = Payload(df_train)
        self.df_test = Payload(df_test)
        print(self.df_train.value)
        print("Finished data collection.")


class DataIntegrityCheck(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.train_results_path = None
        self.test_results_path = None

    def run(self, df_train: Payload, df_test: Payload, config: dict):
        print(f"Starting {config['dataset']} Data Integrity Check....")
        deepchecks_suites_module = importlib.import_module(
            f"deepchecks.{config['domain']}.suites"
        )
        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )

        df_train = Dataset(
            df_train.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )
        df_test = Dataset(
            df_test.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )

        train_results = deepchecks_suites_module.data_integrity().run(df_train)
        test_results = deepchecks_suites_module.data_integrity().run(df_test)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_results_path = os.path.join(
            self.dir_path, f"{config['dataset']}_train_integrity_{run_time}.html"
        )
        test_results_path = os.path.join(
            self.dir_path, f"{config['dataset']}_test_integrity_{run_time}.html"
        )

        train_results.save_as_html(train_results_path, as_widget=False)
        test_results.save_as_html(test_results_path, as_widget=False)

        self.train_results_path = Path(train_results_path)
        self.test_results_path = Path(test_results_path)
        print("Finished data integrity check.")


class TrainTestValidation(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.train_test_validation_results_path = None

    def run(self, df_train: Payload, df_test: Payload, config: dict):
        print(f"Starting {config['dataset']} train test validation suite...")

        deepchecks_suites_module = importlib.import_module(
            f"deepchecks.{config['domain']}.suites"
        )
        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )

        df_train = Dataset(
            df_train.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )
        df_test = Dataset(
            df_test.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )

        train_test_validation_results = (
            deepchecks_suites_module.train_test_validation().run(df_train, df_test)
        )

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_test_validation_results_path = os.path.join(
            self.dir_path, f"{config['dataset']}_train_test_validation_{run_time}.html"
        )

        train_test_validation_results.save_as_html(
            train_test_validation_results_path, as_widget=False
        )

        self.train_test_validation_results_path = Path(
            train_test_validation_results_path
        )
        print("Finished train test validation suite.")


class ModelEvaluation(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.evaluation_results_path = None

    def run(self, df_train: Payload, df_test: Payload, config: dict):
        print(f"Starting {config['dataset']} model evaluation suite...")
        deepchecks_suites_module = importlib.import_module(
            f"deepchecks.{config['domain']}.suites"
        )

        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )

        model = deepchecks_module.load_fitted_model()

        df_train = Dataset(
            df_train.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )
        df_test = Dataset(
            df_test.value,
            label=deepchecks_module._target,
            cat_features=deepchecks_module._CAT_FEATURES,
        )

        evaluation_results = deepchecks_suites_module.model_evaluation().run(
            df_train, df_test, model
        )

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        evaluation_results_path = os.path.join(
            self.dir_path, f"{config['dataset']}_model_evaluation_{run_time}.html"
        )

        evaluation_results.save_as_html(evaluation_results_path, as_widget=False)
        self.evaluation_results_path = Path(evaluation_results_path)

        print("Finished model evaluation suite.")
