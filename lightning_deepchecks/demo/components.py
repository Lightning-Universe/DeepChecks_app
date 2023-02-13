import importlib
import os
from dataclasses import dataclass
from datetime import datetime

import lightning as L
from deepchecks.tabular import Dataset
from lightning.app import BuildConfig
from lightning.app.storage import Payload


@dataclass
class CustomBuildConfig(BuildConfig):
    def build_commands(self):
        return [
            "pip uninstall -y opencv-python",
            "pip uninstall -y opencv-python-headless",
            "pip install opencv-python-headless==4.5.5.64",
        ]


class GetDataWork(L.LightningWork):
    """This component is responsible to download some data and store them with a PayLoad."""

    def __init__(self):
        cloud_build_config = CustomBuildConfig()

        super().__init__(cloud_build_config=cloud_build_config)
        self.df_train = None
        self.df_test = None

    def run(self, config: dict):

        print(f"Starting {config['dataset']} data collection...")
        if config["domain"] == "vision":
            df_train = eval(
                f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}.load_dataset(train=True, object_type='VisionData')"
            )
            df_test = eval(
                f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}.load_dataset(train=True, object_type='VisionData')"
            )
        else:
            df_train, df_test = eval(
                f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}.load_data(data_format='Dataframe')"
            )
        self.df_train = Payload(df_train)
        self.df_test = Payload(df_test)
        print(self.df_train.value)
        print("Finished data collection.")


class DataIntegrityCheck(L.LightningWork):
    def __init__(self):
        cloud_build_config = CustomBuildConfig()

        super().__init__(cloud_build_config=cloud_build_config)
        self.dir_path = "suite_results"
        self.train_results_path = None
        self.test_results_path = None
        self.processed = False

    def run(self, df_train: Payload, df_test: Payload, config: dict):

        print(f"Starting {config['dataset']} Data Integrity Check....")
        self.train_results_path, self.test_results_path = None, None
        self.processed = False

        deepchecks_suites_module = importlib.import_module(f"deepchecks.{config['domain']}.suites")
        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )
        if config["domain"] == "vision":
            df_train, df_test = df_train.value, df_test.value
        else:
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
        train_results_path = os.path.join(self.dir_path, f"{config['dataset']}_train_integrity_{run_time}.html")
        test_results_path = os.path.join(self.dir_path, f"{config['dataset']}_test_integrity_{run_time}.html")

        train_results.save_as_html(train_results_path, as_widget=False)
        test_results.save_as_html(test_results_path, as_widget=False)

        with open(train_results_path, encoding="utf-8") as f:
            self.train_results_path = f.read()

        with open(test_results_path, encoding="utf-8") as f:
            self.test_results_path = f.read()

        self.processed = True
        print("Finished data integrity check.")


class TrainTestValidation(L.LightningWork):
    def __init__(self):
        cloud_build_config = CustomBuildConfig()

        super().__init__(cloud_build_config=cloud_build_config)
        self.dir_path = "suite_results"
        self.results_path = None
        self.processed = False

    def run(self, df_train: Payload, df_test: Payload, config: dict):

        print(f"Starting {config['dataset']} train test validation suite...")
        self.results_path = None
        self.processed = False

        deepchecks_suites_module = importlib.import_module(f"deepchecks.{config['domain']}.suites")
        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )

        if config["domain"] == "vision":
            df_train, df_test = df_train.value, df_test.value
        else:
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

        train_test_validation_results = deepchecks_suites_module.train_test_validation().run(df_train, df_test)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = os.path.join(self.dir_path, f"{config['dataset']}_train_test_validation_{run_time}.html")

        train_test_validation_results.save_as_html(results_path, as_widget=False)

        with open(results_path, encoding="utf-8") as f:
            self.results_path = f.read()

        self.processed = True

        print("Finished train test validation suite.")


class ModelEvaluation(L.LightningWork):
    def __init__(self):
        cloud_build_config = CustomBuildConfig()

        super().__init__(cloud_build_config=cloud_build_config)
        self.dir_path = "suite_results"
        self.results_path = None
        self.processed = False

    def run(self, df_train: Payload, df_test: Payload, config: dict):

        print(f"Starting {config['dataset']} model evaluation suite...")
        self.results_path = None
        self.processed = False

        if config["dataset"] == "coco":
            ## TODO: fix this
            results = "<h1>Model Evaluation Suite for COCO dataset is not supported yet.</h1>"
            os.makedirs(self.dir_path, exist_ok=True)

            run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_path = os.path.join(self.dir_path, f"{config['dataset']}_model_evaluation_{run_time}.html")
            with open(results_path, "w") as file:
                file.write(results)

            self.results_path = results
            self.processed = True

            print("Finished model evaluation suite.")
            return

        deepchecks_suites_module = importlib.import_module(f"deepchecks.{config['domain']}.suites")

        deepchecks_module = importlib.import_module(
            f"deepchecks.{config['domain']}.datasets.{config['algo']}.{config['dataset']}"
        )

        if config["domain"] == "vision":
            model = deepchecks_module.load_model()
            df_train, df_test = df_train.value, df_test.value
        else:
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

        evaluation_results = deepchecks_suites_module.model_evaluation().run(df_train, df_test, model)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = os.path.join(self.dir_path, f"{config['dataset']}_model_evaluation_{run_time}.html")

        evaluation_results.save_as_html(results_path, as_widget=False)
        with open(results_path, encoding="utf-8") as f:
            self.results_path = f.read()
        self.processed = True

        print("Finished model evaluation suite.")
