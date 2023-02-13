import os
from datetime import datetime

import lightning as L
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.datasets.classification.adult import _CAT_FEATURES, _target, load_fitted_model
from deepchecks.tabular.suites import data_integrity, model_evaluation, train_test_validation
from lightning.app.storage import Path, Payload


class GetDataWork(L.LightningWork):
    """This component is responsible to download some data and store them with a PayLoad."""

    def __init__(self):
        super().__init__()
        self.df_train = None
        self.df_test = None

    def run(self):
        print("Starting data collection...")
        df_train, df_test = adult.load_data(data_format="Dataframe")
        self.df_train = Payload(df_train)
        self.df_test = Payload(df_test)
        print("Finished data collection.")


class DataIntegrityCheck(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.train_results_path = None
        self.test_results_path = None

    def run(self, df_train: Payload, df_test: Payload):
        print("Starting Data Integrity Check....")
        df_train = Dataset(df_train.value, label=_target, cat_features=_CAT_FEATURES)
        df_test = Dataset(df_test.value, label=_target, cat_features=_CAT_FEATURES)

        train_results = data_integrity().run(df_train)
        test_results = data_integrity().run(df_test)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_results_path = os.path.join(self.dir_path, f"train_integrity_{run_time}.html")
        test_results_path = os.path.join(self.dir_path, f"test_integrity_{run_time}.html")

        train_results.save_as_html(train_results_path)
        test_results.save_as_html(test_results_path)

        self.train_results_path = Path(train_results_path)
        self.test_results_path = Path(test_results_path)
        print("Finished data integrity check.")


class TrainTestValidation(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.train_test_validation_results_path = None

    def run(self, df_train: Payload, df_test: Payload):
        print("Starting train test validation suite...")
        df_train = Dataset(df_train.value, label=_target, cat_features=_CAT_FEATURES)
        df_test = Dataset(df_test.value, label=_target, cat_features=_CAT_FEATURES)

        train_test_validation_results = train_test_validation().run(df_train, df_test)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_test_validation_results_path = os.path.join(self.dir_path, f"train_test_validation_{run_time}.html")

        train_test_validation_results.save_as_html(train_test_validation_results_path)

        self.train_test_validation_results_path = Path(train_test_validation_results_path)
        print("Finished train test validation suite.")


class ModelEvaluation(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.dir_path = "suite_results"
        self.evaluation_results_path = None

    def run(self, df_train: Payload, df_test: Payload):
        print("Starting model evaluation...")
        model = load_fitted_model()

        df_train = Dataset(df_train.value, label=_target, cat_features=_CAT_FEATURES)
        df_test = Dataset(df_test.value, label=_target, cat_features=_CAT_FEATURES)

        evaluation_results = model_evaluation().run(df_train, df_test, model)

        os.makedirs(self.dir_path, exist_ok=True)

        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        evaluation_results_path = os.path.join(self.dir_path, f"model_evaluation_{run_time}.html")

        evaluation_results.save_as_html(evaluation_results_path)
        self.evaluation_results_path = Path(evaluation_results_path)
        print("Finished model evaluation.")
