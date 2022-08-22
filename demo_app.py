import importlib
import os
from datetime import datetime

import deepchecks
import lightning as L
import streamlit as st
from deepchecks.tabular import Dataset, datasets
from deepchecks.vision import datasets
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.storage import Path, Payload
from lightning.app.structures import List

from lightning_deepchecks.components import GetDataWork

DOMAINS = ["Tabular", "Vision"]
SUITES = ["Data Integrity", "Train Test Validation", "Model Evaluation"]


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
        deepchecks_suites_module = importlib.import_module(
            f"deepchecks.{config['domain']}.suites"
        )

        print(f"Starting {config['dataset']} Data Integrity Check....")
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

        train_results.save_as_html(train_results_path)
        test_results.save_as_html(test_results_path)

        self.train_results_path = Path(train_results_path)
        self.test_results_path = Path(test_results_path)
        print("Finished data integrity check.")


class DeepchecksSuites(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.suites = List()
        self.data_collector = GetDataWork()
        self.data_integrity_check = DataIntegrityCheck()

    def run(self, config: dict):
        self.data_collector.run(config)

        for suite in config["suites"]:
            if suite not in SUITES:
                raise ValueError(
                    f"{suite} is not supported. Supported suites are {SUITES}"
                )
            if suite == "Data Integrity":
                self.data_integrity_check.run(
                    self.data_collector.df_train, self.data_collector.df_test, config
                )
            # elif suite == "Train Test Validation":
            #     self.suites.append(TrainTestValidation())
            # elif suite == "Model Evaluation":
            #     self.suites.append(ModelEvaluation())


class DeepchecksFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.deepchecks_config = None
        self.deepchecks_suites = DeepchecksSuites()

    def run(self):
        if self.deepchecks_config is not None:
            self.deepchecks_suites.run(
                config=self.deepchecks_config,
            )
        pass

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_deepchecks_flow)


def get_datasets(domain: str):
    if domain not in DOMAINS:
        raise ValueError(f"{domain} is not supported. Supported domains are {DOMAINS}")

    DATASETS_MODULE = f"deepchecks.{domain.lower()}.datasets"
    ALGOS = eval(f"{DATASETS_MODULE}.__all__")

    DATASETS = {}
    for algo in ALGOS:
        DATASETS[algo] = eval(f"{DATASETS_MODULE}.{algo}.__all__")

    return DATASETS


def render_deepchecks_flow(state):

    st.title("Welcome to Deepchecks' Demo! :rocket:")

    domain = st.sidebar.selectbox("Select a domain", DOMAINS, index=0)

    algo_datasets = get_datasets(domain)

    datasets = [
        dataset for dataset_list in algo_datasets.values() for dataset in dataset_list
    ]

    dataset = st.sidebar.selectbox("Select a dataset", datasets)

    st.write(domain)
    st.write(dataset)

    suites = st.sidebar.multiselect("Select suites", SUITES, default=SUITES)

    run = st.sidebar.button("Run", disabled=not bool(suites))

    st.write(run)
    if run:
        algo = None
        for dc_algo, datasets in algo_datasets.items():
            if dataset in datasets:
                algo = dc_algo
                break
        state.deepchecks_config = {
            "domain": domain.lower(),
            "algo": algo,
            "dataset": dataset,
            "suites": suites,
        }
        st.write(state.deepchecks_config)


app = L.LightningApp(DeepchecksFlow())
