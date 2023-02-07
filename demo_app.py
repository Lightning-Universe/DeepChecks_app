import deepchecks
import lightning as L
import streamlit as st
import streamlit.components.v1 as components
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.structures import List

from lightning_deepchecks.demo.components import (
    DataIntegrityCheck,
    GetDataWork,
    ModelEvaluation,
    TrainTestValidation,
)

DOMAINS = ["Tabular", "Vision"]
SUITES = ["Data Integrity", "Train Test Validation", "Model Evaluation"]

DATASETS = {
    "Tabular": {
        "classification": [
            "iris",
            "breast_cancer",
            "phishing",
            "adult",
            "lending_club",
        ],
        "regression": ["avocado", "wine_quality"],
    },
    "Vision": {
        "classification": ["mnist"],
        "detection": ["coco"],
    },
}


class DeepchecksSuites(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.data_collector = GetDataWork()
        self.data_integrity_check = DataIntegrityCheck()
        self.train_test_validation = TrainTestValidation()
        self.model_evaluation = ModelEvaluation()
        self.suites = List()

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
            elif suite == "Train Test Validation":
                self.train_test_validation.run(
                    self.data_collector.df_train, self.data_collector.df_test, config
                )
            elif suite == "Model Evaluation":
                self.model_evaluation.run(
                    self.data_collector.df_train, self.data_collector.df_test, config
                )


class DeepchecksFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.deepchecks_config = None
        self.deepchecks_suites = DeepchecksSuites()
        self.processed = False

    def run(self):
        if self.deepchecks_config is not None:
            self.deepchecks_suites.run(
                config=self.deepchecks_config,
            )
            self.deepchecks_config = None
            self.processed = True

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_deepchecks_flow)


# def get_datasets(domain: str):
#     if domain not in DOMAINS:
#         raise ValueError(f"{domain} is not supported. Supported domains are {DOMAINS}")

#     DATASETS_MODULE = f"deepchecks.{domain.lower()}.datasets"
#     ALGOS = eval(f"{DATASETS_MODULE}.__all__")

#     DATASETS = {}
#     for algo in ALGOS:
#         DATASETS[algo] = eval(f"{DATASETS_MODULE}.{algo}.__all__")

#     ## TODO: add support for YOLO Dataset as well
#     if domain == "Vision":
#         DATASETS["detection"] = ["coco"]

#     return DATASETS


def render_deepchecks_flow(state):
    st.title("Welcome to Deepchecks' Demo! :rocket:")
    st.caption(
        "Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort. Test Suites for Validating ML Models & Data."
    )

    domain = st.sidebar.selectbox("Select a domain", DOMAINS, index=0)

    datasets = [
        dataset
        for dataset_list in DATASETS[domain].values()
        for dataset in dataset_list
    ]

    dataset = st.sidebar.selectbox("Select a dataset", datasets)
    suites = st.sidebar.multiselect("Select suites", SUITES, default=SUITES)

    run = st.sidebar.button("Run", disabled=not bool(suites))

    if run:
        algo = None
        for dc_algo, datasets in DATASETS[domain].items():
            if dataset in datasets:
                algo = dc_algo
                break
        state.deepchecks_config = {
            "domain": domain.lower(),
            "algo": algo,
            "dataset": dataset,
            "suites": suites,
        }
        state.processed = False

    selected_suite = st.selectbox("View Suite Results For", suites)

    display_results = None

    if (
        selected_suite == "Data Integrity"
        and state.deepchecks_suites.data_integrity_check.processed
    ):
        display_results = (
            state.deepchecks_suites.data_integrity_check.train_results_path
        )
    elif (
        selected_suite == "Train Test Validation"
        and state.deepchecks_suites.train_test_validation.processed
    ):
        display_results = state.deepchecks_suites.train_test_validation.results_path
    elif (
        selected_suite == "Model Evaluation"
        and state.deepchecks_suites.model_evaluation.processed
    ):
        display_results = state.deepchecks_suites.model_evaluation.results_path

    if display_results is not None:
        TEMPLATE_WRAPPER = """
        <div style="height:{height}px;overflow-y:auto;position:relative;">
            {body}
        </div>
        """

        components.html(
            TEMPLATE_WRAPPER.format(body=display_results, height=1000),
            height=1000,
            width=1000,
        )


app = L.LightningApp(DeepchecksFlow())
