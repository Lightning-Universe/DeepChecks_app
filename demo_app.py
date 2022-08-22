import deepchecks
import lightning as L
import streamlit as st
from deepchecks.tabular import Dataset, datasets
from deepchecks.vision import datasets
from lightning.app.frontend.stream_lit import StreamlitFrontend

from lightning_deepchecks.demo.components import (
    DataIntegrityCheck,
    GetDataWork,
    ModelEvaluation,
    TrainTestValidation,
)

DOMAINS = ["Tabular", "Vision"]
SUITES = ["Data Integrity", "Train Test Validation", "Model Evaluation"]


class DeepchecksSuites(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.data_collector = GetDataWork()
        self.data_integrity_check = DataIntegrityCheck()
        self.train_test_validation = TrainTestValidation()
        self.model_evaluation = ModelEvaluation()

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
                self.data_integrity_check.stop()
            elif suite == "Train Test Validation":
                self.train_test_validation.run(
                    self.data_collector.df_train, self.data_collector.df_test, config
                )
                self.train_test_validation.stop()
            elif suite == "Model Evaluation":
                self.model_evaluation.run(
                    self.data_collector.df_train, self.data_collector.df_test, config
                )
                self.model_evaluation.stop()


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
    st.caption(
        "Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort. Test Suites for Validating ML Models & Data."
    )

    domain = st.sidebar.selectbox("Select a domain", DOMAINS, index=0)

    algo_datasets = get_datasets(domain)

    datasets = [
        dataset for dataset_list in algo_datasets.values() for dataset in dataset_list
    ]

    dataset = st.sidebar.selectbox("Select a dataset", datasets)

    # st.write(domain)
    # st.write(dataset)

    suites = st.sidebar.multiselect("Select suites", SUITES, default=SUITES)

    run = st.sidebar.button("Run", disabled=not bool(suites))

    # st.write(run)
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
        # st.write(state.deepchecks_config)

    selected_suites = st.selectbox("View Suite Results For", SUITES)

    import streamlit.components.v1 as components

    result = open("output.html").read()

    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """

    components.html(
        TEMPLATE_WRAPPER.format(body=result, height=1000), height=1000, width=1000
    )


app = L.LightningApp(DeepchecksFlow())
