import deepchecks
import lightning as L
import streamlit as st
from deepchecks.vision import datasets
from lightning.app.frontend.stream_lit import StreamlitFrontend

DOMAINS = ["Tabular", "Vision"]
SUITES = ["Data Integrity", "Train Test Validation", "Model Evaluation"]


class DeepchecksFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()

    def run(self):
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

    domain = st.sidebar.selectbox("Select a domain", DOMAINS)

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


app = L.LightningApp(DeepchecksFlow())
