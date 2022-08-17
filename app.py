import lightning as L
from lightning.app.structures import List

from lightning_deepchecks import (
    DataIntegrityCheck,
    GetDataWork,
    ModelEvaluation,
    TrainTestValidation,
)


class DeepchecksDAG(L.LightningFlow):

    """This flow is a DAG with Deepchecks components."""

    def __init__(self, **dag_kwargs):
        super().__init__()
        # Step 1: Create a work to get the data.
        self.data_collector = GetDataWork()

        # Step 2: Create a work for data integrity check
        self.data_integrity_check = DataIntegrityCheck()

        # Step 3: Create a work for train test validation suite
        self.train_test_validation = TrainTestValidation()

        # Step 4: Create a work for model evaluation
        self.model_evaluation = ModelEvaluation()

        self.has_completed = False

    def run(self):
        # Step 1: Download and load data.
        self.data_collector.run()

        # Step 2: Do data integrity check.
        self.data_integrity_check.run(
            df_train=self.data_collector.df_train,
            df_test=self.data_collector.df_test,
        )
        self.data_integrity_check.stop()

        # Step 3: Run the train test validation suite
        self.train_test_validation.run(
            df_train=self.data_collector.df_train,
            df_test=self.data_collector.df_test,
        )
        self.train_test_validation.stop()

        # Step 4: Start model evaluation
        self.model_evaluation.run(
            df_train=self.data_collector.df_train,
            df_test=self.data_collector.df_test,
        )
        if self.model_evaluation.evaluation_results_path:
            self.has_completed = True


class ScheduledDAG(L.LightningFlow):
    def __init__(self, dag_cls, **dag_kwargs):
        super().__init__()
        self.dags = List()
        self._dag_cls = dag_cls
        self.dag_kwargs = dag_kwargs

    def run(self):
        """Example of scheduling an infinite number of DAG runs continuously."""

        # Step 1: Every hour, create and launch a new DAG.
        if self.schedule("0 * * * *"):
            print("Launching a new DAG")
            self.dags.append(self._dag_cls(**self.dag_kwargs))

        for dag in self.dags:
            if not dag.has_completed:
                dag.run()


app = L.LightningApp(ScheduledDAG(DeepchecksDAG))
