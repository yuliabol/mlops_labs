import os
from airflow.models import DagBag

def test_dag_import_errors():
    """ Verify that all DAGs in the dags directory can be loaded without errors. """
    dag_path = os.path.abspath("dags")
    dag_bag = DagBag(dag_folder=dag_path, include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"

def test_ml_pipeline_dag_exists():
    """ Verify that the specific 'ml_training_pipeline' DAG exists. """
    dag_path = os.path.abspath("dags")
    dag_bag = DagBag(dag_folder=dag_path, include_examples=False)
    assert "ml_training_pipeline" in dag_bag.dags
