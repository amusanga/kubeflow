import kfp  # the Pipelines SDK.
from kfp import compiler
import kfp.dsl as dsl
from kfp import components


OUTPU_DIR = "./data"

loadData_op = components.load_component_from_file(
    "/home/aime/kubeflow/LoadData/component.yaml"  # pylint: disable=line-too-long
)


@dsl.pipeline(
    name="CIFAR-10 experiment",
    description="Experimental kubeflow pipeline on CIFAR10 Dataset",
)
def preprocess_train_deploy():

    loadData = loadData_op()


if __name__ == "__main__":

    compiler.Compiler().compile(preprocess_train_deploy, "First_pipeline" + ".tar.gz")
    # client = kfp.Client()
    # client.create_run_from_pipeline_func(
    #     preprocess_train_deploy, arguments={"outputPath": "./data"}
    # )
