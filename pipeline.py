import kfp.dsl as dsl
import kfp.gcp as gcp


class LoadData(dsl.ContainerOp):
    def __init__(self, name, bucket, cutoff_year):
        super(LoadData, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image="gcr.io/<project>/<image-name>/cpu:v1",
            command=["python3", "run_preprocess.py"],
            arguments=["--bucket", bucket, "--cutoff_year", cutoff_year, "--kfp"],
            file_outputs={"blob-path": "/blob_path.txt"},
        )


class Preprocessing(dsl.ContainerOp):
    def __init__(self, name, blob_path, tag, bucket, model):
        super(Preprocessing, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image="gcr.io/<project>/<image-name>/cpu:v1",
            command=["python3", "run_train.py"],
            arguments=[
                "--tag",
                tag,
                "--blob_path",
                blob_path,
                "--bucket",
                bucket,
                "--model",
                model,
                "--kfp",
            ],
            file_outputs={
                "mlpipeline_metrics": "/mlpipeline-metrics.json",
                "accuracy": "/tmp/accuracy",
            },
        )


class Training(dsl.ContainerOp):
    def __init__(self, name, tag, bucket):
        super(Training, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image="gcr.io/<project>/<image-name>/cpu:v1",
            command=["python3", "run_deploy.py"],
            arguments=[
                "--tag",
                tag,
                "--bucket",
                bucket,
            ],
        )


class Serving(dsl.ContainerOp):
    def __init__(self, name, tag, bucket):
        super(Serving, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image="gcr.io/<project>/<image-name>/cpu:v1",
            command=["python3", "run_deploy.py"],
            arguments=[
                "--tag",
                tag,
                "--bucket",
                bucket,
            ],
        )


@dsl.pipeline(name="financial time series", description="Train Financial Time Series")
def preprocess_train_deploy(
    bucket: str = "<bucket>",
    cutoff_year: str = "2010",
    tag: str = "4",
    model: str = "DeepModel",
):
    """Pipeline to train financial time series model"""
    preprocess_op = Preprocess("preprocess", bucket, cutoff_year).apply(
        gcp.use_gcp_secret("user-gcp-sa")
    )
    # pylint: disable=unused-variable
    train_op = Train("train", preprocess_op.output, tag, bucket, model).apply(
        gcp.use_gcp_secret("user-gcp-sa")
    )
    with dsl.Condition(train_op.outputs["accuracy"] > 0.7):
        deploy_op = Deploy("deploy", tag, bucket).apply(
            gcp.use_gcp_secret("user-gcp-sa")
        )


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(preprocess_train_deploy, __file__ + ".tar.gz")
