import kfp
import kfp.components as comp

create_step_get_lines = comp.load_component_from_text(
    """
name: Get Lines
description: Load CIFAR-10 data from Torch Vision and do preprocessing on it. 

outputs:
- {name: output_path, type: string, description: ' path to the output file'}

implementation:
  container:
    image: docker.io/amusanga/experiment:latest
    command: [
      python3,
      program.py,
      --output-path, {outputPath: output_path}
    ]
"""
)

# create_step_get_lines is a "factory function" that accepts the arguments
# for the component's inputs and output paths and returns a pipeline step
# (ContainerOp instance).
#
# To inspect the get_lines_op function in Jupyter Notebook, enter
# "get_lines_op(" in a cell and press Shift+Tab.
# You can also get help by entering `help(get_lines_op)`, `get_lines_op?`,
# or `get_lines_op??`.

# Create a simple component using only bash commands. The output of this component
# can be passed to a downstream component that accepts an input with the same type.
create_step_write_lines = comp.load_component_from_text(
    """
name: Write Lines
description: Writes text to a file.

inputs:
- {name: text, type: String}

outputs:
- {name: data, type: Data}

implementation:
  container:
    image: busybox
    command:
    - sh
    - -c
    - |
      mkdir -p "$(dirname "$1")"
      echo "$0" > "$1"
    args:
    - {inputValue: text}
    - {outputPath: data}
"""
)

# Define your pipeline
def my_pipeline():
    write_lines_step = create_step_write_lines(
        text="one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten"
    )

    get_lines_step = create_step_get_lines()


# If you run this command on a Jupyter notebook running on Kubeflow,
# you can exclude the host parameter.
# client = kfp.Client()
client = kfp.Client()

# Compile, upload, and submit this pipeline for execution.
client.create_run_from_pipeline_func(my_pipeline, arguments={})
