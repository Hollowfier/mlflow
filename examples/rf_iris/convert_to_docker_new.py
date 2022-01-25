'''
This script reads a ONNX model, uploads it to ML Flow and 
converts it into a Docker Image using ML Flow commands. 
Later the Docker Image is saved at the same location of the input file.  

Required libraries: 
os - to execute cmd line scripts
argparse - read files from arguments
platform - to read installed python version (required in MLmodel file for mlflow)
mlflow - log/upload model onto mlflow
onnx - to read installed onnx version (required in MLmodel file for mlflow)

Command to run program
python convert_to_docker.py -f <complete_path_of_ONNX_model>

To load docker image from saved ".tar" file
docker load -i <path_to_tar_file>
'''

import argparse
import os
import platform
import sys

import mlflow
import onnx
import torch
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# Contents of the "MLmodel" file that has to be written with the ONNX model
def get_file_contents(file_name):
    contents = ""
    two_spaces = "  "
    four_spaces = 2*two_spaces
    contents += "artifact_path: model\nflavors:\n"
    contents += two_spaces + "onnx:\n" 
    contents += four_spaces + "data: " + file_name + "\n" 
    contents += four_spaces + "onnx_version: " + onnx.__version__ + "\n"
    contents += two_spaces + "python_function:\n"
    contents += four_spaces + "data: model.onnx\n"
    #TODO: read about different environment types in MLmodel file
    contents += four_spaces + "env: conda.yaml\n"
    contents += four_spaces + "loader_module: mlflow.onnx\n"
    contents += four_spaces + "python_version: " + platform.python_version()

    return contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Read input ONNX file location
    parser.add_argument("--file", "-f")
    parser.add_argument("--type", "-t")
    #TODO: Relative Path?
    arguments = parser.parse_args()
    
    separator = "\\"
    image_name = "_dckrimg"

    # Convert model to Onnx
    if arguments.type == None:
        sys.exit("Enter the type of model. 1: Tensorflow 2: PyTorch 3: Sklearn")
    elif arguments.type == 1:
        os.system('cmd /c python -m tf2onnx.convert --saved-model ' + arguments.file + ' --output' + 'tfmodel.onnx  --opset 12')
    elif arguments.type == 2:
        # Export the model
        torch.onnx.export('''INSERT MODEL HERE''',      # model being run
                  '''INSERT INPUT HERE''',              # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12) 
    elif arguments.type == 3:
        initial_type = [('float_input', FloatTensorType([None, 11]))]
        model_onnx = convert_sklearn('''INSERT MODEL HERE''', initial_types=initial_type)
        with open("model.onnx", "wb") as f:
            f.write(model_onnx.SerializeToString())
    # Separate filename and location of input ONNX model
    file_name = arguments.file.split(separator)[-1]
    file_path = separator.join(arguments.file.split(separator)[:-1])
    image_name = file_name + image_name
    # Create MLmodel file for the input
    # TODO: If-condition: check if the MLmodel file already exists, then do not create it again
    MLmodel_file = open(file_path + separator + "MLmodel", "w")
    MLmodel_file.write(get_file_contents(file_name))
    MLmodel_file.close()

    # Add model to mlflow and get its id
    with mlflow.start_run() as run:
        #TODO: Check if file is onnx or not
        onnx_model = mlflow.onnx.load_model('file:'+file_path)
        mlflow.onnx.log_model(onnx_model, file_name)
    run_id = run.info.run_id
    
    print("MLFlow Run ID: " + run_id)
    
    # Convert model into docker and save as tar file
    print("Converting ONNX model (" + file_name + ") to Docker Image (" + image_name + ")")
    os.system('cmd /c mlflow models build-docker -m "runs:/' + run_id + '/' + file_name + '" -n "' + image_name + '"')
    print("Saving Docker Image with input ONNX model: " + file_path)
    os.system('cmd /c docker save -o ' + file_path + separator + image_name + '.tar ' + image_name + '')

    print("MLFlow Run ID: " + run_id)
    print("Done!")
