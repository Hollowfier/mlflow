import os
from subprocess import Popen, PIPE, STDOUT
import logging

import mlflow
import mlflow.version
from mlflow.utils.file_utils import TempDir, _copy_project
from mlflow.utils.logging_utils import eprint

_logger = logging.getLogger(__name__)

DISABLE_ENV_CREATION = "MLFLOW_DISABLE_ENV_CREATION"
_DOCKERFILE_TEMPLATE = """
# Build an image that can serve mlflow models.
FROM ubuntu:18.04

{install_os_dependencies}

# Download and setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"

ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
# Set up the program in the image
WORKDIR /opt/mlflow

{install_mlflow}

{custom_setup_steps}
{entrypoint}
"""
JAVA_DEP = """
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         build-essential \
         cmake \
         openjdk-8-jdk \
         git-core \
         maven \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
"""
PYTHON_DEP = """
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         git-core \
         build-essential \
         cmake \
    && rm -rf /var/lib/apt/lists/*
"""


def _get_mlflow_install_step(dockerfile_context_dir, mlflow_home, python_only):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    if mlflow_home:
        mlflow_dir = _copy_project(src_path=mlflow_home, dst_path=dockerfile_context_dir)
        install_string = (
                "COPY {mlflow_dir} /opt/mlflow\n"
                "RUN pip install /opt/mlflow\n"
                )
        if not python_only:
            install_string += (
                "RUN cd /opt/mlflow/mlflow/java/scoring && "
                "mvn --batch-mode package -DskipTests && "
                "mkdir -p /opt/java/jars && "
                "mv /opt/mlflow/mlflow/java/scoring/target/"
                "mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n"
                )
        return install_string.format(mlflow_dir=mlflow_dir)
    else:
        install_string = "RUN pip install mlflow=={version}\n"

        if not python_only:
            install_string += (
                "RUN pip install mlflow=={version}\n"
                "RUN mvn "
                " --batch-mode dependency:copy"
                " -Dartifact=org.mlflow:mlflow-scoring:{version}:pom"
                " -DoutputDirectory=/opt/java\n"
                "RUN mvn "
                " --batch-mode dependency:copy"
                " -Dartifact=org.mlflow:mlflow-scoring:{version}:jar"
                " -DoutputDirectory=/opt/java/jars\n"
                "RUN cp /opt/java/mlflow-scoring-{version}.pom /opt/java/pom.xml\n"
                "RUN cd /opt/java && mvn "
                "--batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars\n"
            )
        return install_string.format(version=mlflow.version.VERSION)


def _build_image(image_name, entrypoint, mlflow_home=None, custom_setup_steps_hook=None,
                 python_only=False):
    """
    Build an MLflow Docker image that can be used to serve a
    The image is built locally and it requires Docker to run.

    :param image_name: Docker image name.
    :param entry_point: String containing ENTRYPOINT directive for docker image
    :param mlflow_home: (Optional) Path to a local copy of the MLflow GitHub repository.
                        If specified, the image will install MLflow from this directory.
                        If None, it will install MLflow from pip.
    :param custom_setup_steps_hook: (Optional) Single-argument function that takes the string path
           of a dockerfile context directory and returns a string containing Dockerfile commands to
           run during the image build step.
    :param python_only: To build docker image for python flavor only.
    """
    mlflow_home = os.path.abspath(mlflow_home) if mlflow_home else None
    with TempDir() as tmp:
        cwd = tmp.path()
        install_mlflow = _get_mlflow_install_step(cwd, mlflow_home, python_only)
        custom_setup_steps = custom_setup_steps_hook(cwd) \
            if custom_setup_steps_hook else ""
        os_deps = PYTHON_DEP if python_only else JAVA_DEP
        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(
                    install_os_dependencies=os_deps,
                    install_mlflow=install_mlflow,
                    custom_setup_steps=custom_setup_steps,
                    entrypoint=entrypoint,
                )
            )
        _logger.info("Building docker image with name %s", image_name)
        os.system("find {cwd}/".format(cwd=cwd))
        proc = Popen(
            [
                "docker",
                "build",
                "-t",
                image_name,
                "-f",
                "Dockerfile",
                # Enforcing the AMD64 architecture build for Apple M1 users
                "--platform",
                "linux/amd64",
                ".",
            ],
            cwd=cwd,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True,
        )
        for x in iter(proc.stdout.readline, ""):
            eprint(x, end="")

        if proc.wait():
            raise RuntimeError("Docker build failed.")
