# ML4GW focus demo at ADASS 2023
Notebook slideshow demonstrating how using fast, simple tools tailored for a domain can improve the robustness of the ML models that we train.

## Getting setup
All notebooks are intended to be run inside the apptainer container specified in [`apptainer.def`](./apptainer.def). Start by building this:

```
apptainer build demo.sif apptainer.def
```

Then you can launch jupyter lab inside the container
```console
# pick which GPU index you want to use
DEVICE=0
APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE APPTAINERENV_CUDA_VISIBLE_DEVICES=$DEVICE \
    apptainer run --bind .:/opt/demo --bind /cvmfs:/cvmfs --nv \
    demo.sif jupyter lab \
        --ip 0.0.0.0 \
        --port 8858 \
        --no-browser \
        --NotebookApp.token=''
```

### Compile slides
To compile the slides used for presentation, first pull a Triton container locally and launch it on 2 GPUs

```console
DEVICES=0,1  # or whichever GPU indices you want to use
mkdir repo

apptainer pull triton.sif docker://nvcr.io/nvidia/tritonserver:23.07-py3
APPTAINERENV_CUDA_VISIBLE_DEVICES=$DEVICES apptainer run --nv triton.sif \
    /opt/tritonserver/bin/tritonserver \
        --model-repository repo \
        --model-control-mode poll \
        --repository-poll-secs 10
```

Then launch the slide server on a specified port (can't use default of 8000 since Triton uses that)
```console
# whichever device you want to run on, ideally
# different from the ones Triton is running on
DEVICE=2
APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE APPTAINERENV_CUDA_VISIBLE_DEVICES=$DEVICE \
    apptainer run --bind .:/opt/demo --bind /cvmfs:/cvmfs --nv \
    demo.sif jupyter nbconvert \
        --execute \
        --to slides \
        --post serve train.ipynb \
        --CSSHTMLHeaderPreprocessor.style=default \
        --ServePostProcessor.port 8810 \
        --ServePostProcessor.open_in_browser=False \
        --SlidesExporter.reveal_scroll=True  \
        --SlidesExporter.reveal_number 'c/t'
```
