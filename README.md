### Compile slides

```console
APPTAINERENV_HDF5_USE_FILE_LOCKING=FALSE APPTAINERENV_CUDA_VISIBLE_DEVICES=0 apptainer run \
    --bind .:/opt/demo --bind /cvmfs:/cvmfs --nv \
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
