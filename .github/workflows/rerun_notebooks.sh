#!/usr/bin/env bash

set -e # exit on first error

notebooks_directory="docs/source/notebooks"

run_notebook () {
    echo converting $notebook_name
    jupyter nbconvert --clear-output --inplace "$notebook"
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
    jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace "$notebook"
}

SCRIPT_DIR=$(dirname "$(realpath $0)")
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
cd $GIT_ROOT_DIR

# apply patch to notebooks which allows execution
git apply "$SCRIPT_DIR/notebooks.patch"

for notebook in `find "$notebooks_directory" -name "*.ipynb"`; do
    notebook_name="$(basename ${notebook})"
    if [ "$ENABLE_CLOUD_NOTEBOOKS" = "True" ]
    then
        run_notebook
    elif [ "$notebook_name" = "Gedik_qudit.ipynb" ]
    then
        echo "  Skip $notebook_name"
    elif [ "$notebook_name" = "Remote_Computation_Tutorial.ipynb" ]
    then
        echo "  Skip $notebook_name"
    else
        run_notebook
    fi
done

# Revert patch
git apply -R "$SCRIPT_DIR/notebooks.patch"
