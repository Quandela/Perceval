#!/usr/bin/env bash

set -e # exit on first error

run_notebook () {
    echo converting $notebook
    jupyter nbconvert --clear-output --inplace "$notebook"
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
    jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace "$notebook"
}

# apply patch to notebooks which allows execution
SCRIPT_DIR=$(dirname "$(realpath $0)")
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
(cd $GIT_ROOT_DIR && git apply "$SCRIPT_DIR/notebooks.patch")

nb_dir="docs/source/notebooks"
for entry in `ls $nb_dir | grep \.ipynb`; do
    notebook=$nb_dir/$entry
    if [[ $notebook =~ ".ipynb" ]]
    then
        if [ "$ENABLE_CLOUD_NOTEBOOKS" = "True" ]
        then
            run_notebook
        elif [ "$notebook" = "docs/source/notebooks/Gedik_qudit.ipynb" ]
        then
            echo "  Skip $notebook"
        elif [ "$notebook" = "docs/source/notebooks/Remote_Computation_Tutorial.ipynb" ]
        then
            echo "  Skip $notebook"
        else
            run_notebook
        fi
    fi
done
