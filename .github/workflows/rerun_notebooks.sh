my_function () {
    echo converting $notebook
    jupyter nbconvert --clear-output --inplace "$notebook"
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
    jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace "$notebook"
}
nb_dir="docs/source/notebooks"
for entry in `ls $nb_dir | grep \.ipynb`; do
    notebook=$nb_dir/$entry
    if [[ $notebook =~ ".ipynb" ]]
    then
        if [ "$notebook" = "docs/source/notebooks/BS-based_implementation.ipynb" ]
        then
            echo $notebook is ignore
        elif [ "$notebook" = "docs/source/notebooks/Boson_sampling.ipynb" ]
        then
            echo $notebook is ignore
        elif [ "$notebook" = "docs/source/notebooks/Gedik_qudit.ipynb" ]
        then
            echo $notebook is ignore
        elif [ "$notebook" = "docs/source/notebooks/Remote_computing.ipynb" ]
        then
            echo $notebook is ignore
        else
            my_function
        fi
    fi
done
