my_function () {
    echo converting $notebook
    jupyter nbconvert --clear-output --inplace "$notebook"
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
    jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace "$notebook"
}
cd nb_dir
nb_dir="docs/source/notebooks/"
notebook=$nb_dir
for entry in `ls $nb_dir | grep \.ipynb`; do
    if [[ $entry =~ ".ipynb" ]]
    then
        notebook+=$entry
        if [ "$notebook" = "docs/source/notebooks/Boson Sampling.ipynb" ]
        then
            echo $notebook is ignore
        elif [ "$notebook" = "docs/source/notebooks/BS-based implementation notebook.ipynb" ]
        then
            echo $notebook is ignore
        elif [ "$notebook" = "docs/source/notebooks/Remote computing.ipynb" ]
        then
            echo $notebook is ignore
        else
            my_function
            exit
        fi
        notebook=$nb_dir
    else
        notebook+="${entry} "
    fi
done
