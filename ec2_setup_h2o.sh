conda deactivate

conda create -n py36 python=3.6 anaconda
conda activate py36
conda config --append channels conda-forge
conda install -c h2oai h2o

sudo yum install jre

