sudo yum install wget

wget https://repo.continuum.io/archive/Anaconda3-2020.02-Linux-x86_64.sh

bash Anaconda3-2020.02-Linux-x86_64.sh

source ~/.bashrc

conda create --name py37 python=3.7 -y
conda activate py37

pip install lightgbm
pip install xgboost
pip install scikit.learn==0.22.0
