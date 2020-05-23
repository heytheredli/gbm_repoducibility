# GBM Reproducibility
Testing GBM reproducibility

## Setup

```
sudo yum install git
git config --global user.name "Daniel Li"
git config --global user.email daniel.li317@gmail.com
cd 
git clone https://github.com/heytheredli/gbm_reproducibility.git
cd gbm_reproducibility
. ec2_setup.sh
```

### Jupyter notebook setup
```
#ec2
jupyter notebook password
nohup jupyter notebook --no-browser --port=8888

#Local machine
ssh -i my-private-key.pem -N -f -L localhost:8888:localhost:8888 user-name@remote-hostname
```

