#!/bin/bash
module load python/3.8
virtualenv ~/featstore
source ~/featstore/bin/activate
pip install --no-index --upgrade pip
pip install --no-index pandas scipy scikit_learn matplotlib seaborn jupyterlab imbalanced_learn xgboost
pip install numba==0.53.1 tsfresh pyphm
pip install -e .

# create bash script for opening jupyter notebooks https://stackoverflow.com/a/4879146/9214620
cat << EOF >$VIRTUAL_ENV/bin/notebook.sh
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter-lab --ip \$(hostname -f) --no-browser
EOF

chmod u+x $VIRTUAL_ENV/bin/notebook.sh