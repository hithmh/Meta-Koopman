## Setup environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```
conda env create -f environment.yml
conda activate meta-Koopman
```

Mosek requires a license, which can be obtained at https://www.mosek.com/products/academic-licenses/.



## Run the code

Next is an example for training Meta Koopman in the Cartpole example:

```
python3 train_meta.py
```
After training is complete, run 
```
python3 robustness_eval.py
```
Results can be found in \\log\\cartpole\\Koopman\_meta\_deterministic directory.



To try other systems and methods, revise 'env\_name' and 'alg\_name' in variant.py. For DeSKO, use train.py to train the model.
After training is complete, include the name of alg directory, e.g. Koopman\_meta\_deterministic, in the 'eval\_list'. Then run robustness\_eval.py.

