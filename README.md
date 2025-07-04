## Setup environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.



To create a conda env with python3, one runs
conda env create -f environment.yml
conda activate meta-Koopman



# Run the code

Next is an example for training Meta Koopman in the Cartpole example:
```
Run train\_meta.py

After training is complete, run robustness\_eval.py
```

Results can be found in \\log\\cartpole\\Koopman_meta_deterministic directory.



To try other systems and methods, revise 'env_name' and 'alg_name' in variant.py.
After training is complete, include the name of alg directory, e.g. Koopman_meta_deterministic, in the 'eval_list'. Then run robustness\_eval.py.


