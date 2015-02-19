# lsd -- Large Survey Database

## building

- clone 'master' to a directory (I usually have it in ~/project)
- run:
```
python ./setup.py build_ext --inplace
```
to build the required modules.
- after that, you should be able to run all `lsd-*` stuff directly from src/;
  there's no need to source the environment setup scripts.
