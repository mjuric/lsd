# lsd -- Large Survey Database

## Building

- clone 'master' to a directory (I usually have it in ~/project)
- run:
```
python ./setup.py build_ext --inplace
```
to build the required modules.
- run:
```
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```
to set up the environment.
- after that, you should be able to run all `lsd-*` stuff directly from src/
