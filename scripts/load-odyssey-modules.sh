# Source this script to set up your LSD environment
# For odyssey.rc.fas.harvard.edu
# -- mjuric@cfa.harvard.edu

module load hpc/python-2.7.1_no_modules
module load hpc/gtk+-2.24.4
module load hpc/numpy-1.6.0_python-2.7.1
module load hpc/scipy-0.9.0_python-2.7.1
module load hpc/matplotlib-1.0.0_python-2.7.1
module load hpc/ipython-0.10.2_python-2.7.1
module load hpc/python-2.7_modules

module load math/hdf5-1.8.5_gnu
module load hpc/git-1.6.4
module load hpc/gsl-gnu

# Test if we have extra modules in PYTHONPATH
EXTRAMOD=/n/home06/mjuric/lfs/lib/python2.7/site-packages
test x$(echo -n $PYTHONPATH | awk -v RS=: '"'$EXTRAMOD'" == $1' | head -n 1) == x"$EXTRAMOD" || export PYTHONPATH="$EXTRAMOD:$PYTHONPATH"

