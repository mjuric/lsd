#!/bin/bash
# 
# Builds and installs the checked out version into /n/panlfs/mjuric/lsd
# and links it to 'dev', unless told otherwise
#

test -d /n/panlfs || { echo "This script can only be used on Harvard Odyssey cluster"; exit -1; }
test -f setup.py || { echo "This script must be run from the directory that contains setup.py"; exit -1; }

NAME=$(git describe)
DEST=${1:-dev}

. ./scripts/load-odyssey-modules.sh 

rm -rf build
python setup.py install --home=/n/panlfs/mjuric/lsd/$NAME

rm -f /n/panlfs/mjuric/lsd/$DEST
ln -s $NAME /n/panlfs/mjuric/lsd/$DEST
