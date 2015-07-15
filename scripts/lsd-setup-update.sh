#!/bin/bash
# 
# Builds a .tar.bz2 source package, and makes it the default in
# lsd-setup's Makefile. Assumes (but checks) that lsd-setup has
# been checked out in ../lsd-setup.
#

test -f setup.py || { echo "This script must be run from LSD's base directory (the one containing setup.py)"; exit -1; }
test -d ../lsd-setup/var/pkg/repo || { echo "lsd-setup, with source repository, must be checked out in ../lsd-setup"; exit -1; }
test ! -d var/pkg || { echo "You must run this package from LSD, not lsd-setup's base directory"; exit -1; }

PYTHONVER=`python --version 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2`
test x"$PYTHONVER" == x"2.7" || { echo "Python 2.7 required. Your Python is reporting $PYTHONVER."; exit -1; }

rm -rf dist && \
python setup.py sdist --format bztar && \
cp dist/* ../lsd-setup/var/pkg/repo || exit -1

LSDPKG=$(basename $(ls dist/*.bz2) .tar.bz2)
sed "s/LSDPKG=.*/LSDPKG=$LSDPKG/" ../lsd-setup/var/pkg/scripts/Makefile > ../lsd-setup/var/pkg/scripts/Makefile.tmp && \
	mv ../lsd-setup/var/pkg/scripts/Makefile.tmp ../lsd-setup/var/pkg/scripts/Makefile

echo "lsd-setup Makefile updated to LSDPKG=$LSDPKG:"
echo "======================================================="
(cd ../lsd-setup && git diff)
echo "======================================================="
echo "Press ENTER to continue..."
read
(cd ../lsd-setup && git commit -a -e -m "Adding $LSDPKG")
