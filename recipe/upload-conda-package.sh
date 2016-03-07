#!/bin/bash

#
# Upload conda package(s) to http://www.astro.washington.edu/users/mjuric/conda
# channel.
#

if [[ -z $1 ]]; then
	echo "usage: $0 <package_tarball> [<package_tarball> [...]]"
	exit -1
fi

set -e

HOST="mjuric@gateway3.phys.washington.edu"
CONDA="/astro/apps6/opt/anaconda2.4/bin/conda"
DIR_BASE="public_html/html/conda"

# Decide where to upload
if [[ $(uname) == Linux ]]; then
	DIR="$DIR_BASE/linux-64/"
elif [[ $(uname) == Darwin ]]; then
	DIR="$DIR_BASE/osx-64/"
else
	echo "unknown OS $(uname). aborting."
fi

# Upload the packages
scp "$@" "$HOST:$DIR"

# Reindex the channel
ssh "$HOST" "$CONDA" index "$DIR"
