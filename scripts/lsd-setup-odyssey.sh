# Source this script to set up your LSD environment
# For odyssey.rc.fas.harvard.edu
# -- mjuric@cfa.harvard.edu

test -d /scratch && export TMP=/scratch

# Load system defaults
source ~mjuric/.lsdrc-default

if [ "x$1" != "x" ]
then
	RC=.lsdrc-$1

	# Load defaults from ~mjuric, if he has them
	if [ -f ~mjuric/$RC ]
	then
		source ~mjuric/$RC
	fi
else
	RC=.lsdrc
fi

# Override with whatever the user has set
if [ -f ~/$RC ]
then
	source ~/$RC
fi

export LSD_BASE
export LSD_VER
export LSD_DB

# ==============================================
. $LSD_ODYSSEY_MODULES_SCRIPT

export LSD_BIN="$LSD_BASE/lsd_$LSD_VER/bin"
export PYLSD="$LSD_BASE/lsd_$LSD_VER/lib/python"

# Test for whether we have LSD in PYTHONPATH
test x$(echo -n $PYTHONPATH | awk -v RS=: '"'$PYLSD'" == $1' | head -n 1) == x"$PYLSD" || export PYTHONPATH="$PYLSD:$PYTHONPATH"

# Test if the correct LSD binaries are in the path
test x"$(which lsd-query 2>&1)" == x"$LSD_BIN/lsd-query" || export PATH="$LSD_BIN:$PATH"

echo
echo "Python 2.7 in       : $(which python)"
echo "LSD binaries in     : $LSD_BIN        (added to path)"
echo "LSD modules in      : $PYLSD (added to PYTHONPATH)"
echo "Default database in : $LSD_DB"
echo
echo "LSD set up complete. Type 'lsd-query' to verify. Have fun."
echo "                   -- Mario Juric <mjuric@cfa.harvard.edu>"
echo

