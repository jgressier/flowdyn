#!/bin/sh
#
rootdir=$(git rev-parse --show-toplevel)
if [ -z "$rootdir" ] ; then
	echo unable to find git top level
	exit
else
	echo git root dir: $rootdir
	cd $rootdir 
	find lessons -name \*.ipynb -exec python lessons/remove_output.py {} \;
	find . -name \*.ipynb.bak -exec rm {} \;
	for dirname in .ipynb_checkpoints __pycache__ ; do
		find . -name .ipynb_checkpoints -type d -exec rm -R {} \;
	done
fi