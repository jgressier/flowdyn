#!/bin/bash

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

pwd1=$(cd $(dirname $0) ; pwd)
if [ "$pwd1" != "$(pwd)" ] ; then
  echo "Warning:"
  echo "--------"
  echo "  run dir    = $(pwd)"
  echo "  script dir = $pwd1"
  echo
fi

valdir=validation
if [ ! -d $valdir ] ; then
  echo "Error:"
  echo "======"
  echo "  run dir = $(pwd)"
  echo "  directory $(pwd)/$valdir/ does not exist"
  echo
  exit 1
fi

mkdir -p results
cd results

for fic in ../validation/*.py ; do
  echo run $fic
  ${PYTHON:-python} $fic
done
