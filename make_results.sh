#!/bin/bash

if [ -z "$PYTHONPATH" ] ; then
  export PYTHONPATH=$PWD
else
  export PYTHONPATH=$PWD:$PYTHONPATH
fi

mkdir -p results
cd results
for fic in ../validation/*.py ; do
  echo run $fic
  ${PYTHON:-python} $fic
done
