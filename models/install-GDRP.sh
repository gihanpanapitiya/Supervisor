#!/bin/bash -norc
set -eu

# Install GraphDRP in non-container mode

THIS=${0:h:A}
source $THIS/../workflows/common/sh/utils.sh

signature SYSTEM -- ${*}

which python
echo
echo -n "Install GraphDRP dependencies for $SYSTEM? "
echo -n "Hit enter or Ctrl-C to cancel."
read -t 10 _
echo
echo "Installing..."

# NEW:

# Requires Python 3.8 for Torch 2.0.0
# Trying with ALCF Conda, Python 3.10

# ALCF: The ALCF-maintained data science installation
#       Make a conda clone
# Anaconda: Plain Anaconda-based install.  Good for Lambda

# "pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch" # ORIG

# ALCF has pytorch, tensorflow, h5py, matplotlib, networkx, psutil, pyarrow
case $SYSTEM in

  ALCF)
    SPECS=(
      "-c bioconda pubchempy"
    )
    ;;

  Anaconda)
    SPECS=(
      "-c pytorch pytorch"
      "-c pyg -c conda-forge pyg" # 2.1.0
      "-c bioconda pubchempy"
      # "-c rdkit rdkit" # hangs
    )
    ;;

  *) echo "No such SYSTEM='$SYSTEM'"
     exit 1
     ;;
esac

timestamp()
{
  # Prevent newline:
  echo -n $( date +"%Y-%m-%d %H:%M:%S" )
}

LIST=$SPECS_$SYSTEM
for SPEC in
do
  timestamp
  echo "CONDA:" $SPEC
  echo
  conda install --yes $=SPEC
  echo
done

timestamp
echo "PIP: deap"
pip install deap
echo

timestamp
echo "DONE."
