#!/bin/zsh -f
set -eu

# Install GraphDRP in non-container mode

which python
echo
echo "Install GraphDRP dependencies?  Hit enter or Ctrl-C to cancel."
read -t 10 _
echo "Installing..."

PKGS_PIP=(
  h5py    # 3.1
  pyarrow  # 10.0
  "torch==2.0.1"
  torch_geometric
  torch_sparse
  torch_scatter
  torch-cluster
  pubchempy
  rdkit
  networkx

)

set -x
for PKG in $PKGS_PIP
do
  python -m pip install $PKG
done

# candle does not import with scikit-learn 1.2.0
# /opt/conda/bin/pip install scikit-learn==1.1.3
