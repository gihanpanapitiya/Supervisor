
# CFG GraphDRP TEST

# export CANDLE_MODEL_TYPE="SINGULARITY"
# export MODEL_NAME=/software/improve/images/GraphDRP.sif  #Lambda

export MODEL_NAME=graphdrp
export MODEL_PYTHON_DIR=$HOME/proj/GraphDRP
export MODEL_RETURN=val_loss

export PARAM_SET_FILE=graphdrp_param_space-3.json

export CANDLE_FRAMEWORK="pytorch"

# SMALL:
export PROCS=3
export POPULATION_SIZE=2
export NUM_ITERATIONS=2

# # MEDIUM:
# export PROCS=4
# export POPULATION_SIZE=16
# export NUM_ITERATIONS=4

# LARGE:
# export POPULATION_SIZE=80
# export NUM_ITERATIONS=10
