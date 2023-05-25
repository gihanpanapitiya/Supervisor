#!/bin/bash
set -eu

# MODEL.SH

# Supervisor shell wrapper around CANDLE model
# Used for CANDLE_MODEL_IMPL types: "app" and "container"

# Note that APP_PYTHONPATH is used by models here and
# not just PYTHONPATH

# Note: Under Swift/T, the initial output from here will go
# to the main Swift/T stdout and be mixed with output from
# other models.
# Thus, we redirect to a separate model.log file for each model run
# and normally we do not produce output until after the redirection.

usage()
{
  echo "Usage: model.sh FRAMEWORK PARAMS EXPID RUNID MODEL_TYPE MODEL_NAME MODEL_ACTION"
  echo "MODEL_TYPE is BENCHMARK or SINGULARITY"
  echo "MODEL_NAME is the CANDLE Benchmark name (e.g., 'uno')"
  echo "           or a /path/to/image.sif"
  echo "MODEL_ACTION is unused for a Benchmark,"
  echo "             for Singularity it is a script (e.g., 'ACTION.sh')"
  echo "The environment should have:"
  echo "                EMEWS_PROJECT_ROOT|WORKFLOWS_ROOT TURBINE_OUTPUT"
  echo "                SITE OBJ_RETURN BENCHMARK_TIMEOUT"
  echo "                CANDLE_DATA_DIR"
  echo "If SH_TIMEOUT is set, we run under the shell command timeout"
}

if (( ${#} != 7 ))
then
  echo
  echo "model.sh: Wrong number of arguments: received ${#} , required: 7"
  echo
  usage
  exit 1
fi

FRAMEWORK=$1 # Usually "keras" or "pytorch"
# JSON string of parameters:
PARAMS="$2"
export EXPID=$3
export RUNID=$4
export MODEL_TYPE=$5
export MODEL_NAME=$6
export MODEL_ACTION=$7

# Each model run, runs in its own "instance" directory
# Set instance_directory to that and cd into it.
# # TODO: rename INSTANCE_DIRECTORY to OUTPUT_DIR
#set -x
if [[ $MODEL_TYPE = "SINGULARITY" ]]
then
  # TODO: Rename "instance" to "run"
  MODEL_TOKEN=$( basename $MODEL_NAME .sif )
  INSTANCE_DIRECTORY=$CANDLE_DATA_DIR/$MODEL_TOKEN/Output/$EXPID/$RUNID
  INTERNAL_DIRECTORY=$MODEL_NAME/Output/$EXPID/$RUNID
else # "BENCHMARKS"
  INSTANCE_DIRECTORY=$TURBINE_OUTPUT/$RUNID
  export CANDLE_OUTPUT_DIR=$INSTANCE_DIRECTORY
fi

# All stdout/stderr after this point goes into model.log !
mkdir -pv $INSTANCE_DIRECTORY
LOG_FILE=$INSTANCE_DIRECTORY/model.log
echo "redirecting to: LOG_FILE=$INSTANCE_DIRECTORY/model.log"
set +x
exec >> $LOG_FILE
exec 2>&1
cd $INSTANCE_DIRECTORY

TIMEOUT_CMD=""
if [[ ${SH_TIMEOUT:-} != "" ]] && [[ $SH_TIMEOUT != "-1" ]]
then
  TIMEOUT_CMD="timeout $SH_TIMEOUT"
fi

log()
{
  echo $( date "+%Y-%m-%d %H:%M:%S" ) "MODEL.SH:" $*
}

log "START"
log "MODEL_NAME: $MODEL_NAME"
log "RUNID: $RUNID"
log "HOST: $( hostname )"
log "ADLB_RANK_OFFSET: $ADLB_RANK_OFFSET"
log "MODEL_TYPE: $MODEL_TYPE"

# Source langs-app-{SITE} from workflow/common/sh/ (cf. utils.sh)
if [[ ${WORKFLOWS_ROOT:-} == "" ]]
then
  WORKFLOWS_ROOT=$( cd $EMEWS_PROJECT_ROOT/.. ; /bin/pwd )
fi
source $WORKFLOWS_ROOT/common/sh/utils.sh
source_site langs-app $SITE

echo
log "PARAMS:"
echo $PARAMS | print_json

echo
log "USING PYTHON:" $( which python3 )
echo

# Cf. utils.sh
log_path APP_PYTHONPATH
log_path PYTHONPATH
log_path LD_LIBRARY_PATH
show     PYTHONHOME

# Set up PYTHONPATH for app tasks
export PYTHONPATH=${APP_PYTHONPATH:-}:${PYTHONPATH:-}

# Construct the desired model command MODEL_CMD based on MODEL_TYPE:
if [[ ${MODEL_TYPE:-} == "SINGULARITY" ]]
then

  # No model_runner, need to write parameters.txt explicitly:
  #  get hyper_parameter_map to pass as 2nd argument

  FLAGS=$( python3 $WORKFLOWS_ROOT/common/python/runner_utils.py expand_params \
                   "$PARAMS" )

  # Remove --candle image flag and the second argument, assume it is the last argument
  export FLAGS="${FLAGS/ --candle_image*/}"

  # The Singularity command line arguments:
  MODEL_CMD=( singularity exec --nv
              --bind $CANDLE_DATA_DIR:/candle_data_dir
              $MODEL_NAME ${MODEL_ACTION}.sh $ADLB_RANK_OFFSET
              /candle_data_dir
              $FLAGS # $INTERNAL_DIRECTORY/parameters.txt
              --experiment_id $EXPID
              --run_id $RUNID
            )

else # "BENCHMARKS"

  # The Python command line arguments:
  PY_CMD=( "$WORKFLOWS_ROOT/common/python/model_runner.py"
           "$PARAMS"
           "$INSTANCE_DIRECTORY"
           "$FRAMEWORK"
           "$RUNID"
           "$BENCHMARK_TIMEOUT" )

  MODEL_CMD=( python3 -u "${PY_CMD[@]}" )
  # model_runner/runner_utils writes result.txt
fi

log "MODEL_CMD: ${MODEL_CMD[@]}"

# Run Python!
$TIMEOUT_CMD "${MODEL_CMD[@]}" &
PID=$!

if [[ ${MODEL_TYPE:-} == "SINGULARITY" ]]
then
  wait $PID
  ls -ltrh
  sleep 1  # Wait for initial output
  # Get last results of the format "CANDLE_RESULT xxx" in model.log
  # NOTE: Enabling set -x will break the following (token CANDLE_RESULT)
  RES=$( awk -v FS="IMPROVE_RESULT" 'NF>1 {x=$2} END {print x}' \
             $INSTANCE_DIRECTORY/model.log )
  RESULT="$(echo $RES | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')" || true
  echo "CANDLE RESULT: '$RESULT'"
  echo $RESULT > $INSTANCE_DIRECTORY/result.txt
else
  wait $PID
  CODE=$?
  if (( CODE ))
  then
    echo # spacer
    if (( $CODE == 124 ))
    then
      log "TIMEOUT ERROR! (timeout=$SH_TIMEOUT)"
      # This will trigger a NaN (the result file does not exist)
      exit 0
    else
      log "MODEL ERROR! (CODE=$CODE)"
      if (( ${IGNORE_ERRORS:-0} ))
      then
        log "IGNORING ERROR."
        # This will trigger a NaN (the result file does not exist)
        exit 0
      fi
      log "ABORTING WORKFLOW (exit 1)"
      exit 1 # Unknown error in Python: abort the workflow
    fi
  fi

  # Get results from model.log: last occurrence of "loss: xxx"
  RESULT=$(awk -v FS="loss:" 'NF>1{print $2}' model.log | tail -1)
  log "RESULT: $RESULT"
  echo $RESULT > $INSTANCE_DIRECTORY/result.txt
fi

log "END: SUCCESS"

exit 0 # Success

# Local Variables:
# sh-basic-offset: 2
# End:
