A=(
  --nv
  --bind /homes/woz/CANDLE_DATA_DIR:/candle_data_dir
  /software/improve/images/GraphDRP.sif train.sh 1 /candle_data_dir
  --learning_rate 0.044686897181095996
  --batch_size 64
  --epochs 100
  --train_ml_data_dir /candle_data_dir/HPO/GraphDRP/CTRPv2
  --val_ml_data_dir /candle_data_dir/HPO/GraphDRP/CTRPv2
  --test_ml_data_dir /candle_data_dir/HPO/GraphDRP/CTRPv2
  --experiment_id EXP040
  --run_id run_01_001_0001
)
singularity exec $A
