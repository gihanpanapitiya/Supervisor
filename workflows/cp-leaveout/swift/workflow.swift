
/*
  WORKFLOW SWIFT

  Simply run with: 'swift-t workflow.swift <FILE SETTINGS>'
  Or specify the N, S values:
  'swift-t workflow.swift -N=6 -S=6 <FILE SETTINGS>'
  for 55,986 tasks.
  Flags:
  -N : Number of nodes per stage (see default in code)
  -S : Number of stages          (see default in code)
  -r : Use RunType.RESTART, default is RunType.RUN_ALL
       RUN_ALL means this is a fresh run with no prior results

  FILE SETTINGS:
  These are mandatory:
  --plan_json=<FILE>     : The JSON plan for topN_to_uno
  --dataframe_csv=<FILE> : The CSV data file for topN_to_uno
  --db_file=<FILE>       : The SQLite DB file
  --benchmark_data=<DIR> : Used by data_setup to set softlinks to
                           Uno cache and uno_auc_model.txt

  NOTE: This workflow has some complex Python Exception handling
        code that will be pushed into Swift/T for conciseness...
  NOTE: On Summit, you have to use sys.stdout.flush() after
        Python output on stdout
*/

import assert;
import io;
import files;
import math;
import python;
import string;
import sys;

import candle_utils;

report_env();

// BEGIN WORKFLOW ARGUMENTS
// Data split factor with default
int N;
N_s = argv("N", "2");
if (strlen(N_s) > 0)
{
  N = string2int(N_s);
}
else
{
  N = 0;
}
// Maximum stage number with default
// (tested up to S=7, 21,844 dummy tasks)
int S;
S_s = argv("S", "2");
assert(strlen(S_s) > 0, "Set argument S with -S=<S>") =>
  S = string2int(S_s);
string runtype;
if (argv_contains("r"))
{
  runtype = "plangen.RunType.RESTART";
}
else
{
  runtype = "plangen.RunType.RUN_ALL";
}
E_s = argv("E", "20");
assert(strlen(E_s) > 0, "workflow.swift: you must provide an argument to -E");
int max_epochs = string2int(E_s); // epochs=20 is just under 2h on Summit.
string plan_json      = argv("plan_json");
string dataframe_csv  = argv("dataframe_csv");
string db_file        = argv("db_file");
string benchmark_data = argv("benchmark_data");
int    epoch_mode       = string2int(argv("epoch_mode", "1"));
int    benchmark_timeout = string2int(argv("benchmark_timeout", "-1"));
string model_name     = getenv("MODEL_NAME");
string exp_id         = getenv("EXPID");
string turbine_output = getenv("TURBINE_OUTPUT");
// END WORKFLOW ARGUMENTS

// For compatibility with obj():
global const string FRAMEWORK = "keras";

/** RUN STAGE: A recursive function that manages the stage dependencies */
(void v)
run_stage(int N, int S, string this, int stage, void block,
          string plan_id, string db_file, string runtype)
{

  printf("stage: %i this: %s", stage, this);
  // Run the model
  void parent = run_single(this, stage, block, plan_id);

  if (stage < S)
  {
    // Recurse to the child stages
    foreach id_child in [1:N]
    {
      run_stage(N, S, this+"."+id_child, stage+1, parent,
                plan_id, db_file, runtype);
    }
  }
  v = propagate();
}

/** RUN SINGLE: Set up and run a single model via obj(), plus the SQL ops */
(void v) run_single(string node, int stage, void block, string plan_id)
{
  if (stage == 0)
  {
    v = propagate();
  }
  else
  {
    json_fragment = make_json_fragment(node, stage);
    json = "{\"node\": \"%s\", %s}" % (node, json_fragment);
    block =>
      printf("run_single(): running obj(%s)", node) =>
      // Insert the model run into the DB
      result1 = plangen_start(node, plan_id);
    assert(result1 != "EXCEPTION", "Exception in plangen_start()!");
    if (result1 == "0")
    {
      // Run the model
      obj_result = obj(json, node)
        // Update the DB to complete the model run
        => result2 = plangen_stop(node, plan_id);
      printf("run_single(): completed: node: '%s' result: '%s'",
             node, obj_result);
      assert(obj_result != "EXCEPTION" && obj_result != "",
             "Exception in obj()!");
      assert(result2 != "EXCEPTION", "Exception in plangen_stop()!");
      printf("run_single(): stop_subplan result: '%s'", result2);
      v = propagate(obj_result);
    }
    else
    {
      printf("run_single(): plan node already marked complete: " +
             "%s result=%s", node, result1) =>
        v = propagate();
    }
  }
}

// This DB configuration and python_db() function will put all
// calls to python_db() on rank DB corresponding to
// environment variable TURBINE_DB_WORKERS:

pragma worktypedef DB;

@dispatch=DB
(string output) python_db(string code, string expr="repr(0)")
"turbine" "0.1.0"
 [ "set <<output>> [ turbine::python 1 1 <<code>> <<expr>> ]" ];

// Simply use python_db() to log the DB rank:
python_db(
----
import os, sys
print("This rank is the DB rank: %s" % os.getenv("ADLB_RANK_SELF"))
sys.stdout.flush()
----
);

(string result) plangen_start(string node, string plan_id)
{
  result = python_db(
----
import fcntl, sys, traceback
import plangen
try:
    result = str(plangen.start_subplan('%s', '%s', %s, '%s', %s))
except Exception as e:
    info = sys.exc_info()
    s = traceback.format_tb(info[2])
    print(str(e) + ' ... \\n' + ''.join(s))
    sys.stdout.flush()
    result = "EXCEPTION"
----  % (db_file, plan_json, plan_id, node, runtype),
    "result");
}

(string result) plangen_stop(string node, string plan_id)
{
  result = python_db(
----
import plangen
import fcntl, sys, traceback
try:
    result = str(plangen.stop_subplan('%s', '%s', '%s', {}))
except Exception as e:
    info = sys.exc_info()
    s = traceback.format_tb(info[2])
    sys.stdout.write(str(e) + ' ... \\n' + ''.join(s) + '\\n')
    sys.stdout.flush()
    result = 'EXCEPTION'
---- % (db_file, plan_id, node),
      "result");
}

/** MAKE JSON FRAGMENT: Construct the JSON parameter fragment for the model */
(string result) make_json_fragment(string this, int stage)
{
  int epochs = compute_epochs(stage);
  json_fragment = ----
"pre_module":     "data_setup",
"post_module":    "data_setup",
"plan":           "%s",
"config_file":    "uno_auc_model.txt",
"cache":          "cache/top6_auc",
"dataframe_from": "%s",
"save_weights":   "model.h5",
"gpus": "0",
"epochs": %i,
"es": "True",
"use_exported_data": "topN.uno.h5",
"benchmark_data": "%s"
---- %
(plan_json, dataframe_csv, epochs, benchmark_data);
  if (stage > 1)
  {
    n = strlen(this);
    parent = substring(this, 0, n-2);
    result = json_fragment + ----
,
"initial_weights": "../%s/model.h5"
---- % parent;
  }
  else
  {
    result = json_fragment;
  }
}

printf("CP LEAVEOUT WORKFLOW: START: N=%i S=%i", N, S);

// First: simple test that we can import plangen
check = python_persist(----
try:
    import plangen
    result = 'OK'
except Exception as e:
    result = str(e)
----,
"result");
printf("python result: import plangen: '%s'", check) =>
  assert(check == "OK", "could not import plangen, check PYTHONPATH!");

// Initialize the DB
plan_id = python_persist(
----
import sys, traceback
import plangen
try:
    result = str(plangen.plan_prep('%s', '%s', %s))
except Exception as e:
    info = sys.exc_info()
    s = traceback.format_tb(info[2])
    print(str(e) + ' ... \\n' + ''.join(s))
    sys.stdout.flush()
    result = 'EXCEPTION'
---- % (db_file, plan_json, runtype),
"result");
printf("DB plan_id: %s", plan_id);
assert(plan_id != "EXCEPTION", "Plan prep failed!");

// If the plan already exists and we are not doing a restart, abort
assert(plan_id != "-1", "Plan already exists!");

// Kickoff the workflow
stage = 0;
run_stage(N, S, "1", stage, propagate(), plan_id, db_file, runtype);
// printf("CP LEAVEOUT WORKFLOW: RESULTS: COMPLETE");
