
// OBJ APP

/**
    The main objective function used by the CANDLE/Supervisor
    model exploration (optimization) loop.
    params : The JSON string of params to be passed to the Benchmark
    run_id : A string run ID that will be the output directory name
*/
(string obj_result) obj_container(string params,
                        string expid,
                        string runid,
                        string model_name,
                        string candle_image)
{
  string model_sh       = getenv("MODEL_SH");
  string turbine_output = getenv("TURBINE_OUTPUT");
  // string model_name     = getenv("MODEL_NAME");

  string outdir;

  outdir = "%s/%s" % (turbine_output, runid);
  // outdir = "%s/%s/Output/%s/%s" % (turbine_output, model_name, expid, runid);

  printf("obj_app: running model shell script in: %s", outdir);

  // We do not use a file type here because this file may not be created,
  // which is handled by get_results()
  string result_file = outdir/"result.txt";
  wait (run_model(model_sh, params, expid, runid, model_name, candle_image))
  {
    obj_result = get_results(result_file);
  }
  printf("obj_app: result(%s): '%s'", runid, obj_result);
}

/**
   Swift/T app function that runs the Benchmark
*/
app (void o) run_model (string model_sh, string params,
                        string expid, string runid, string model_name, string candle_image)
{
  //                  1       2      3     4       5           6
  "bash" model_sh FRAMEWORK params expid runid model_name candle_image;
}

/**
   Extracts the Benchmark output if it exists,
   else, provides a NaN so the workflow can keep running
*/
(string obj_result) get_results(string result_file) {
  if (file_exists(result_file)) {
    file line = input(result_file);
    obj_result = trim(read(line));
  } else {
    printf("File not found: %s", result_file);
    // return with a large value
    obj_result = "1e7";
  }
}
