import os
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

conditions = pd.DataFrame([[ 'nAromAtom' , 5, 10 ],
                            ['nAtom', 20, 50], 
                            ['BertzCT', 800, 1000]],
                         columns=['prop', 'low', 'high'])
# from cmp_utils import conditions, Benchmark

CANDLE_DATA_DIR = os.getenv("CANDLE_DATA_DIR")

def compare(exp_id, run_id):
    cmp_results={}
    print(f"compare: run_id={run_id}")
    # gParams = read_params(exp_id, run_id)
    # model = gParams("model_name")

    model = "DrugCell" # TODO: Hardcoded. have to get this from output dir?
    turbine_output = os.getenv("TURBINE_OUTPUT")

    outdir = os.path.join(turbine_output, run_id, 'Output', 'EXP000', 'RUN000') # TODO: Have to fix this
    directory = outdir
    # directory = f"{CANDLE_DATA_DIR}/Output/{exp_id}/{run_id}"
    print("reading the predictions....")
    df_res = pd.read_csv(f"{directory}/test_predictions.csv")

    # a class to calculate errors for subsets of the validation/test set
    print("reading the drug feature file....")
    # TODO: Should have to save the above file in this file
    bmk = Benchmark(fp_path=f'{CANDLE_DATA_DIR}/drug_features.csv') # TODO: have to have a drug features for a common test set
    subset_err, final_domain_err = bmk.error_by_feature_domains_model(df_res, conditions)

    # or this
    # fp_path=f'{CANDLE_DATA_DIR}/drug_features.csv'
    # subset_err, final_domain_err = error_by_feature_domains_model(fp_path, df_res, conditions)

    # collect results for comparison
    cmp_prop = 'nAtom' # TODO: Get this from gParameters 
    subset_err.set_index('prop', inplace=True) # TODO: use 'prop' as a parameter and move it to cmp_models.txt
    cmp_results[run_id] = subset_err.loc[cmp_prop, 'error'] # this is the property based on which we want to do the comparison
    with open(f"{directory}/subset_err.txt", "w") as fp:
        fp.write(str(cmp_results[run_id]))

    return str(cmp_results[run_id])




def error_by_feature_domains_model(fp_path, preds, conditions):
    
   
    fps = pd.read_csv(fp_path)
    report = []
    preds['err'] = abs(preds['true'] - preds['pred'])
    keep = preds.copy()
    for i in range(conditions.shape[0]):

        prop = conditions.loc[i, 'prop']
        low = conditions.loc[i, 'low']
        high = conditions.loc[i, 'high']

        locs = np.logical_and(fps[prop] <= high , fps[prop] > low)
        smiles = fps.loc[locs, 'smiles'].values
        tmp = preds[preds.smiles.isin(smiles)]
        mean_err = tmp.err.mean()

        report.append([prop, low, high, mean_err])

        keep = keep[keep.smiles.isin(smiles)]

    final_domain_err = keep.err.mean() # return this
    report = pd.DataFrame(report, columns=['prop', 'low', 'high', 'error'])
    return report, final_domain_err





class Benchmark:
    
    def __init__(self, fp_path):
        
        self.fps = pd.read_csv(fp_path)
        # self.model_preds = model_preds
        # self.feature_conditions = feature_conditions
        self.reports = {}
        
        
    def error_by_feature_domains_model(self, preds, conditions):
        
        fps = self.fps
        report = []
        preds['err'] = abs(preds['true'] - preds['pred'])
        keep = preds.copy()
        for i in range(conditions.shape[0]):

            prop = conditions.loc[i, 'prop']
            low = conditions.loc[i, 'low']
            high = conditions.loc[i, 'high']

            locs = np.logical_and(fps[prop] <= high , fps[prop] > low)
            smiles = fps.loc[locs, 'smiles'].values
            tmp = preds[preds.smiles.isin(smiles)]
            mean_err = tmp.err.mean()

            report.append([prop, low, high, mean_err])

            keep = keep[keep.smiles.isin(smiles)]

        final_domain_err = keep.err.mean() # return this
        report = pd.DataFrame(report, columns=['prop', 'low', 'high', 'error'])
        return report, final_domain_err






    def error_by_feature_domains(self, feature_conditions):

        results=[]
        for model_name, pred in self.model_preds.items():
            
            report = self.error_by_feature_domains_model(pred, feature_conditions)
            report.loc[:, 'model'] = model_name
            results.append(report)
            
        results = pd.concat(results, axis=0)
        results = results.loc[:, ['model', 'prop', 'low', 'high', 'error']]
        results.reset_index(drop=True, inplace=True)

        return results
    

    def rank_by_acc(self, metric='rmse', th=3):
    
        results={}
        for model_name, pred in self.model_preds.items():
            sub = pred[pred.labels > th]
            rmse = mean_squared_error(y_true=sub['labels'], y_pred=sub['preds'])**.5

            results[model_name] = {'rmse': rmse}

        results = pd.DataFrame.from_dict(results)
        results = results.T
        return results



def create_grid_files():

    dc_grid = {'epochs': [1, 2], 'lr': [1e-2, 1e-3]}
    sw_grid = {'epochs': [3, 4], 'lr': [1e-2, 1e-5]}

    with open('DrugCell_grid.json', 'w') as fp:
        json.dump(dc_grid, fp)

    with open('SWnet_CCLE_grid.json', 'w') as fp:
        json.dump(sw_grid, fp)