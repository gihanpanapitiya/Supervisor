import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import numpy as np
import re
import improve_utils

from tqdm import tqdm


def remove_nonfloat_features(fe):    
    features=[]
    for i in fe.columns:
        try:
            fe[i].astype(float)
            features.append(i)
        except:
            pass
    return features


def get_average_domain_error(cdir, exp, model, runs, conditions, fe):
    
    # cdir = './'
    # exp='EXP000'
    # model = 'GraphDRP'
    # i = gdrp_runs[1]
    de_list=[]
    for i in tqdm(runs):    
        df = pd.read_csv(f'{cdir}/{model}/Output/{exp}/{i}/test_predictions.csv')
        domain_error, _ = error_by_feature_domains(fe, df, conditions)

        de_list.append(domain_error)
    
    

    df_error = pd.DataFrame(np.column_stack([i['error'] for i in de_list]))
    
    merror = df_error.mean(axis=1)
    sdeverror = df_error.std(axis=1)

    domain_error_final = domain_error.loc[:, ['prop','low','high'] ]
    domain_error_final['mean_error'] = merror
    domain_error_final['sdev_error'] = sdeverror
    
    return domain_error_final




# def get_grouped_and_averaged_results(cdir, model, exp, runs):
def get_grouped_and_averaged_results(out_dir, runs, sub_dir=None):
    
    de_list=[]
    for i in tqdm(runs):    
        if sub_dir:
        # df = pd.read_csv(f'{cdir}/{model}/Output/{exp}/{i}/test_predictions.csv')
            df = pd.read_csv(f'{out_dir}/{i}/{sub_dir}/test_predictions.csv')
        else:
            df = pd.read_csv(f'{out_dir}/{i}/test_predictions.csv')
        # domain_error, _ = error_by_feature_domains(fe, df, conditions)

        de_list.append(df)



    # df_error = pd.DataFrame(np.column_stack([i['error'] for i in de_list]))

    de_list = pd.concat(de_list, axis=0)
    
    if ('cell_line_id' not in de_list.columns) and ('improve_sample_id' in de_list.columns):
        de_list["cell_line_id" ] = de_list["improve_sample_id"]

    grps = de_list.groupby(['smiles','cell_line_id'])

    res=[]
    for k in grps.groups.keys():
        tmp = grps.get_group(k)
        tmp_error = abs(tmp['true' ] - tmp['pred']).values

        emean = tmp_error.mean()
        esdev = tmp_error.std()


        res.append( tmp[['smiles','cell_line_id']].values[0].tolist()+ [emean, esdev])


    res = pd.DataFrame(res, columns=['smiles', 'cell_line_id', 'err', 'esdev'])
    return res


def error_by_feature_domains(fp, preds, conditions):

    if isinstance(fp, pd.DataFrame ):
        fps = fp
    else:
        fps = pd.read_csv(fp)

    report = []
    if 'err' not in preds.columns:
        preds['err'] = abs(preds['true'] - preds['pred'])
        
    keep = preds.copy()
    for i in range(conditions.shape[0]):

        prop = conditions.loc[i, 'prop']
        low = conditions.loc[i, 'low']
        high = conditions.loc[i, 'high']

        locs = np.logical_and(fps[prop] <= high, fps[prop] > low)
        smiles = fps.loc[locs, 'smiles'].values
        tmp = preds[preds.smiles.isin(smiles)]
        mean_err = tmp.err.mean()
        sdev_err = tmp.err.std()

        report.append([prop, low, high, mean_err, sdev_err])

        keep = keep[keep.smiles.isin(smiles)] # this is in case we want to progressively
                                            # consider domains. A domain composed of multiple domains

    final_domain_err = keep.err.mean()  # return this

    
    report = pd.DataFrame(report, columns=['prop', 'low', 'high', 'mean_error', 'sdev_error'])
    return report, final_domain_err


def get_conditions(fe, features, nbins=10):

    conditions=[]
    if len(features)==0:
         features = ['nAtom', 'nAromAtom', 'nRing', 'BertzCT', 'nBondsS', 'nBondsO', 'nHBDon', 'nHBAcc', 'FilterItLogS', 'SLogP']
    for prop  in features:

        # a,b,c = plt.hist(fe[prop], bins=10);
        a, bin_edges = np.histogram(fe[prop].values,  nbins)
        c = [[prop, bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
        conditions.extend(c)
    conditions = pd.DataFrame(conditions, columns=['prop', 'low', 'high'])

    return conditions




def get_mean_mE(df_error):
    
    """
    for each property, find the errors for each domain.
    then find the average of these errors. the properties 
    corresponding to maximum error can be selected for further
    study.
    """
    
    res=[]
    for p in df_error.prop.unique():

        tmp=df_error[df_error.prop == p]
        # print(p, np.nanmean(tmp.mean_error))
        res.append([p, np.nanmean(tmp.mean_error) ])

    res = pd.DataFrame(res, columns=['prop', 'mean_mE'])
    res= res.sort_values(by='mean_mE', ascending=False)
    return res





def cat_domain_error_for_run(preds, fe, features):
    if 'err' not in preds.columns:
        preds['err'] = abs(preds['true'] - preds['pred'])
    report=[]
    # prop = cat[0]
    for prop in features:

        uniques= sorted(fe[prop].unique())


        for u in uniques:

            tmp=fe[fe[prop] == u]

            pred_tmp = preds[preds.smiles.isin(tmp.smiles)]

            mean_err = pred_tmp.err.mean()
            sdev_err = pred_tmp.err.std()

            # if prop == 'nAcid':
            #     print(u, mean_err, tmp.shape)
            report.append([prop, u, mean_err, sdev_err])


    report = pd.DataFrame(report, columns=['prop', 'pvalue', 'mean_error', 'sdev_error'])
    return report



def cat_error_for_model(cdir, model, exp, runs, fe, features):
    
    # model='SWnet'
    # i = swn_runs[0]
    errors=[]
    for run in tqdm(runs):
        preds = pd.read_csv(f'{cdir}/{model}/Output/{exp}/{run}/test_predictions.csv')
        report = cat_domain_error_for_run(preds, fe, features)
        errors.append(report['error'])
    
    tmp = pd.DataFrame(np.column_stack([errors]))
#     error_mean = tmp.mean(axis=1)
#     error_sdev = tmp.std(axis=1)
    error_mean = np.nanmean(tmp.values, axis=0)
    error_sdev = np.nanstd(tmp.values, axis=0)
    
    tmp2 = report[['prop', 'pvalue']]
    tmp2['mean_error'] = error_mean
    tmp2['sdev_error'] = error_sdev
    
    
    return tmp2

# # def get_res(nrlow, nrhigh, runs, model, exp, prop):
# def get_res(runs, cod, model, exp, prop, calculate_derror=False):
    
#     res=[]
#     # for i in range(nrlow,nrhigh):
#     for i in runs:
        
#         if calculate_derror:
#             df = pd.read_csv(f'{cod}/{model}/Output/{exp}/{i}/test_predictions.csv')
#             domain_error, _ = error_by_feature_domains(fe, df, conditions)
#         else:
#             domain_error = pd.read_csv(f'{cod}/{model}/Output/{exp}/{i}/domain_err.csv')

#         tmp = domain_error[domain_error.prop== prop]
#         res.append(tmp['error'].values.tolist())
#         # os.system(f"rm tmpcmp/DrugCell/Output/EXP002/{i}/*.hidden")
#         # os.system(f"rm tmpcmp/DrugCell/Output/EXP002/{i}/model_[0-9].pt")
#         # os.system(f"rm tmpcmp/DrugCell/Output/EXP002/{i}/model_[0-9][0-9].pt")
#     res=np.array(res)
#     res.shape
#     err_mean = np.nanmean(res, axis=0)
#     err_sdev = np.nanstd(res, axis=0)

#     return err_mean, err_sdev, tmp, res



def find_cont_and_cat_features(fe_tmp):
    
    cat, cont=[],[]
    for i in fe_tmp.columns:
        if fe_tmp[i].nunique()<20 and all([float(a).is_integer() for a in fe_tmp[i].unique()]):

            cat.append(i)
        else:
            cont.append(i)
            
    return cont, cat


