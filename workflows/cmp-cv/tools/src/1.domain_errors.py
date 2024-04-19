import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import numpy as np
import re
import improve_utils
from tqdm import tqdm
from utils import remove_nonfloat_features
from utils import get_conditions
from utils import get_mean_mE
from utils import get_average_domain_error

from utils import get_grouped_and_averaged_results, error_by_feature_domains
from tqdm import tqdm
from utils import cat_error_for_model

from utils import cat_domain_error_for_run
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import find_cont_and_cat_features
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from feature_utils import get_explainable_features
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""
This script is used to find the domain errors. The results get saved in the save_path.
"""

def concat_sort_error_dfs(cat_df, cont_df):

    '''
    concatenates the categorical and continuous dataframes and
    sorts by the mean error
    '''
    cat_df['prop_type']='cat'

    cont_df['prop_type']='cont'

    df_E = pd.concat([cat_df, cont_df], axis=0)

    df_E.sort_values(by='mean_mE', ascending=False, inplace=True)
    df_E.reset_index(drop=True, inplace=True)

    return df_E

def concat_mean_error_dfs(model_dict, ftype='cont'):
    
    tmps=[]
    to_drop =[]
    for i, (k,v) in enumerate(model_dict.items()):
        if k !='dc' or k!='gdrp':

            if ftype=='cont':
                tmp = v[['prop', 'low','high','mean_error']]
                tmp['pvalue'] = 0.5*(tmp['low'] + tmp['high']) 
            else: 
                tmp = v[['prop', 'pvalue', 'mean_error' ]]
            tmp.rename(columns={'mean_error': f'{k}_mean_error', 'prop':f'{k}_prop'}, inplace=True)
            tmps.append(tmp)
            if i!=0:
                to_drop.extend([ f'{k}_prop'])
            else:
                to_keep={f'{k}_prop':'prop'}
        
    
    tmps = pd.concat(tmps, axis=1)
    tmps = tmps.drop(to_drop, axis=1)
    tmps = tmps.rename(columns=to_keep)

    return tmps
    
def get_sdev_max(df_concat):
    
    df_concat['sdev'] = df_concat.iloc[:, 3:].std(axis=1)
    maxs=[]
    for p in df_concat.prop.unique():
        tmp = df_concat[df_concat.prop == p]
        maxs.append([p,tmp.sdev.max()])
    
    maxs = pd.DataFrame(maxs, columns=['prop', 'sdev_max'])
    maxs.sort_values(by='sdev_max', ascending=False, inplace=True)
    maxs.reset_index(drop=True, inplace=True)
    return maxs
    

def get_max_mE(df_error):
    
    """
    for each property, find the errors for each domain.
    then find the average of these errors. the properties 
    corresponding to maximum error can be selected for further
    study.
    """
    
    res=[]
    for p in df_error.prop.unique():

        tmp = df_error[df_error.prop == p]
        res.append([p, tmp.mean_error.max() ])

    res = pd.DataFrame(res, columns=['prop', 'max_mE'])
    res= res.sort_values(by='max_mE', ascending=False)
    return res


def plot_single_row(model_dict, props, nr=3,  w=6,h=8, legend_y=1.4, hspace=.4, wspace=0,
                    y_label_x=0.06, legend_ncol=1, img_name='tmp', plot_err_bar=False):
    
    fig, ax = plt.subplots(nr, len(props)//nr, figsize=(w,h))
    axes = ax.ravel()
    
    for mi, pv in enumerate(props):
        for j, (k, v) in enumerate(model_dict.items()):
            etot_df, c = v
            e_mean = etot_df[etot_df.prop == pv]['mean_error'].values
            e_sd = etot_df[etot_df.prop == pv]['sdev_error'].values
            prop = etot_df[etot_df.prop == pv]['pvalue'].values

            if plot_err_bar:
                axes[mi].errorbar(prop, e_mean, yerr=e_sd, marker='s', color=c, label=k)
            else:
                axes[mi].plot(prop, e_mean, marker='s', color=c, label=k)

            if pv in pv_rename:
                axes[mi].set_xlabel(f'{pv_rename[pv]}', fontsize=16)
            else:
                axes[mi].set_xlabel(f'{pv}', fontsize=16)
                
            axes[mi].tick_params(axis='both', which='major', labelsize=16)
        if mi ==0:
            axes[mi].legend(prop={'size':16},  ncol=legend_ncol, bbox_to_anchor=[0, legend_y], loc='upper left')
            
    fig.text(y_label_x, 0.5, 'Absolute Prediction Error', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.tight_layout()
    plt.savefig(img_name, dpi=150)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help="path to save results", type=str, required=False, default='ctrpv2')
    parser.add_argument('--scale', help="whether the descriptors should be scaled", type=str, required=False, default='')
    parser.add_argument('--fname_suffix', help="suffix for results saving path", type=str, required=False, default=None)
    parser.add_argument('--drug_feature_path', help="path of the drug feature dataframe. 1st column should have smiles, with column name smiles", type=str, required=False, default=None)

    parser.add_argument('--n_experiments', help="number of training runs", type=int, required=False, default=10)
    parser.add_argument('--graphdrp_res_path', help="GraphDRP results path", type=str, required=False, default=None)
    parser.add_argument('--deepttc_res_path', help="DeepTTC results path", type=str, required=False, default=None)
    parser.add_argument('--hidra_res_path', help="HiDRA results path", type=str, required=False, default=None)
    parser.add_argument('--et_res_path', help="ExtraTrees results path", type=str, required=False, default=None)

    parser.add_argument('--gnn_res_path', help="GNN results path", type=str, required=False, default=None)
    parser.add_argument('--smiles_res_path', help="SMILES_ENC results path", type=str, required=False, default=None)
    parser.add_argument('--morgan_res_path', help="Morgan_ENC results path", type=str, required=False, default=None)
    parser.add_argument('--descriptor_res_path', help="Descriptor_ENC results path", type=str, required=False, default=None)

    


    args = parser.parse_args()
  

    if args.fname_suffix:
        save_path = args.save_path+'_'+args.fname_suffix
    os.makedirs(save_path, exist_ok=True)
    
    
    expl_features, dict_modules, exaplainable = get_explainable_features()
    
       
    fe = pd.read_csv(args.drug_feature_path)


    if args.scale=='minmax':
        sc = MinMaxScaler()
        fe.iloc[:, 1:] = sc.fit_transform(fe.iloc[:, 1:])
    elif args.scale=='standard':
        sc = StandardScaler()
        fe.iloc[:, 1:] = sc.fit_transform(fe.iloc[:, 1:])
        
        
    features = remove_nonfloat_features(fe)
    fe_tmp = fe[features]
    print(fe.shape)


    
    
    # find continuous and cateforical features
    cont, cat =  find_cont_and_cat_features(fe_tmp)
    print(len(features), len(cont), len(cat))
    # get the bins
    conditions = get_conditions(fe, cont, nbins=10)
    
    experiment_nums = range(args.n_experiments)

    gdrp_grouped_avg = get_grouped_and_averaged_results(out_dir=args.graphdrp_res_path, runs=experiment_nums)
    dptc_grouped_avg = get_grouped_and_averaged_results(out_dir=args.deepttc_res_path, runs=experiment_nums)
    hdra_grouped_avg = get_grouped_and_averaged_results(out_dir=args.hidra_res_path, runs=experiment_nums)    
    et_grouped_avg = get_grouped_and_averaged_results(out_dir=args.et_res_path, runs=experiment_nums)
    
    gnnenc_grouped_avg = get_grouped_and_averaged_results(out_dir=args.gnn_res_path, runs=experiment_nums)
    trnsfenc_grouped_avg = get_grouped_and_averaged_results(out_dir=args.smiles_res_path, runs=experiment_nums)
    mrgnenc_grouped_avg = get_grouped_and_averaged_results(out_dir=args.morgan_res_path, runs=experiment_nums)
    desenc_grouped_avg = get_grouped_and_averaged_results(out_dir=args.descriptor_res_path, runs=experiment_nums)
    
    gdrp_grouped_avg.to_csv(f'{save_path}/gdrp_grouped_avg.csv', index=False)
    dptc_grouped_avg.to_csv(f'{save_path}/dptc_grouped_avg.csv', index=False)
    hdra_grouped_avg.to_csv(f'{save_path}/hdra_grouped_avg.csv', index=False)
    et_grouped_avg.to_csv(f'{save_path}/et_grouped_avg.csv', index=False)
    gnnenc_grouped_avg.to_csv(f'{save_path}/gnnenc_grouped_avg.csv', index=False)
    trnsfenc_grouped_avg.to_csv(f'{save_path}/trnsfenc_grouped_avg.csv', index=False)
    mrgnenc_grouped_avg.to_csv(f'{save_path}/mrgnenc_grouped_avg.csv', index=False)
    desenc_grouped_avg.to_csv(f'{save_path}/desenc_grouped_avg.csv', index=False)
    
    #### error by feature domains for continuous features
    # only continuous features are considered in the conditions here
    
    de_dptc, _ = error_by_feature_domains(fe, dptc_grouped_avg, conditions)
    de_gdrp, _ = error_by_feature_domains(fe, gdrp_grouped_avg, conditions)
    de_hdra, _ = error_by_feature_domains(fe, hdra_grouped_avg, conditions)
    de_et, _ = error_by_feature_domains(fe, et_grouped_avg, conditions)
    
    de_gnnenc, _ = error_by_feature_domains(fe, gnnenc_grouped_avg, conditions)
    de_trnsfenc, _ = error_by_feature_domains(fe, trnsfenc_grouped_avg, conditions)
    de_mrgnenc, _ = error_by_feature_domains(fe, mrgnenc_grouped_avg, conditions)
    de_desenc, _ = error_by_feature_domains(fe, desenc_grouped_avg, conditions)
    
    # error by feature domain for categorical features
    decat_dptc = cat_domain_error_for_run(dptc_grouped_avg, fe, cat)
    decat_gdrp = cat_domain_error_for_run(gdrp_grouped_avg, fe, cat)
    decat_hdra = cat_domain_error_for_run(hdra_grouped_avg, fe, cat)
    decat_et = cat_domain_error_for_run(et_grouped_avg, fe, cat)
    
    decat_gnnenc = cat_domain_error_for_run(gnnenc_grouped_avg, fe, cat)
    decat_trnsfenc = cat_domain_error_for_run(trnsfenc_grouped_avg, fe, cat)
    decat_mrgnenc = cat_domain_error_for_run(mrgnenc_grouped_avg, fe, cat)
    decat_desenc = cat_domain_error_for_run(desenc_grouped_avg, fe, cat)
    
    
    md_cont = {'gdrp': de_gdrp, 'dptc':de_dptc, 'hdra':de_hdra, 'et': de_et}
    md_cat = {'gdrp': decat_gdrp, 'dptc':decat_dptc,   'hdra':decat_hdra, 'et': decat_et}
    
    concat_cont = concat_mean_error_dfs(md_cont, ftype='cont')
    concat_cat = concat_mean_error_dfs(md_cat, ftype='cat')

    concat_cont2 = get_sdev_max(concat_cont)
    concat_cat2 = get_sdev_max(concat_cat)
    
    
    
    de_dptc['pvalue'] = 0.5*(de_dptc['low'] + de_dptc['high'])
    de_gdrp['pvalue'] = 0.5*(de_gdrp['low'] + de_gdrp['high'])
    de_hdra['pvalue'] = 0.5*(de_hdra['low'] + de_hdra['high'])
    de_et['pvalue'] = 0.5*(de_et['low'] + de_et['high'])
    de_gnnenc['pvalue'] = 0.5*(de_gnnenc['low'] + de_gnnenc['high'])
    de_trnsfenc['pvalue'] = 0.5*(de_trnsfenc['low'] + de_trnsfenc['high'])
    de_mrgnenc['pvalue'] = 0.5*(de_mrgnenc['low'] + de_mrgnenc['high'])
    de_desenc['pvalue'] = 0.5*(de_desenc['low'] + de_desenc['high'])
    
    
    etot_dptc = pd.concat([de_dptc, decat_dptc], axis=0)
    etot_gdrp = pd.concat([de_gdrp, decat_gdrp], axis=0)
    etot_hdra = pd.concat([de_hdra, decat_hdra], axis=0)
    etot_et = pd.concat([de_et, decat_et], axis=0)
    etot_gnnenc = pd.concat([de_gnnenc, decat_gnnenc], axis=0)
    etot_trnsfenc = pd.concat([de_trnsfenc, decat_trnsfenc], axis=0)
    etot_mrgnenc = pd.concat([de_mrgnenc, decat_mrgnenc], axis=0)
    etot_desenc = pd.concat([de_desenc, decat_desenc], axis=0)

    etot_dptc.to_csv(f'{save_path}/etot_dptc.csv', index=False)
    etot_gdrp.to_csv(f'{save_path}/etot_gdrp.csv', index=False)
    etot_hdra.to_csv(f'{save_path}/etot_hdra.csv', index=False)
    etot_et.to_csv(f'{save_path}/etot_et.csv', index=False)
    etot_gnnenc.to_csv(f'{save_path}/etot_gnnenc.csv', index=False)
    etot_trnsfenc.to_csv(f'{save_path}/etot_trnsfenc.csv', index=False)
    etot_mrgnenc.to_csv(f'{save_path}/etot_mrgnenc.csv', index=False)
    etot_desenc.to_csv(f'{save_path}/etot_desenc.csv', index=False)
    
    
    pv_rename = {'log_sol': 'logS', 'MW': 'molecular weight', 'logp': 'LogP'}
    colors = {'DrugCell': 'b', 'HiDRA': '#A4E0F8', 'GraphDRP':'r', 'DRPreter': 'orange', 'DeepTTC': 'purple', 'Paccmann': '#A9A4F8',  'ET': 'g',
             'Morgan': '#34495E', 'Graph':'#FF6833', 'Transformer':'#D36ED3',  'Descriptor':'#58D68D'}
    
    
    nc = 5
    model_dict = { 
                'GraphDRP': (etot_gdrp, colors['GraphDRP']), # graph
                'Graph':(etot_gnnenc, colors['Graph'] ),
        
                'HiDRA': (etot_hdra, colors['HiDRA']  ), # morgan
                'Morgan':(etot_mrgnenc, colors['Morgan'] ),
        
                'DeepTTC': (etot_dptc, colors['DeepTTC']), # smiles
                 'SMILES':(etot_trnsfenc, colors['Transformer'] ),
        
                  'ET':(etot_et, colors['ET']), # descriptor
              'Descriptor':(etot_desenc, colors['Descriptor']),
             
                 }
    
    plot_single_row(model_dict, [ 'log_sol', 'MW', 'logp', 'nHBDon'], nr=1,w=12, h=7,legend_y=1.2,
                    legend_ncol=4, wspace=.4,
                    img_name=f'{save_path}/prop_errors.png', plot_err_bar=False)



