import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import argparse

def get_model_rank_table(prop, prop_name):
    
    ms, names =[], []
    for j, (k, v) in enumerate(model_dict.items()):
        etot_df, c = v
        m = etot_df[etot_df.prop == prop][['mean_error']]
        ms.append(m)
        names.append(k)

    ms = pd.concat(ms, axis=1)
    ms.columns = names
    
    
    des=[]
    for i in range(ms.shape[0]):
        a = ms.iloc[i,:].values.argsort()
        des.append(ms.columns[a].tolist())
    
    res2 = pd.DataFrame(des)
    res2.columns = [str(i+1) for i in range(res2.shape[1])]
    
    
    pv = etot_df[etot_df.prop == prop][['pvalue']]
    pv.reset_index(drop=True, inplace=True)
    
    pv.columns=[prop_name]
    res2 = pd.concat([pv, res2], axis=1)

    return res2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help="path to save results", type=str, required=False, default='../ctrpv2')
    parser.add_argument('--prop', help="property name", type=str, required=False, default='log_sol')
    parser.add_argument('--prop_name', help="property name in the results table", type=str, required=False, default='logS')
    args = parser.parse_args()

    save_path = args.save_path

    etot_dptc = pd.read_csv(f'{save_path}/etot_dptc.csv')
    etot_gdrp = pd.read_csv(f'{save_path}/etot_gdrp.csv')
    etot_hdra = pd.read_csv(f'{save_path}/etot_hdra.csv')
    etot_et = pd.read_csv(f'{save_path}/etot_et.csv')
    etot_gnnenc = pd.read_csv(f'{save_path}/etot_gnnenc.csv')
    etot_trnsfenc = pd.read_csv(f'{save_path}/etot_trnsfenc.csv')
    etot_mrgnenc = pd.read_csv(f'{save_path}/etot_mrgnenc.csv')
    etot_desenc = pd.read_csv(f'{save_path}/etot_desenc.csv')


    prop = args.prop # 'log_sol'
    prop_name = args.prop_name # 'logS'

    colors = {'DrugCell': 'b', 'HiDRA': '#0E79F2', 'GraphDRP':'r', 'DRPreter': 'orange', 'DeepTTC': '#CA04FB', 'Paccmann': '#A9A4F8',  'ET': 'g',
            'Morgan': '#0ED0F2', 'Graph':'#FF6833', 'Transformer':'#483843',  'Descriptor':'#29F20E'}
    nc = 5
    model_dict = { 
                'GraphDRP': (etot_gdrp, colors['GraphDRP']), # graph
                'Graph':(etot_gnnenc, colors['Graph'] ),
        
                'HiDRA': (etot_hdra, colors['HiDRA']  ),
                'Morgan':(etot_mrgnenc, colors['Morgan'] ),
        
                'DeepTTC': (etot_dptc, colors['DeepTTC']), # smiles
                'SMILES':(etot_trnsfenc, colors['Transformer'] ),
        
                'ExtraTrees':(etot_et, colors['ET']), # descriptor
            'Descriptor':(etot_desenc, colors['Descriptor']),
            
                }




    res2 = get_model_rank_table(prop=prop, prop_name=prop_name)

    print(res2)
