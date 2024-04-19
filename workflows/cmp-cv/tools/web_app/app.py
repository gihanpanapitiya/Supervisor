import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# col1, col2 = st.columns(2)

save_path='ctrpv2'
etot_dptc = pd.read_csv(f'{save_path}/etot_dptc.csv')
etot_gdrp = pd.read_csv(f'cmp_ctrpv2_seed/etot_gdrp.csv')
etot_hdra = pd.read_csv(f'{save_path}/etot_hdra.csv')
etot_et = pd.read_csv(f'{save_path}/etot_et.csv')
etot_gnnenc = pd.read_csv(f'{save_path}/etot_gnnenc.csv')
etot_trnsfenc = pd.read_csv(f'{save_path}/etot_trnsfenc.csv')
etot_mrgnenc = pd.read_csv(f'{save_path}/etot_mrgnenc.csv')
etot_desenc = pd.read_csv(f'{save_path}/etot_desenc.csv')
etot_gdrp_pt = pd.read_csv(f'ctrpv2_pt/etot_gdrp.csv')



colors = {'DrugCell': 'b', 'HiDRA': '#0E79F2', 'GraphDRP':'r', 'DRPreter': 'orange', 'DeepTTC': '#CA04FB', 'Paccmann': '#A9A4F8',  'ET': 'g',
         'Morgan': '#0ED0F2', 'Graph':'#FF6833', 'Transformer':'#483843',  'Descriptor':'#29F20E'}
nc = 5
model_dict = { 
            'GraphDRP': (etot_gdrp, colors['GraphDRP']), # graph
            # 'GraphDRP-PT': (etot_gdrp_pt, 'yellow'), # graph
            # 'DRPreter': (etot_drprtr, colors['DRPreter']), 
            'Graph':(etot_gnnenc, colors['Graph'] ),
    
            # 'DrugCell': (etot_dc, colors['DrugCell']  ), # morgan
            'HiDRA': (etot_hdra, colors['HiDRA']  ),
            'Morgan':(etot_mrgnenc, colors['Morgan'] ),
    
            'DeepTTC': (etot_dptc, colors['DeepTTC']), # smiles
            # 'Paccmann': (etot_pcm,colors['Paccmann']),
             'SMILES':(etot_trnsfenc, colors['Transformer'] ),
    
              'ExtraTrees':(etot_et, colors['ET']), # descriptor
          'Descriptor':(etot_desenc, colors['Descriptor']),
         
             }


def get_model_rank_table(prop, prop_name):
    
    ms, names =[], []
    for j, (k, v) in enumerate(model_dict.items()):
        etot_df, c = v
        m = etot_df[etot_df.prop == prop][['mean_error']]
        ms.append(m)
        names.append(k)
        # break
    # m1 = etot_dptc[etot_dptc.prop == 'log_sol']
    # m2 = etot_gdrp[etot_gdrp.prop == 'log_sol']
    
    ms = pd.concat(ms, axis=1)
    ms.columns = names
    
    # print(ms.shape)
    
    des=[]
    for i in range(ms.shape[0]):
        a = ms.iloc[i,:].values.argsort()
        des.append(ms.columns[a].tolist())
    
    res2 = pd.DataFrame(des)
    res2.columns = [str(i+1) for i in range(res2.shape[1])]
    
    
    pv = etot_df[etot_df.prop == prop][['pvalue']].round(2)
    pv.reset_index(drop=True, inplace=True)
    
    pv.columns=[prop_name]
    res2 = pd.concat([pv, res2], axis=1)

    return res2


def get_ranks(df_res):
    prop = df_res.columns[0]
    n_bins  = df_res.shape[0]
    a = [{v:k+1 for k,v in dict(enumerate(df_res.iloc[i, 1:].values)).items()} for i in df_res.index]
    
    des = list(set(df_res.iloc[:, 1:].values.ravel()))
    ranks = dict(zip(df_res[prop], a))

    scores={}
    for name_des in des:
        vs = sum([a[i][name_des] for i in range(n_bins)])
        scores[name_des] = vs/n_bins
        # break
    
    return ranks, scores


options = ['nHBDon', 'logS', 'LogP', 'MW']
df_name = {'nHBDon': 'nHBDon', 'logS': 'log_sol', 'LogP':'logp', 'MW':'MW'}


selected_props = st.multiselect("Select Properties", options, default='logS')
# selected_props =['nHBDon', 'logS']


dict_selected={}
for prop in selected_props:
    # get the model rank table for each selected property
    dict_selected[prop] = get_model_rank_table(prop=df_name[prop], prop_name=prop)


# dict_selected['nHBDon']
res_logs = get_model_rank_table(prop='log_sol', prop_name='logS')

r_logs, s_logs = get_ranks(res_logs)
des_names = s_logs.keys()

rank_dicts={}
for prop, res_df in  dict_selected.items():
    rank, scores = get_ranks(res_df)
    rank_dicts[prop] = rank, scores # rank contains model ranks for each property value

slider_values={}
for prop, rank_scores in rank_dicts.items():
    rank, scores = rank_scores
    _values = list(rank.keys())
    slider_values[prop] = st.select_slider(prop, _values)


# print(slider_values)
# print(rank_dicts)
sums={}
dicts = [rank_dicts[prop][0][slider_values[prop]] for prop in selected_props]

selected_props

for d in des_names:
    s=0
    for c_dict in dicts:
        s+=c_dict[d]
    sums[d] = s/len(dicts)

sums = {k:v for k,v in sorted(sums.items(), key=lambda x: x[1])}

fc='white'
fig = plt.figure(figsize=(8,4))
fig.patch.set_facecolor(fc)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((fc))
# ax.hist(arr, bins=20)

x, y = np.array(list(sums.keys())), np.array(list(sums.values()))
plt.bar(x, y, color='#34495E')
# plt.bar(x , y)
# for i, v in enumerate(y):
#     plt.text(i - 0.25, v + 0.01, str(v))

plt.ylabel('Model Rank (lower is better)')
plt.xticks(rotation=90)

st.pyplot(fig)

# st.write(x, "nHBDon", hbdon_values)

# GENES


from data_utils import DataProcessor

data_type='ctrpv2'

etot_dptc = pd.read_csv(f'{data_type}/dptc_grouped_avg.csv')
etot_gdrp = pd.read_csv(f'cmp_{data_type}_seed/gdrp_grouped_avg.csv')
etot_hdra = pd.read_csv(f'{data_type}/hdra_grouped_avg.csv')
etot_et = pd.read_csv(f'{data_type}/et_grouped_avg.csv')
etot_gnnenc = pd.read_csv(f'{data_type}/gnnenc_grouped_avg.csv')
etot_trnsfenc = pd.read_csv(f'{data_type}/trnsfenc_grouped_avg.csv')
etot_mrgnenc = pd.read_csv(f'{data_type}/mrgnenc_grouped_avg.csv')
etot_desenc = pd.read_csv(f'{data_type}/desenc_grouped_avg.csv')

colors = {'DrugCell': 'b', 'HiDRA': '#0E79F2', 'GraphDRP':'r', 'DRPreter': 'orange', 'DeepTTC': '#CA04FB', 'Paccmann': '#A9A4F8',  'ExtraTrees': 'g',
         'Morgan': '#0ED0F2', 'Graph':'#FF6833', 'Transformer':'#483843',  'Descriptor':'#29F20E'}

model_res = { 'DeepTTC': etot_dptc, 
             'GraphDRP': etot_gdrp,
             'HiDRA': etot_hdra,
             'ExtraTrees': etot_et,
             'Graph': etot_gnnenc,
             'Transformer': etot_trnsfenc,
             'Morgan': etot_mrgnenc,
             'Descriptor': etot_desenc
            }

@st.cache_data
def get_mut_res(model_df, mut, genes):
    
    df = pd.merge(model_df, mut, left_on='cell_line_id', right_on='improve_sample_id', how='left') 

    # df['error'] = abs(df['true']-df['pred'])

    res =[]
    for gene in genes:

        # if gene in all_genes:
            tmp = df[df[gene] > 0]

            mut_error = tmp['err'].mean()
            res.append([mut_error, gene])

    # len(genes)

    res=pd.DataFrame(res, columns=['error', 'gene'])
    return res




dp = DataProcessor('data_ctrpv2/')
mut = dp.load_cell_mutation_data('data_ctrpv2/', gene_system_identifier="Entrez")
mut.reset_index(inplace=True)
    
avail_genes=['ATM', 'PTEN', 'RAD51D', 'HOXB13', 'BRCA2', 'BARD1', 'CDKN2A', 'PMS2', 'NBN', 'BRIP1', 'MUTYH', 'CDK4', 'NF1', 'CHEK2',
              'EPCAM', 'CDH1', 'RAD51C', 'PALB2', 'MSH2', 'MLH1', 'BRCA1', 'APC', 'STK11', 'TP53', 'BAP1', 'MSH6']

mut2 = mut.loc[:, ['improve_sample_id']+avail_genes]


default_genes = ['ATM', 'CHEK2', 'BRIP1', 'PTEN', 'BRCA1', 'STK11', 'RAD51D', 'CDH1', 'RAD51C', 'TP53', 'PALB2', 'BRCA2', 'BARD1', 'NF1']
selected_genes = st.multiselect("Select Genes", avail_genes, default=default_genes)
err_res = {}
for model, m_error in model_res.items():
    err_res[model] = get_mut_res(m_error, mut2, selected_genes)


fc='white'
fig = plt.figure(figsize=(12,8))
fig.patch.set_facecolor(fc)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((fc))

for model, df in err_res.items():
    plt.plot(df.index, df.error, '.-', label=model, color=colors[model])
plt.legend(ncol=2, frameon=False, loc=1)
plt.xticks(df.index, df.gene, rotation=90);
plt.ylabel("Prediction Error");
st.pyplot(fig)
