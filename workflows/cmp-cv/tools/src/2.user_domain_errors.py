import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help="path to save results", type=str, required=False, default='../ctrpv2')
    parser.add_argument('--props', nargs='+', help='property list seperated by sapce')
    
    args = parser.parse_args()

    save_path = args.save_path
    props = args.props  # props = ['GATS1Z', 'C3SP3', 'SlogP_VSA4', 'JGI2']

    # save_path='../ctrpv2'
    etot_dptc = pd.read_csv(f'{save_path}/etot_dptc.csv')
    etot_gdrp = pd.read_csv(f'{save_path}/etot_gdrp.csv')
    etot_hdra = pd.read_csv(f'{save_path}/etot_hdra.csv')
    etot_et = pd.read_csv(f'{save_path}/etot_et.csv')
    etot_gnnenc = pd.read_csv(f'{save_path}/etot_gnnenc.csv')
    etot_trnsfenc = pd.read_csv(f'{save_path}/etot_trnsfenc.csv')
    etot_mrgnenc = pd.read_csv(f'{save_path}/etot_mrgnenc.csv')
    etot_desenc = pd.read_csv(f'{save_path}/etot_desenc.csv')

    nc = 5
    colors = {'DrugCell': 'b', 'HiDRA': '#0E79F2', 'GraphDRP':'r', 'DRPreter': 'orange', 'DeepTTC': '#CA04FB', 'Paccmann': '#A9A4F8',  'ET': 'g',
            'Morgan': '#0ED0F2', 'Graph':'#FF6833', 'Transformer':'#483843',  'Descriptor':'#29F20E'}
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
    

    legend_ncol=4
    legend_y=1.21
    w=18
    h=6
    plot_err_bar=False
    fig, ax = plt.subplots(1, 4, figsize=(w,h))
    label_format = '{:,.2f}'
    axes = ax.ravel()
    for mi, pv in enumerate(props):
        for j, (k, v) in enumerate(model_dict.items()):
            etot_df, c = v
            # print(mi, j)
            tmp = etot_df.dropna(subset=['mean_error'])
            tmp.pvalue = tmp.pvalue.astype(float)
            e_mean = tmp[tmp.prop == pv]['mean_error'].values
            e_sd = tmp[tmp.prop == pv]['sdev_error'].values
            prop = tmp[tmp.prop == pv]['pvalue'].values
        
            if plot_err_bar:
                axes[mi].errorbar(prop, e_mean, yerr=e_sd, marker='s', color=c, label=k)
            else:
                axes[mi].plot(prop, e_mean, marker='o', ms=5, linestyle=None, color=c, label=k)
            axes[mi].set_xlabel(f'{pv}', fontsize=14)
            axes[mi].tick_params(axis='x', which='major', labelsize=None, rotation=90)
            axes[mi].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


        if mi ==0:
            axes[mi].legend(prop={'size':16},  ncol=legend_ncol, bbox_to_anchor=[0, legend_y], loc='upper left')
    fig.text(0.08, 0.5, 'Absolute Prediction Error', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    plt.subplots_adjust(wspace=.2, bottom=0.2)


    plt.savefig('./higherr_th.png')
