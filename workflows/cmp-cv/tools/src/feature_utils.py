def get_high_corr_features():
    
    from data_utils import DataProcessor
    from data_utils import add_smiles

    dp = DataProcessor('benchmark-data-imp-2023')


    tr=dp.load_drug_response_data(data_path='CCLE/SWnet/Data/', data_type='CCLE', split_id=0, split_type='train')
    vl=dp.load_drug_response_data(data_path='CCLE/SWnet/Data/', data_type='CCLE', split_id=0, split_type='val')
    ts=dp.load_drug_response_data(data_path='CCLE/SWnet/Data/', data_type='CCLE', split_id=0, split_type='test')

    smiles_df = dp.load_smiles_data('CCLE/SWnet/Data/')

    df = pd.concat([tr, vl, ts], axis=0)
    df.reset_index(drop=True, inplace=True)
    df = add_smiles(smiles_df, df, 'ic50')


    # dffe = pd.merge(df, fe.loc[:,  ['smiles']+expl_features ], on='smiles', how='left')
    dffe = pd.merge(df, fe, on='smiles', how='left')
    c = dffe.corr()

    high_corr =  c['ic50'].sort_values(ascending=False)
    hc_keys = high_corr.keys()[1:]
    hc_values = high_corr.values[1:]
    
    return hc_keys, hc_values, high_corr, dffe
    # return high_corr

# hc_keys, hc_values, high_corr, dfc  = get_high_corr_features()

def get_explainable_features():
    from mordred import Calculator, descriptors
    
    n_all = Calculator(descriptors, ignore_3D=False).descriptors
    n_2D = Calculator(descriptors, ignore_3D=True).descriptors

    modules=[]
    for i in n_2D:
        modules.append(i.__module__.split('.')[1])

    dict_modules = {k:[] for k in set(modules)}
    # modules=[]
    for i in n_2D:
        module = i.__module__.split('.')[1]
        n = str(i)
        dict_modules[module].extend([n])
        # break

    exaplainable = ['Aromatic', 'AtomCount', 'CPSA', 'TopoPSA', 'LogS', 'AcidBase',
                    'McGowanVolume', 'FragmentComplexity','CarbonTypes', 'BertzCT', 'BondCount',
                    'Polarizability', 'Weight', 'RotatableBond', 'RingCount', 'HydrogenBond']
# 'EState'
    expl_props=[]
    for i in exaplainable:
        # print(dict_modules[i])
        expl_props.extend(dict_modules[i])
        
    return expl_props, dict_modules, exaplainable
