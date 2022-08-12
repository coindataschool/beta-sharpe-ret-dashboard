import statsmodels.api as sm
import pandas as pd

def extract_frame_from_dune_data(dune_data, date_col='day'):    
    dd = dune_data['data']['get_result_by_result_id']
    df = pd.json_normalize(dd, record_prefix='')
    df = df.loc[:, df.columns.str.startswith('data')]
    df.columns = df.columns.str.replace('data.', '', regex=False)
    df['date'] = pd.to_datetime(df[date_col].str.replace('T.*', '', regex=True))
    if date_col != 'date':
        df = df.drop(date_col, axis=1)
    df = df.set_index('date')
    # drop the last row cuz it may not always be a full day
    return df.iloc[:-1, :]

def calc_beta(df_ret, token='BTC', benchmark='SP500'):
    X = df_ret[benchmark]
    y = df_ret[token]
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    results = model.fit()
    return {'beta': results.params[benchmark], 
            'p-val': results.pvalues['SP500'], 
            'R2': results.rsquared}

def annualize_tot_ret(tot_ret, dur_years):
    return (1+tot_ret)**(1/dur_years) - 1