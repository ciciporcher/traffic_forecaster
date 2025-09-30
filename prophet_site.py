# %% import libraries
import pandas as pd
import numpy as np
from datetime import datetime

import pickle
from tqdm import tqdm
tqdm.pandas()

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

import fcapacity as cap

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

# %% input config
tec = 'lte' # tecnologia

# inicio dos dados
start_date = '2022-06-01'
end_date = f'2025-06-01'

# horizonte de predição
horizon = 12

# define atributo target
if tec == 'nr':
    target_var = 'VOL_NR'
elif tec == 'lte':
    target_var = 'VOL_LTE'

# incluir variáveis exogenas
include_exog = True
detail_prophet_exog = True
if tec == 'nr':
    exogenous_var = ['UE','PRB_UL','VOL_LTE']
elif tec == 'lte':
    exogenous_var = ['UE','PRB_DL','VOL_NR']

print('--- I N P U T S')
print(f'Projeção de Volume {tec.upper()}')
print(f'Dados históricos de {start_date} a {end_date}')
print(f'Horizonte de previsão de {horizon} meses')
if include_exog:
    print(f'O modelo será treinado considerando as seguintes variáveis exógenas: \n{exogenous_var}')

# %% import data kpis
path = fr'caminho_para_os_dados'

date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
yymm_list = [d.strftime('%y%m') for d in date_range]
                                
if tec == 'nr':
    tec_secundaria = 'lte'
elif tec == 'lte':
    tec_secundaria = 'nr'

df = cap.importKPI(yymm_list, tec, path, 'site')

df_outro = cap.importKPI(yymm_list, tec_secundaria, path, 'site')
df_outro = df_outro[['DATA', 'year', 'month', 'SITE', f'VOL_{tec_secundaria.upper()}']]
# adiciona vol da tec secundária
df = df.merge(df_outro, on=['DATA', 'year', 'month', 'SITE'], how='left')

print('Importação dos KPIs concluída.')

# %% group ibge
df = cap.groupbyIBGE(df, tec, tec_secundaria, exogenous_var)

if tec == 'lte':
    filename = f'dataframe_historico_nao_normalizado.pkl'
    with open(filename, 'wb') as arquivo:
        pickle.dump(df, arquivo)

# normalizar kpis
volume_cols = ['VOL_LTE', 'VOL_NR']
volume_scaler = MinMaxScaler()
all_volume_values = df[volume_cols].values.ravel().reshape(-1, 1)
volume_scaler.fit(all_volume_values)
for col in volume_cols:
    df[col] = volume_scaler.transform(df[[col]])
filename = 'volume_scaler_mestre.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(volume_scaler, arquivo)
ue_scaler = MinMaxScaler()
df['UE'] = ue_scaler.fit_transform(df[['UE']])
prb_scaler = MinMaxScaler()
prb = [var for var in exogenous_var if var.startswith('PRB')]
df[prb] = prb_scaler.fit_transform(df[prb])

# %%
df_temp = df.groupby('MUNICIPIO')[f'VOL_{tec.upper()}'].agg(['size','sum'])
with open(f'sites.pkl', 'rb') as f:
        sites = pickle.load(f)

if tec == 'nr':
    for mun in tqdm(df.MUNICIPIO.unique()):
        df_ibge = sites.loc[sites.ibge == mun]
        dt_ativacao_nr = df_ibge.data_ativacao_nr.min()
        df_temp.loc[mun, 'data_ativacao_nr'] = dt_ativacao_nr
        if pd.notna(dt_ativacao_nr):
            df_temp.loc[mun, 'meses_5G'] = len(
                pd.date_range(start=dt_ativacao_nr,
                              end=datetime.now(),
                              freq='MS'))-1
        else:
            df_temp.loc[mun, 'meses_5G'] = np.nan

# drop municípios com pouca informação
if tec == 'nr':
    list_mun = list(df_temp.loc[df_temp.meses_5G >= 6].index)
else:
    list_mun = list(df_temp.loc[df_temp['sum'] > 0].index)

if tec == 'lte':
    filename = f'dataframe_historico.pkl'
    with open(filename, 'wb') as arquivo:
        pickle.dump(df, arquivo)

# %% list sazonais
total_por_mun = sites.groupby(by='ibge')['sazonalidade'].count()
sazonal_por_mun = sites[sites.sazonalidade == True].groupby(by='ibge')['sazonalidade'].count()
percent_sazonal = (sazonal_por_mun/total_por_mun).fillna(0)

lst_sazonais = percent_sazonal[percent_sazonal > 0.5].index

filename = f'lst_sazonais.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(lst_sazonais, arquivo)
# %% prophet mun
df_mun_pred = pd.DataFrame(columns=['municipio','error','r2','h'] + [f'beta_{var}' for var in exogenous_var] + [f'y_hat_{i}' for i in range(horizon)])

if tec == 'nr':
    with open(f'dataframe_historico.pkl', 'rb') as f:
        df_historico = pickle.load(f)
    df_historico.DATA = pd.to_datetime(df_historico.DATA)

for mun in tqdm(list_mun):
    df_temp = df.loc[df.MUNICIPIO == mun].dropna(subset=exogenous_var[0]).reset_index(drop=True)
    start_date_mun = df_temp.DATA.min()

    if len(df_temp) > 5:
        if mun in lst_sazonais:
            include_season = True
        else:
            include_season = False

        exog_var = cap.prophet_exog(df_temp, tec, exogenous_var)

        if include_season and tec == 'nr' and len(df_temp) < 24:
            exog_var = []
            prophet_df = cap.prophet_df_mun(df_temp, exog_var, target_var,
                                            include_exog, start_date_mun)
            df_lte_mun = df_historico.loc[df_historico.MUNICIPIO == mun,
                                          ['DATA','VOL_LTE']].copy()
            df_lte_mun.columns = ['ds', 'VOL_LTE']
            h, mae, r2 = cap.prophet_test_nr_seasonal(prophet_df, horizon, df_lte_mun, exog_var)
            vol_hat, model = cap.prophet_future_nr_seasonal(prophet_df, horizon, df_lte_mun, exog_var)
            exog_var = ['VOL_LTE']

        elif df_temp[f'VOL_{tec.upper()}'][-6:].sum() == 0:
            vol_hat = pd.Series([0]*horizon)

        else:
            prophet_df = cap.prophet_df_mun(df_temp, exog_var, target_var,
                                            include_exog, start_date_mun)
            h, mae, r2 = cap.prophet_test_mun(prophet_df, horizon,
                        exog_var, include_exog, include_season)
            vol_hat, model = cap.prophet_future_mun(prophet_df, horizon,
                        exog_var, include_exog, include_season)

        if vol_hat.sum() == 0:
            model_beta_dict = {f'beta_{var}': np.nan for var in exog_var}
            vol_hat_dict = {f'y_hat_{i}': vol_hat.iloc[i] for i in range(len(vol_hat))}
        elif len(vol_hat) == horizon:
            model_beta_dict = {f'beta_{var}': model.params['beta'][0][i] for i,var in enumerate(exog_var)}
            vol_hat_dict = {f'y_hat_{i}': vol_hat.iloc[i] if vol_hat.iloc[i] >= 0 else 0  for i in range(len(vol_hat))}
        else:
            model_beta_dict = {f'beta_{var}': np.nan for var in exog_var}
            vol_hat_dict = {f'y_hat_{i}': np.nan for i in range(horizon)}

        df_temp_pred = pd.DataFrame([{**{'municipio': mun, 'error': mae, 'r2': r2, 'h': h},
                                      **model_beta_dict,
                                      **vol_hat_dict}])
        df_mun_pred = pd.concat([df_mun_pred, df_temp_pred], ignore_index=True)

filename = f'prophet_mun_{tec}.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(df_mun_pred, arquivo)

# %%
