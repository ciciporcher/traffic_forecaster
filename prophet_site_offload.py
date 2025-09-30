# %% import libraries
import pandas as pd
import numpy as np

import pickle
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import fcapacity as cap

import logging

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

# %% input config
tec = 'lte' # tecnologia - MANTER LTE PARA ESTUDO OFFLOAD!

# inicio dos dados
start_date = '2022-07-01'
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

# %% list sazonais
with open(f'sites.pkl', 'rb') as f:
        sites = pickle.load(f)

total_por_mun = sites.groupby(by='ibge')['sazonalidade'].count()
sazonal_por_mun = sites[sites.sazonalidade == True].groupby(by='ibge')['sazonalidade'].count()
percent_sazonal = (sazonal_por_mun/total_por_mun).fillna(0)

lst_sazonais = percent_sazonal[percent_sazonal > 0.5].index

filename = f'lst_sazonais.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(lst_sazonais, arquivo)

# %% group ibge
df = cap.groupbyIBGE(df, tec, tec_secundaria, exogenous_var)

# normalizar kpis
volume_cols = ['VOL_LTE', 'VOL_NR']
volume_scaler = MinMaxScaler()
all_volume_values = df[volume_cols].values.ravel().reshape(-1, 1)
volume_scaler.fit(all_volume_values)
for col in volume_cols:
    df[col] = volume_scaler.transform(df[[col]])
ue_scaler = MinMaxScaler()
df['UE'] = ue_scaler.fit_transform(df[['UE']])
prb_scaler = MinMaxScaler()
prb = [var for var in exogenous_var if var.startswith('PRB')][0]
df[[prb]] = prb_scaler.fit_transform(df[[prb]])

df_temp = df.groupby('MUNICIPIO')[f'VOL_{tec.upper()}'].agg(['size','sum'])
# drop municípios com pouca informação
drop_ind = df_temp.loc[(df_temp['size'] <= horizon) | (df_temp['sum'] == 0)].index
list_mun = [m for m in df_temp.index.unique() if m not in drop_ind]

# add colunas com infos para lstm
df['volume_total'] = df['VOL_LTE'] + df['VOL_NR']
df['percent_lte'] = df.VOL_LTE / df.volume_total
df['percent_nr'] = df.VOL_NR / df.volume_total

date_range_12m = pd.date_range(start=start_date,
                               end=pd.to_datetime(end_date) + pd.DateOffset(months=horizon),
                               freq='MS')

for mun in tqdm(list_mun):
    df_mun_temp = df.loc[df.MUNICIPIO == mun]
    df_ibge = sites.loc[sites.ibge == mun]
    data_ativacao_nr = df_ibge.data_ativacao_nr.min()
    sites4G = [len(df_ibge.loc[(df_ibge.data_ativacao_lte <= pd.to_datetime(date)) &
                               (df_ibge.LTE == 1)]) for date in date_range_12m]
    sites5G = [len(df_ibge.loc[(df_ibge.dt_ativacao_nr <= pd.to_datetime(date)) &
                               (df_ibge.NR == 1)]) for date in date_range_12m] 
    
    vol_total, percent_lte, percent_nr = [], [], []
    for date in date_range_12m:
        try:
            row = df_mun_temp.loc[
                (df_mun_temp.year.astype(int) == date.year) & 
                (df_mun_temp.month.astype(int) == date.month), 
                ['volume_total', 'percent_lte', 'percent_nr']
            ].values[0]
            vol_total.append(row[0])
            percent_lte.append(row[1])
            percent_nr.append(row[2])
        except:
            vol_total.append(np.nan)
            percent_lte.append(np.nan)
            percent_nr.append(np.nan)

    df_temp = pd.DataFrame({
        'municipio': mun,
        'ano_mes': date_range_12m,
        'meses_5G': np.nan,
        'sites_lte': sites4G,
        'sites_nr': sites5G,
        'volume_total': vol_total,
        'percent_lte': percent_lte,
        'percent_nr': percent_nr})
    if pd.isnull(data_ativacao_nr):
        df_temp['meses_5G'] = np.nan
    else:
        df_temp['meses_5G'] = [(d.year - data_ativacao_nr.year)*12 + (d.month - data_ativacao_nr.month) for d in df_temp['ano_mes']]
    
    if mun == df.MUNICIPIO.unique()[0]:
        df_offload = df_temp
    else:
        df_offload = pd.concat([df_offload, df_temp], ignore_index=True)

# %% prophet volume
lst_mun_2_model = df_offload.loc[(df_offload.ano_mes == end_date) &
                                 (df_offload.meses_5G < 6) & (df_offload.meses_5G > -12),
                                 'municipio'].values
# prophet mun
exogenous_var_lte = [var for var in exogenous_var if not var.endswith('NR')]

df_mun_pred = pd.DataFrame(columns=['municipio','error','h'] + [f'beta_{var}' for var in exogenous_var_lte] + [f'y_hat_{i}' for i in range(horizon)])

for mun in tqdm(lst_mun_2_model):
    df_temp = df.loc[df.MUNICIPIO == mun].dropna(subset=exogenous_var_lte[0]).reset_index(drop=True)
    exog_var = cap.prophet_exog(df_temp, tec, exogenous_var_lte)

    prophet_df = cap.prophet_df_mun(df_temp, exog_var, target_var, include_exog, start_date)

    if mun in lst_sazonais:
        include_season = True
    else:
        include_season = False

    h, mape = cap.prophet_test_mun(prophet_df, horizon,
                                   exog_var, include_exog, include_season)
    vol_hat, model = cap.prophet_future_mun(prophet_df, horizon,
                                            exog_var, include_exog, include_season)

    if len(vol_hat) == 12:
        model_beta_dict = {f'beta_{var}': model.params['beta'][0][i] for i,var in enumerate(exog_var)}
        vol_hat_dict = {f'y_hat_{i}': vol_hat.iloc[i] for i in range(len(vol_hat))}
    else:
        model_beta_dict = {f'beta_{var}': np.nan for var in exogenous_var_lte}
        vol_hat_dict = {f'y_hat_{i}': np.nan for i in range(horizon)}

    df_temp_pred = pd.DataFrame([{**{'municipio': mun, 'error': mape, 'h': h},
                                  **model_beta_dict,
                                  **vol_hat_dict}])
    df_mun_pred = pd.concat([df_mun_pred, df_temp_pred], ignore_index=True)

# %%
df_offload = df_offload.loc[df_offload.municipio.isin(lst_mun_2_model)]
window_size = 6
inicial_dt = pd.to_datetime(end_date) - pd.DateOffset(months=window_size)
horizon_dt = pd.to_datetime(end_date) + pd.DateOffset(months=12)

df_offload = df_offload.loc[(df_offload.ano_mes > inicial_dt) &
                                      (df_offload.ano_mes <= horizon_dt)]
for mun in tqdm(lst_mun_2_model):
    ind_mun = df_offload.loc[(df_offload.municipio == mun) &
                                  (df_offload.ano_mes > end_date)].index
    vol_future = df_mun_pred.loc[df_mun_pred.municipio == mun,
                                 [f'y_hat_{i}' for i in range(horizon)]].values[0]
    
    df_offload.loc[ind_mun, 'volume_total'] = vol_future

# %%
attention_layer = tf.keras.layers.Attention(name='attention_layer')

with open(f'model_lstm_offload.pkl', 'rb') as f:
    offload_model = pickle.load(f)

# %%
with open(f'scalers_offload.pkl', 'rb') as f:
    scalers = pickle.load(f)

months_offload = np.array(df_offload.meses_5G).reshape(-1,1)
df_offload.meses_5G = scalers['meses_5G'].transform(months_offload).reshape(1,-1)[0]

sites_lte = np.array(df_offload.sites_lte).reshape(-1,1)
df_offload.sites_lte = scalers['sites'].transform(sites_lte).reshape(1,-1)[0]
sites_nr = np.array(df_offload.sites_nr).reshape(-1,1)
df_offload.sites_nr = scalers['sites'].transform(sites_nr).reshape(1,-1)[0]

vol_offload = np.array(df_offload.volume_total).reshape(-1,1)
df_offload.volume_total = scalers['volume'].transform(vol_offload).reshape(1,-1)[0]

# %%
features = ['meses_5G','sites_lte','sites_nr','volume_total']

for mun in tqdm(lst_mun_2_model):
    dados_features = df_offload.loc[df_offload.municipio == mun][features].values
    X = []
    for i in range(len(dados_features) - window_size):
        sequencia_x = dados_features[i:(i+window_size)]
        X.append(sequencia_x)
    X = np.array(X)

    y_hat = offload_model.predict(X)

    df_offload.loc[(df_offload.municipio == mun) &
                        (df_offload.ano_mes > end_date),
                        'percent_nr'] = y_hat.flatten()
    
df_offload.loc[df_offload.percent_nr < 0, 'percent_nr'] = 0
df_offload['percent_lte'] = df_offload.apply(
    lambda row: 1 - row['percent_nr'] if pd.isnull(row['percent_lte']) else
    row['percent_lte'], axis=1)
# %%
for mun in tqdm(lst_mun_2_model):
    ind_mun = df_offload.loc[(df_offload.municipio
                               == mun) &
                             (df_offload.ano_mes > end_date) &
                             (df_offload.ano_mes <= horizon_dt)].index
    vol_norm = np.array(df_offload.loc[ind_mun, 'volume_total']).reshape(-1,1)

    df_offload.loc[ind_mun, ['percent_lte','percent_nr']] = df_offload.loc[ind_mun, ['percent_lte','percent_nr']]
    df_offload.loc[ind_mun, 'volume_total'] = scalers['volume'].inverse_transform(vol_norm).reshape(1,-1)[0]

# %%
df_offload = df_offload.loc[df_offload.ano_mes > '2025-06-01']

df_offload['VOL_LTE'] = df_offload.volume_total * df_offload.percent_lte
df_offload['VOLL_NR'] = df_offload.volume_total * df_offload.percent_nr

filename = f'prophet_mun_offload_percent.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(df_offload, arquivo)

# %%
