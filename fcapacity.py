import pandas as pd
import numpy as np
import math
from datetime import datetime

import os
from tqdm import tqdm
import pickle

import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

def importKPI(date, tec, path, nivel):
    """
    Importa os csv com dados horários dos indicadores, transforma em dataframe,
    calcula o valor dos kpis na 2hmm do dia e agrupa esses valores por mês definindo
    a mediana. Além disso, transforma o volume em bytes.

    Parameters:
        date (list): lista de datas dos csv a serem importados.
        tec (str): tecnologia dos indicadores.
        dict_kpi (dict): dicionário com nome das métricas.
        path (str): caminho para importar csv.
        nivel (str): granularidade dos dados

    Returns:
        df_month: dataframe com kpis agrupados por mês.
    """    
    df = pd.read_csv(path + fr'\{date[0]}_{nivel}_{tec}.csv', sep=';', encoding='latin-1')
    for i in tqdm(range(1, len(date))):
        df_temp = pd.read_csv(path+ fr'\{date[i]}_{nivel}_{tec}.csv', sep=';', encoding='latin-1')
        df = pd.concat([df, df_temp], ignore_index=True)
    
    # anonimizar eixo tempo
    df['DATA'] = pd.to_datetime(df['DATA'])

    if nivel == 'nacional':
        # cria colunas para data
        df.DATA = pd.to_datetime(df.DATA)
        df.insert(1,'year',df.DATA.dt.year)
        df.insert(2,'month',df.DATA.dt.month)
        df.insert(3,'day',df.DATA.dt.day)
        df.insert(4,'hour',df.DATA.dt.hour)
    else:
        df.insert(1,'year',df.DATA.dt.year)
        df.insert(2,'month',df.DATA.dt.month)

    vol_direto = f'VOL_DL'
    vol_reverso = f'VOL_UL'

    nan_ind = df[np.isnan(df[vol_direto])].index
    df.loc[nan_ind, vol_direto] = 0

    df_month = df
    df_month[vol_direto] = df_month[vol_direto]*1e6 # volume em MB, passa para B
    df_month[vol_reverso] = df_month[vol_reverso]*1e6
    
    # soma volume ul e dl
    df_month[f'VOL_{tec.upper()}'] = df_month[vol_direto] + df_month[vol_reverso]
    df_month = df_month.drop(columns=[vol_direto, vol_reverso])

    return df_month

def import_vol_offload(dates, path):
    df5 = pd.read_csv(path + fr'\{dates[0]}_site_nr.csv', sep=';', encoding='latin-1')
    df4 = pd.read_csv(path + fr'\{dates[0]}_site_lte.csv', sep=';', encoding='latin-1')
    for i in tqdm(range(1, len(dates))):
        df5_temp = pd.read_csv(path+ fr'\{dates[i]}_site_nr.csv', sep=';', encoding='latin-1')
        df4_temp = pd.read_csv(path+ fr'\{dates[i]}_site_lte.csv', sep=';', encoding='latin-1')
       
        df5 = pd.concat([df5, df5_temp], ignore_index=True)
        df4 = pd.concat([df4, df4_temp], ignore_index=True)

    df5.insert(1,'year',df5.DATA.str[:4])
    df5.insert(2,'month',df5.DATA.str[-2:])
    df4.insert(1,'year',df4.DATA.str[:4])
    df4.insert(2,'month',df4.DATA.str[-2:])

    nan_ind = df5[np.isnan(df5['VOL_DL_NR'])].index
    df5.loc[nan_ind, 'VOL_DL_NR'] = 0
    nan_ind = df4[np.isnan(df4['VOL_DL_LTE'])].index
    df4.loc[nan_ind, 'VOL_DL_LTE'] = 0

    df5['VOL_NR'] = (df5['VOL_DL_NR'] + df5['VOL_UL_NR'])*1e6
    df4['VOL_LTE'] = (df4['VOL_DL_LTE'] + df4['VOL_UL_LTE'])*1e6
    
    df5 = df5[['DATA', 'year', 'month', 'SITE', 'VOL_NR']]
    df4 = df4[['DATA', 'year', 'month', 'SITE', 'VOL_LTE']]

    df = df4.merge(df5, on=['DATA', 'year', 'month', 'SITE'], how='left')

    return df

def groupbyIBGE_offload(df, path):
    with open(f'sites.pkl', 'rb') as f:
        sites = pickle.load(f)
    df_ibge = sites[['UF','CN','SITE','MUNICIPIO','IBGE']]

    df = df.merge(df_ibge, on='SITE', how='left')
    
    current_cols = df.columns.tolist()
    new_cols = [col for col in current_cols if col not in ['UF','CN','MUNICIPIO','IBGE']]
    new_cols.insert(3,'IBGE')
    new_cols.insert(4,'UF')
    new_cols.insert(5,'CN')
    new_cols.insert(6,'MUNICIPIO')
    
    df = df[new_cols]

    df = df.dropna(subset='IBGE')

    df_mun = df.groupby(['IBGE','DATA','year','month'])[
        ['VOL_LTE','VOL_NR']].sum()
    
    return df_mun
    
def groupbyIBGE(df, tec, tec_secundaria, exogenous_var):
    with open(f'sites.pkl', 'rb') as f:
        sites = pickle.load(f)
    df_ibge = sites[['UF','CN','SITE','MUNICIPIO','IBGE']]
    
    df = df.merge(df_ibge, on='SITE', how='left')
    
    current_cols = df.columns.tolist()
    new_cols = [col for col in current_cols if col not in ['UF','CN','MUNICIPIO','IBGE']]
    new_cols.insert(3,'IBGE')
    new_cols.insert(4,'UF')
    new_cols.insert(5,'CN')
    new_cols.insert(6,'MUNICIPIO')
    
    df = df[new_cols]

    df = df.dropna(subset='IBGE')

    agg_dict = {col: 'mean' for col in exogenous_var if col != f'VOL_{tec_secundaria.upper()}'}
    if f'VOL_{tec_secundaria.upper()}' in exogenous_var:
        agg_dict[f'VOL_{tec_secundaria.upper()}'] = 'sum'
    agg_dict[f'VOL_{tec.upper()}'] = 'sum'

    df_mun = df.groupby(['IBGE','DATA','year','month']).agg(agg_dict).reset_index()

    return df_mun  

def prophet_exog(df, tec, exogenous_var):
    if tec == 'lte': # verifica se possui 5G
        vol_nr = df.VOL_NR.sum()
        if vol_nr == 0: # se não possui 5G no município
            df = df.drop(columns='VOL_NR')
            exog_var = [x for x in exogenous_var if x != 'VOL_NR']
        else:
            exog_var = exogenous_var
    else:
        exog_var = exogenous_var
    
    return exog_var

def prophet_df_mun(df, exog_var, target_var, include_exog, start_date):    
    prophet_df = pd.DataFrame({
            'ds': pd.date_range(start=start_date,
                                periods=len(df),
                                freq='MS'),
            'y': df[target_var].values
        })
    if include_exog:
        list_ds = [('ds', pd.date_range(start=start_date, periods=len(df), freq='MS'))]
        for var in exog_var:
            scaler = MinMaxScaler()
            var_norm = scaler.fit_transform(df[[var]]).reshape(1,-1)[0]
            list_ds.extend([(var, var_norm)])
        exog_df = pd.DataFrame(dict(list_ds))

        prophet_df = pd.merge(prophet_df, exog_df, on='ds')
        ind_nonnull = prophet_df.loc[prophet_df.y > 0].index[0]
        prophet_df = prophet_df.loc[prophet_df.index >= ind_nonnull].reset_index(drop=True)

    # substitui valores nulos ou iguais a zero
    for col in [col for col in prophet_df.columns if col != 'ds']:
        zero_i = prophet_df.loc[prophet_df[col] == 0].index
        prophet_df.loc[zero_i, col] = np.nan
        prophet_df[col] = prophet_df[col].bfill()

        zero_i = prophet_df.loc[prophet_df[col] == 0].index
        prophet_df.loc[zero_i, col] = np.nan
        prophet_df[col] = prophet_df[col].ffill()

    return prophet_df

def prophet_test_mun(prophet_df, horizon, exog_var, include_exog, include_season):
    # verifica quantidade de dados válidos disponível
    h = math.floor(len(prophet_df)*0.3)
    if h < horizon:
        horizon = h
    # divisão treino e teste
    train_prophet_df = prophet_df.iloc[:-horizon].copy()
    test_prophet_df = prophet_df.iloc[-horizon:].copy()

    if include_season and len(train_prophet_df) > 24:
        model_prophet_eval = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='additive')
    else:
        model_prophet_eval = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode='additive')

    if include_exog:
        for exog_col in exog_var:
            model_prophet_eval.add_regressor(exog_col, mode='additive')
    
    model_prophet_eval.changepoint_prior_scale = 0.1
    model_prophet_eval.fit(train_prophet_df)

    future_dates_all = model_prophet_eval.make_future_dataframe(
        periods=horizon,
        freq='MS',
        include_history=True
    )
    if include_exog:
        for exog_col in exog_var:
            exog_hat = prophet_exog_individual(prophet_df, horizon, exog_col, True, False, include_season)
            exog_train = train_prophet_df[exog_col].values
            future_dates_all[exog_col] = np.concatenate((exog_train , exog_hat))

    # filtrar apenas o período de teste
    future_dates_eval = future_dates_all.iloc[-horizon:]
    forecast_eval = model_prophet_eval.predict(future_dates_eval)
    predictions_eval = forecast_eval['yhat'].values

    # avaliação do desempenho
    mae_prophet = mean_absolute_error(test_prophet_df.y, predictions_eval)
    mape_prophet = mean_absolute_percentage_error(test_prophet_df.y, predictions_eval)
    r2_prophet = r2_score(test_prophet_df.y, predictions_eval)

    return horizon, mae_prophet, r2_prophet

def prophet_future_mun(prophet_df, horizon, exog_var, include_exog, include_season):
    if include_season and len(prophet_df) > 24:
        model_prophet_full = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='additive'
        )
    else:
        model_prophet_full = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode='additive')

    if include_exog:
        for exog_col in exog_var:
            model_prophet_full.add_regressor(exog_col, mode='additive')
    model_prophet_full.changepoint_prior_scale = 0.1
    model_prophet_full.fit(prophet_df)

    # gerar as datas futuras com o make_future_dataframe a partir do df completo
    future_dates = model_prophet_full.make_future_dataframe(
        periods=horizon,
        freq='MS',
        include_history=False)
    
    if include_exog:
        for exog_col in exog_var:
            exog_hat = prophet_exog_individual(prophet_df, horizon, exog_col, False, False, include_season)
            future_dates[exog_col] = exog_hat

    forecast_full = model_prophet_full.predict(future_dates.iloc[-horizon:])
    future_predictions_prophet = forecast_full['yhat']

    return future_predictions_prophet, model_prophet_full

def prophet_exog_individual(df_exog, horizon, exog, test, detail_exog, include_season):
    regressor_df = pd.DataFrame({
        'ds': df_exog['ds'],
        'y': df_exog[exog]})

    if include_season and len(regressor_df) > 24 and test == False:
        model_exog = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='additive'
        )
    else:
        model_exog = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode='additive'
        )
    
    #model_exog.add_seasonality(name='exog_monthly', period=12, fourier_order=3)
    model_exog.changepoint_prior_scale = 0.1
    if test: 
        train_regressor_df = regressor_df.iloc[:-horizon].copy()
        test_regressor_df = regressor_df.iloc[-horizon:].copy()
        model_exog.fit(train_regressor_df)
    else:
        model_exog.fit(regressor_df)

    # previsão do regressor
    future_dates_for_exog = model_exog.make_future_dataframe(
            periods=horizon,
            freq='MS',
            include_history=False)
    forecast_exog = model_exog.predict(future_dates_for_exog)
    predicted_exog_values = forecast_exog['yhat']

    #print(forecast_exog[['ds','yhat']].head())
    #print(forecast_exog[['ds','yhat']].tail())
    
    if detail_exog:
        if test:
            # avaliação do desempenho
            mape_prophet = mean_absolute_percentage_error(test_regressor_df.y,
                                                          predicted_exog_values)
            mae_prophet = mean_absolute_error(test_regressor_df.y, predicted_exog_values)
            print(f'MAPE ({exog}): {mape_prophet*100:.2f}%')
            print(f'MAE ({exog}): {mae_prophet}')

        fig = go.Figure()
        # dados reais
        fig.add_trace(go.Scatter(x=regressor_df['ds'],
                                 y=regressor_df['y'],
                                 mode='lines',
                                 name='Dados reais',
                                 line=dict(color='blue')))
        # previsão
        fig.add_trace(go.Scatter(x=forecast_exog['ds'],
                                 y=predicted_exog_values,
                                 mode='lines',
                                 name='Previsão',
                                 line=dict(color='green', dash='dash')))
        fig.update_layout(
            title=f'Avaliação do modelo para regressor - {exog}',
            xaxis_title='Data',
            yaxis_title=exog,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        fig.show() 
    # Armazenar as previsões do regressor
    return predicted_exog_values.values

def prophet_test_nr_seasonal(prophet_df, horizon, df_lte, exog_var):
    df_lte = df_lte.copy()

    # verifica quantidade de dados válidos disponível
    h = math.floor(len(prophet_df)*0.3)
    if h < horizon:
        horizon = h
    
    # add vol lte sem normalização
    prophet_df = pd.merge(prophet_df, df_lte, on='ds')
    df_lte.columns = ['ds','y']

    # divisão treino e teste
    train_prophet_df = prophet_df.iloc[:-horizon].copy()
    test_prophet_df = prophet_df.iloc[-horizon:].copy()

    # normalização com fit no treino
    scaler = MinMaxScaler()
    train_prophet_df['VOL_LTE'] = scaler.fit_transform(
        train_prophet_df[['VOL_LTE']]).reshape(1,-1)[0]
    test_prophet_df['VOL_LTE'] = scaler.transform(
        test_prophet_df[['VOL_LTE']]).reshape(1,-1)[0]

    model_prophet_eval = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode='additive')
    
    if len(exog_var) > 0:
        for exog_col in exog_var:
            model_prophet_eval.add_regressor(exog_col, mode='additive')
    model_prophet_eval.add_regressor('VOL_LTE', mode='additive')
    model_prophet_eval.changepoint_prior_scale = 0.1
    model_prophet_eval.fit(train_prophet_df)

    future_dates_all = model_prophet_eval.make_future_dataframe(
        periods=horizon,
        freq='MS',
        include_history=True)

    model_lte = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='additive')
    model_lte.changepoint_prior_scale = 0.1
    model_lte.fit(df_lte.iloc[:-horizon])
    # previsão do regressor
    future_dates_for_lte = model_lte.make_future_dataframe(
            periods=horizon,
            freq='MS',
            include_history=False)
    forecast_lte = model_lte.predict(future_dates_for_lte)
    vol_lte_hat = forecast_lte['yhat'].values

    if len(exog_var) > 0:
        for exog_col in exog_var:
            exog_hat = prophet_exog_individual(prophet_df, horizon, exog_col, True, False, False)
            exog_train = train_prophet_df[exog_col].values
            future_dates_all[exog_col] = np.concatenate((exog_train , exog_hat))
    future_dates_all['VOL_LTE'] = np.concatenate(
        (df_lte.y.iloc[-len(prophet_df):-horizon].values,
         vol_lte_hat))

    future_dates_all['VOL_LTE'] = scaler.transform(
        future_dates_all[['VOL_LTE']]).reshape(1,-1)[0]
    # filtrar apenas o período de teste
    future_dates_eval = future_dates_all.iloc[-horizon:]
    forecast_eval = model_prophet_eval.predict(future_dates_eval)
    predictions_eval = forecast_eval['yhat'].values

    # avaliação do desempenho
    mape_prophet = mean_absolute_percentage_error(test_prophet_df.y, predictions_eval)
    mae_prophet = mean_absolute_error(test_prophet_df.y, predictions_eval)
    r2_prophet = r2_score(test_prophet_df.y, predictions_eval)

    return horizon, mae_prophet, r2_prophet

def prophet_future_nr_seasonal(prophet_df, horizon, df_lte, exog_var):
    df_lte = df_lte.copy()

    prophet_df = pd.merge(prophet_df, df_lte, on='ds')
    scaler = MinMaxScaler()
    prophet_df['VOL_LTE'] = scaler.fit_transform(
        prophet_df[['VOL_LTE']]).reshape(1,-1)[0]
    df_lte.columns = ['ds','y']
    
    model_prophet_full = Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode='additive')
    if len(exog_var) > 0:
        for exog_col in exog_var:
            model_prophet_full.add_regressor(exog_col, mode='additive')
    model_prophet_full.add_regressor('VOL_LTE', mode='additive')
    model_prophet_full.changepoint_prior_scale = 0.1
    model_prophet_full.fit(prophet_df)

    # gerar as datas futuras com o make_future_dataframe a partir do df completo
    future_dates = model_prophet_full.make_future_dataframe(
        periods=horizon,
        freq='MS',
        include_history=False)

    model_lte = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode='additive')
    model_lte.changepoint_prior_scale = 0.1
    model_lte.fit(df_lte)
    future_dates_for_lte = model_lte.make_future_dataframe(
            periods=horizon,
            freq='MS',
            include_history=False)
    forecast_lte = model_lte.predict(future_dates_for_lte)
    vol_lte_hat = forecast_lte['yhat'].values  

    if len(exog_var) > 0:
        for exog_col in exog_var:
            exog_hat = prophet_exog_individual(prophet_df, horizon, exog_col, False, False, False)
            future_dates[exog_col] = exog_hat
    future_dates['VOL_LTE'] = vol_lte_hat
    future_dates['VOL_LTE']= scaler.transform(
        future_dates[['VOL_LTE']]).reshape(1,-1)[0]

    forecast_full = model_prophet_full.predict(future_dates.iloc[-horizon:])
    future_predictions_prophet = forecast_full['yhat']

    return future_predictions_prophet, model_prophet_full

def minmax_offload(train, valid, test, volume_scaler=None):
    """
    Normaliza os dataframes de treino, validação e teste para todas as features.
    O scaler é ajustado (fit) APENAS no conjunto de treino.
    """
    scalers = {}

    scaler_meses = MinMaxScaler()
    scaler_meses.fit(train[['meses_5G']])
    train['meses_5G'] = scaler_meses.transform(train[['meses_5G']])
    valid['meses_5G'] = scaler_meses.transform(valid[['meses_5G']])
    test['meses_5G'] = scaler_meses.transform(test[['meses_5G']])
    scalers['meses_5G'] = scaler_meses

    scaler_sites = MinMaxScaler()
    combined_sites_train = np.concatenate([train['sites_lte'].values, train['sites_nr'].values]).reshape(-1, 1)
    scaler_sites.fit(combined_sites_train)
    
    for df in [train, valid, test]:
        df['sites_lte'] = scaler_sites.transform(df[['sites_lte']])
        df['sites_nr'] = scaler_sites.transform(df[['sites_nr']])
    scalers['sites'] = scaler_sites

    scaler_sites_diff = MinMaxScaler()
    combined_sites_diff_train = np.concatenate([train['sites_lte_diff'].values, train['sites_nr_diff'].values]).reshape(-1, 1)
    scaler_sites_diff.fit(combined_sites_diff_train)

    for df in [train, valid, test]:
        df['sites_lte_diff'] = scaler_sites_diff.transform(df[['sites_lte_diff']])
        df['sites_nr_diff'] = scaler_sites_diff.transform(df[['sites_nr_diff']])
    scalers['sites_diff'] = scaler_sites_diff

    if volume_scaler is None:
        scaler_volume = MinMaxScaler()
        scaler_volume.fit(train[['volume_total']])
    else:
        scaler_volume = volume_scaler

    train['volume_total'] = scaler_volume.transform(train[['volume_total']])
    valid['volume_total'] = scaler_volume.transform(valid[['volume_total']])
    test['volume_total'] = scaler_volume.transform(test[['volume_total']])
    scalers['volume'] = scaler_volume

    scaler_volume_diff = MinMaxScaler()
    scaler_volume_diff.fit(train[['volume_total_diff']])
    train['volume_total_diff'] = scaler_volume_diff.transform(train[['volume_total_diff']])
    valid['volume_total_diff'] = scaler_volume_diff.transform(valid[['volume_total_diff']])
    test['volume_total_diff'] = scaler_volume_diff.transform(test[['volume_total_diff']])
    scalers['volume_diff'] = scaler_volume_diff

    return train, valid, test, scalers

def dataset_offload(df, window_size):
    X, y, indices = [], [], []
    features = ['meses_5G','sites_lte','sites_lte_diff','sites_nr','sites_nr_diff','volume_total','volume_total_diff']

    for grp_id, grp_df in df.groupby('ibge'):
        # encontrar a primeira linha onde todos os dados estão presentes
        first_valid = grp_df[features + ['percent_nr']].dropna().first_valid_index()
        if first_valid is not None:
            # filtrar a partir da primeira linha válida
            grp_df = grp_df.loc[first_valid:]
            
        dados_features = grp_df[features].values
        dados_alvo =  grp_df['percent_nr'].values

        # garantir que município tenha dados suficientes para pelo menos uma sequência
        if len(grp_df) > window_size:
            for i in range(len(grp_df) - window_size):
                sequencia_x = dados_features[i:(i+window_size)]
                X.append(sequencia_x)

                alvo_y = dados_alvo[i+window_size]
                y.append(alvo_y)

                ind = grp_df.index[i+window_size]
                indices.append(ind)

    return np.array(X), np.array(y), np.array(indices)