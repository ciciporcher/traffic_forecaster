# %% import libraries
import pandas as pd
import numpy as np
from datetime import datetime

import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import shap

import fcapacity as cap

# %%
start_date = '2022-06-01'
end_date = f'2025-06-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
yymm_list = [d.strftime('%y%m') for d in date_range]

path = fr'caminho_para_os_dados'

df_vol = cap.import_vol_offload(yymm_list, path)
df_vol = cap.groupbyIBGE_offload(df_vol, path)
df_vol['volume_total'] = df_vol.VOL_LTE + df_vol.VOL_NR
df_vol['percent_lte'] = df_vol.VOLL_LTE / df_vol.volume_total
df_vol['percent_nr'] = df_vol.VOL_NR / df_vol.volume_total
df_vol = df_vol.reset_index()
df_vol['DATA'] = pd.to_datetime(df_vol.DATA)
# %%
df = pd.DataFrame(columns=['ibge','ano_mes','meses_5G','sites_lte','sites_nr'])

with open(f'sites.pkl', 'rb') as f:
        sites = pickle.load(f)
        
for mun in tqdm(list(sites.groups.keys())):
    df_mun = sites.get_group(mun)
    data_ativacao_nr = df_mun.data_ativacao_nr.min()
    sites4G = [len(df_mun.loc[(df_mun.data_ativacao_lte <= pd.to_datetime(date)) &
                              (df_mun.LTE == 1)]) for date in date_range]
    sites5G = [len(df_mun.loc[(df_mun.data_ativacao_nr <= pd.to_datetime(date)) &
                              (df_mun.NR == 1)]) for date in date_range]    
    df_temp = pd.DataFrame({
        'ibge': mun,
        'ano_mes': date_range,
        'sites_lte': sites4G,
        'sites_nr': sites5G})
    if pd.isnull(data_ativacao_nr):
        df_temp['meses_5G'] = np.nan
    else:
        df_temp['meses_5G'] = [(d.year - data_ativacao_nr.year)*12 + (d.month - data_ativacao_nr.month) for d in df_temp['ano_mes']]
    df = pd.concat([df, df_temp], ignore_index=True)

df = df.merge(df_vol, how='left',
              left_on=['ibge','ano_mes'], right_on=['IBGE','DATA'])[
                  [*df.columns, *['volume_total','percent_lte','percent_nr']]]

df = df.sort_values(by=['ibge', 'ano_mes'])

df['sites_lte_diff'] = df.groupby('ibge')['sites_lte'].diff().fillna(0)
df['sites_nr_diff'] = df.groupby('ibge')['sites_nr'].diff().fillna(0)
df['volume_total_diff'] = df.groupby('ibge')['volume_total'].diff().fillna(0)

df[['sites_lte_diff', 'sites_nr_diff']] = df[['sites_lte_diff', 'sites_nr_diff']].astype(int)

# %% LSTM
df = df.dropna(subset='meses_5G') # drop mun sem 5G

lst_mun = df.ibge.unique()
ibge_train_val, ibge_test = train_test_split(lst_mun, test_size=0.2, random_state=42)
ibge_train, ibge_val = train_test_split(ibge_train_val, test_size=0.15/0.80, random_state=42) 

df_train = df.loc[df.ibge.isin(ibge_train)].copy()
df_val = df.loc[df.ibge.isin(ibge_val)].copy()
df_test = df.loc[df.ibge.isin(ibge_test)].copy()

train,valid,test,scalers = cap.minmax_offload(df_train, df_val, df_test)
filename = f'scalers_offload.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(scalers, arquivo)

window_size = 6
X_treino, y_treino, _ = cap.dataset_offload(train, window_size)
X_valid, y_valid, _ = cap.dataset_offload(valid, window_size)
X_teste, y_teste, ind_teste = cap.dataset_offload(test, window_size)
num_features = X_treino.shape[2]

print(f'numero de features: {num_features}')

# Define inputs using the Keras Functional API
inputs = tf.keras.layers.Input(shape=(window_size, num_features))

# LSTM layers
x = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True, name='lstm_1')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
lstm_out = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, name='lstm_2')(x)
lstm_out = tf.keras.layers.Dropout(0.3)(lstm_out)

attention_layer = tf.keras.layers.Attention(name='attention_layer')
context_vector, attention_scores = attention_layer(
    [lstm_out, lstm_out], return_attention_scores=True
)

# Continue com o context_vector
x = tf.keras.layers.GlobalAveragePooling1D()(context_vector)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)

# This is your original model for training and prediction
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='prediction_model')

# Create a new model specifically for visualizing the attention scores
visualization_model = tf.keras.Model(inputs=inputs, outputs=attention_scores, name='visualization_model')

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

model.summary()
print("\n--- Visualization Model ---")
visualization_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_treino, y_treino, 
                    epochs=150, 
                    batch_size=64, 
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping])

result = model.evaluate(X_teste, y_teste, verbose=0)
y_hat_teste = model.predict(X_teste)
y_hat_teste = y_hat_teste.flatten()

# %% SHAP
background_indices = np.random.choice(X_treino.shape[0], 10000, replace=False)
background_data_np = X_treino[background_indices]
amostra_teste_np = X_teste[:10000]

explainer = shap.GradientExplainer(model, background_data_np)
shap_values = explainer.shap_values(amostra_teste_np)
shap_values = np.squeeze(shap_values, axis=-1)

shap_reshaped = shap_values.reshape(-1, shap_values.shape[2])
amostra_reshaped = amostra_teste_np.reshape(-1, amostra_teste_np.shape[2])
shap.summary_plot(shap_reshaped, amostra_reshaped,
                  feature_names=['meses_5G','sites_lte','sites_lte_diff','sites_nr','sites_nr_diff','volume_total','volume_total_diff'])

# %% 
df['divisao'] = ['train' if mun in ibge_train else 'test' for mun in df.ibge]
df['percent_nr_hat'] = np.nan
df.loc[ind_teste, 'percent_nr_hat'] = y_hat_teste
df = df.dropna(subset=['volume_total','percent_nr_hat'])
df = df.loc[df.sites_lte > 0]
df.loc[(df.sites_nr == 0) & (df.meses_5G < 0), 'percent_nr_hat'] = 0

with open('df_offload_test.pkl', 'wb') as arquivo:
    pickle.dump(df, arquivo)

# %%
valid_mask = (~np.isnan(y_teste)) & (~np.isnan(y_hat_teste))
y_test_clean = y_teste[valid_mask]
y_hat_test_clean = y_hat_teste[valid_mask]
municipios_clean = test.ibge[ind_teste][valid_mask]

# Calcula as métricas de erro
mse_teste = mean_squared_error(y_test_clean, y_hat_test_clean)
mae_teste = mean_absolute_error(y_test_clean, y_hat_test_clean)
mape_teste = mean_absolute_percentage_error(y_test_clean, y_hat_test_clean)
r2_teste = r2_score(y_test_clean, y_hat_test_clean)

# Exibe os resultados formatados
print("\n--- Métricas de Performance no Conjunto de Teste ---")
print(f"MSE (Erro Quadrático Médio): {mse_teste:.6f}")
print(f"MAE (Erro Absoluto Médio):   {mae_teste:.6f}")
print(f"MAPE (Erro Percentual Absoluto Médio): {mape_teste:.2%}")
print(f"R² (Coeficiente de Determinação):      {r2_teste:.2%}")

filename = f'model_lstm_offload.pkl'
with open(filename, 'wb') as arquivo:
    pickle.dump(model, arquivo)