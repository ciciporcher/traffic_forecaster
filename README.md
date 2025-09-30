# Predição de Tráfego de Redes Móveis e Análise de Offload 5G

## Sobre o Projeto

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) do MBA em Data Science & Analytics da USP/Esalq.

O objetivo principal do projeto é prever a evolução do tráfego de dados em redes móveis (4G/LTE e 5G/NR) e analisar o fenômeno de *offload*, que consiste na migração do tráfego da rede 4G para a 5G.

Para isso, foram utilizadas duas abordagens de modelagem:
1.  **Prophet:** Para a previsão de séries temporais do volume total de tráfego em cada município.
2.  **LSTM:** Para prever o percentual de tráfego que será cursado na rede 5G (percentual de offload).

**Nota sobre a Confidencialidade dos Dados**:
Os dados originais utilizados neste estudo são de natureza confidencial, não podendo ser compartilhados publicamente. Como resultado, os datasets não estão incluídos neste repositório. Partes do código, especialmente nos scripts de importação e manipulação (`src/fcapacity.py`), foram adaptadas para remover caminhos de arquivos, credenciais e outras informações sensíveis, garantindo a total anonimização do processo.

## Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e clareza:

```
/
├── src/
│   ├── fcapacity.py            # Módulo com funções de importação e processamento de dados.
│   ├── prophet_site.py         # Script para treinamento do modelo Prophet de previsão de volume.
│   ├── lstm_offload.py         # Script para treinamento do modelo LSTM de previsão do percentual de offload.
│   └── prophet_site_offload.py # Script final que combina as previsões para gerar o resultado de offload.
│
├── .gitignore                  # Arquivo para ignorar arquivos de dados e outros.
└── README.md                   # Este arquivo.
```

## Autor

**Cíntia Porcher de Oliveira**
* **LinkedIn:** https://linkedin.com/in/cintiaporcher/
