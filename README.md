# Projeto de Análise de Séries Temporais

## Visão Geral

Este projeto tem como objetivo estudar, analisar e modelar séries temporais utilizando **Python 3.11.7**.

Serão aplicados métodos estatísticos e de machine learning para explorar padrões, identificar estacionariedade, realizar previsões e avaliar a qualidade dos modelos.

**Datasets**: 
- Industrial Production: Utilities: Electric and Gas Utilities (https://fred.stlouisfed.org/series/IPG2211A2N)

# Ambiente de Desenvolvimento

O projeto utiliza `pyenv` e `pyenv-virtualenv` para gerenciamento de versões do Python.

## Criação do ambiente

```bash
## Cria o ambiente virtual com Python 3.11.7
pyenv virtualenv 3.11.7 projeto_analise_series_temporais
```

```bash
## Ativa o ambiente para o diretório do projeto
pyenv local projeto_analise_series_temporais
```
## Instalação das dependências

```bash
pip install -r requirements.txt
```

# Estrutura do Projeto

```
projeto_analise_series_temporais/
│── data/                # Conjunto de dados brutos e processados
│── notebooks/           # Notebooks exploratórios
│── src/                 # Código-fonte (pré-processamento, modelos, avaliação)
│── requirements.txt     # Lista de dependências
│── README.md            # Documentação do projeto
```