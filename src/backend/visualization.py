import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt
import pandas as pd

def plot_annual_trend(
    df,
    column: str,
    year: int | tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "b",
    marker: str = "o"
):
    """
    Plota uma série temporal para um ano específico.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com índice do tipo datetime.
    - column (str): Nome da coluna com os dados numéricos a serem plotados.
    - year (int | tuple[int, int]): Ano único (ex: 2022) ou intervalo de anos (ex: (2020, 2023)).
    - title (str): Título do gráfico.
    - xlabel (str): Rótulo do eixo X.
    - ylabel (str): Rótulo do eixo Y.
    - color (str): Cor da linha.
    - marker (str): Estilo do marcador.
    """

    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame.")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("O índice do DataFrame deve ser do tipo datetime.")
    
    if isinstance(year, int):
        df_filtered = df.loc[str(year)]
    elif isinstance(year, tuple) and len(year) == 2:
        start_year, end_year = year
        df_filtered = df.loc[str(start_year):str(end_year)]
    else:
        raise ValueError("O parâmetro 'year' deve ser um inteiro ou uma tupla (ano_inicial, ano_final).")
    
    # Inicia o gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(df_filtered.index, df_filtered[column], marker=marker, linestyle='-', color=color)

    # Título adaptável
    period_str = f"{year}" if isinstance(year, int) else f"{year[0]}–{year[1]}"
    plt.title(f"{title} - {period_str}", fontsize=14)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adiciona valores nos pontos (somente se for um único ano)
    if isinstance(year, int):
        for date, value in zip(df_filtered.index, df_filtered[column]):
            plt.text(date, value + (df_filtered[column].max() - df_filtered[column].min()) * 0.02, f"{value:.1f}", ha='center', fontsize=9)
        # Meses no eixo X
        plt.xticks(df_filtered.index, df_filtered.index.strftime('%b'), rotation=0)
    else:
        # Eixo X padrão (deixa automático para múltiplos anos)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def plot_full_time_series(
    df,
    title: str,
    xlabel: str,
    ylabel: str
):
    """
    Plota todas as colunas numéricas do DataFrame ao longo do tempo completo.

    Parâmetros:
    - df (DataFrame): DataFrame com índice datetime e colunas numéricas.
    - title (str): Título do gráfico.
    - xlabel (str): Rótulo do eixo X.
    - ylabel (str): Rótulo do eixo Y.
    """
    plt.figure(figsize=(12, 6))
    
    # Plota todas as colunas numéricas
    for col in df.select_dtypes(include='number').columns:
        plt.plot(df.index, df[col], label=col)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostra legenda apenas se houver mais de uma série
    if df.select_dtypes(include='number').shape[1] > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()



def plot_acf_pacf(
    series: pd.Series,
    lags: int = 20,
    title: str = "ACF e PACF",
    alpha: float = 0.05
):
    """
    Plota os gráficos de ACF e PACF de uma série temporal.

    Parâmetros:
    - series (pd.Series): Série temporal a ser analisada.
    - lags (int): Número de defasagens a serem exibidas (default=20).
    - title (str): Título do gráfico conjunto.
    - alpha (float): Nível de significância para intervalo de confiança.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("A entrada 'series' deve ser do tipo pandas.Series.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    smt.graphics.plot_acf(series, lags=lags, alpha=alpha, ax=axes[0])
    axes[0].set_title("Autocorrelação (ACF)")

    smt.graphics.plot_pacf(series, lags=lags, alpha=alpha, ax=axes[1], method='ywm')
    axes[1].set_title("Autocorrelação Parcial (PACF)")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Ajusta o título principal
    plt.show()


def plot_train_test_split(
    serie_original,
    treino,
    teste,
    validacao=None,
    title_prefix="Série"
):
    """
    Plota a série original com indicação dos conjuntos de treino, validação e teste.

    Parâmetros:
    - serie_original (pd.Series): Série completa
    - treino (pd.Series): Conjunto de treino
    - teste (pd.Series): Conjunto de teste
    - validacao (pd.Series, opcional): Conjunto de validação (se houver)
    - title_prefix (str): Prefixo do título dos gráficos
    """
    # Eixo X: datas ou posições
    idx = serie_original.index
    idx_treino = treino.index
    idx_validacao = validacao.index if validacao is not None else None
    idx_teste = teste.index

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 🔹 Parte 1: Série completa
    axes[0].plot(idx, serie_original, color="blue", label="Série Completa")
    axes[0].set_title(f"{title_prefix} - Série Completa")
    axes[0].legend()
    axes[0].grid(True)

    # 🔹 Parte 2: Com separações
    axes[1].plot(idx_treino, treino, label="Treino", color="green")
    if validacao is not None:
        axes[1].plot(idx_validacao, validacao, label="Validação", color="orange")
    axes[1].plot(idx_teste, teste, label="Teste", color="red")

    # Linhas verticais de corte
    if validacao is not None:
        axes[1].axvline(idx_validacao[0], color="black", linestyle="--", label="Início Validação")
        axes[1].axvline(idx_teste[0], color="gray", linestyle="--", label="Início Teste")
    else:
        axes[1].axvline(idx_teste[0], color="black", linestyle="--", label="Corte treino/teste")

    axes[1].set_title(f"{title_prefix} - Divisão Treino/Validação/Teste")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
