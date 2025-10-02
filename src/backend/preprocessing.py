from pmdarima.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_series_treino_teste(
    serie,
    percentual_treino: float = 0.8,
    percentual_validacao: float = 0.0,
    validacao: bool = False,
    verbose: bool = True
):
    """
    Divide uma série temporal em treino/teste ou treino/validação/teste.

    Parâmetros:
    - serie (pd.Series ou DataFrame com uma coluna e índice datetime)
    - percentual_treino: proporção de dados para treino
    - percentual_validacao: proporção de dados para validação (se validacao=True)
    - validacao: se True, retorna também conjunto de validação
    - verbose: se True, imprime o resumo da divisão

    Retorna:
    - treino, teste (se validacao=False)
    - treino, validacao, teste (se validacao=True)
    """
    # Verificações iniciais
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError("O índice da série deve ser pd.DatetimeIndex.")
    if isinstance(serie, pd.DataFrame):
        if serie.shape[1] != 1:
            raise ValueError("DataFrame deve conter apenas uma coluna.")
        serie = serie.squeeze()

    n_total = len(serie)

    if validacao:
        if not (0 < percentual_validacao < 1):
            raise ValueError("percentual_validacao deve estar entre 0 e 1.")
        if percentual_treino + percentual_validacao >= 1:
            raise ValueError("Soma de treino + validação deve ser < 1.")

        # Treino vs Resto
        serie_treino, serie_restante = train_test_split(serie, train_size=percentual_treino)
        # Validação vs Teste
        prop_val = percentual_validacao / (1 - percentual_treino)
        serie_validacao, serie_teste = train_test_split(serie_restante, train_size=prop_val)

        if verbose:
            print(f"Série com {n_total} registros dividida em:")
            print(f"Treino ({percentual_treino:.0%}): {len(serie_treino)}")
            print(f"Validação ({percentual_validacao:.0%}): {len(serie_validacao)}")
            print(f"Teste ({(1 - percentual_treino - percentual_validacao):.0%}): {len(serie_teste)}")

        return serie_treino, serie_validacao, serie_teste

    else:
        serie_treino, serie_teste = train_test_split(serie, train_size=percentual_treino)

        if verbose:
            print(f"Série com {n_total} registros dividida em:")
            print(f"Treino ({percentual_treino:.0%}): {len(serie_treino)}")
            print(f"Teste ({(1 - percentual_treino):.0%}): {len(serie_teste)}")

        return serie_treino, serie_teste
