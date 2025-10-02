
from statsmodels.tsa.stattools import adfuller

def teste_adf(serie, regression="c", maxlag=None):
    """
    Executa o teste ADF e retorna um dicionário com os resultados.
    Se a série for constante, retorna aviso e pula o cálculo.
    """

    # Se for DataFrame, pega a primeira coluna
    if hasattr(serie, "columns"):
        if serie.shape[1] > 1:
            return {"Mensagem": "DataFrame com múltiplas colunas recebido. Informe apenas uma série."}
        serie = serie.iloc[:, 0]

    serie = serie.dropna()

    # Se todos os valores forem iguais (resíduo constante)
    if serie.nunique() <= 1:
        print("Série constante - teste ADF não aplicável.")

    else:

        result = adfuller(serie, regression=regression, maxlag=maxlag, autolag="AIC")

        if result[1] <= 0.05:
            print(f"Série é estacionária "
                    f"(p-valor={result[1]:.4f})")
        else:
            print(f"Série NÃO é estacionária "
                    f"(p-valor={result[1]:.4f})")
            
        print('ADF Statistic:', result[0])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
