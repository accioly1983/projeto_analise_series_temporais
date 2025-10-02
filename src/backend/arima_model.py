import pmdarima as pm
import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
import time

from backend.residuals import (
    diagnostico_residuos, plot_acf_residuos,
    plot_residuos
)
from backend.evaluation import (
    calcular_metricas_erro, print_metricas_erro
)

def treinar_modelo_arima(y_train, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    """Treina um modelo ARIMA com a ordem especificada."""

    model = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order)
    
    model.fit(y_train)
    return model

def prever_in_sample(model) -> np.ndarray:
    """Realiza previsão in-sample."""
    return model.predict_in_sample()

def plot_previsao_in_sample(y_train, predictions, order):
    p, d, q = order
    offset = max(p, q) + d

    plt.figure(figsize=(10, 5))
    plt.plot(predictions[offset:], label='Previsto')
    plt.plot(y_train[offset:], label='Real')
    plt.legend()
    plt.title("Previsão in-sample")
    plt.grid(True)
    plt.show()

def prever_out_of_sample(model, y_test: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Previsão 1-passo-à-frente com intervalo de confiança, atualizando o modelo a cada nova observação.
    """
    forecasts, confs = [], []
    for new_ob in y_test:
        fc, conf = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        forecasts.append(fc[0])
        confs.append(conf[0])

        # Atualiza o modelo com o valor real observado em t (new_ob) que será 
        # levado em consideração para prever o próximo valor de teste.
        model.update(new_ob)
    return np.asarray(forecasts), np.asarray(confs)

def plot_forecast_com_intervalo(forecasts, confs, y_test):
    idx = y_test.index
    lo = confs[:, 0]
    hi = confs[:, 1]
    
    plt.figure(figsize=(10, 5))
    plt.plot(idx, y_test, label='Real')
    plt.plot(idx, forecasts, label='Previsão 1-step')
    plt.fill_between(idx, lo, hi, alpha=0.2, label='95% IC')
    plt.legend()
    plt.title("Previsão out-of-sample")
    plt.grid(True)
    plt.show()

def calcular_residuos_in_sample(modelo) -> np.ndarray:
    """
    Retorna o resíduo (real - previsto) in-sample.
    """
    p, d, q = modelo.order
    offset = max(p, q) + d
  
    return modelo.arima_res_.resid[offset:]


def calcular_residuos_out_of_sample(y_test, forecasts) -> np.ndarray:
    """
    Retorna o resíduo (real - previsto) out-of-sample.
    """

    if isinstance(forecasts, np.ndarray):
        forecasts = pd.Series(forecasts, index=y_test.index)

    # Retorna resíduos como Series com o índice temporal
    return y_test - forecasts


def rodar_modelo_arima(
    y_train,
    y_test,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    nome_modelo="ARIMA"
):
    """
    Executa o pipeline completo de treino, avaliação e diagnóstico para um modelo ARIMA.

    Parâmetros:
    - y_train: série temporal de treino
    - y_test: série temporal de teste
    - order: tupla (p, d, q)
    - seasonal_order: tupla (P, D, Q, m)
    - nome_modelo: string para exibir nos gráficos e prints

    Retorna:
    - dicionário com métricas de erro e resíduos out-of-sample
    """

    inicio = time.time()
    print("="*80)
    print(f"Rodando {nome_modelo} com order={order} seasonal_order={seasonal_order}")
    print("="*80)

    # Treino do modelo
    print("\nEtapa 1 - Treinamento do Modelo")
    modelo = treinar_modelo_arima(y_train, order=order, seasonal_order=seasonal_order)

    # Previsões In-Sample
    print("\nEtapa 2 - Previsões In-Sample")
    previsoes_in = prever_in_sample(modelo)
    plot_previsao_in_sample(y_train, previsoes_in, modelo.order)

    print("\nEtapa 3 - Análise de Resíduos In-Sample")
    resid_in = calcular_residuos_in_sample(modelo)
    plot_residuos(resid_in, "Resíduos In-Sample")
    diagnostico_residuos(resid_in)
    plot_acf_residuos(resid_in, "Resíduos In-Sample")

    # Previsões Out-of-Sample
    print("\nEtapa 4 - Previsões Out-of-Sample")
    forecasts, confs = prever_out_of_sample(modelo, y_test)
    plot_forecast_com_intervalo(forecasts, confs, y_test)

    # Métricas
    print("\nEtapa 5 - Avaliação de Métricas")
    metricas = calcular_metricas_erro(y_test, forecasts)
    print_metricas_erro(metricas)

    # Resíduos Out-of-Sample
    print("\nEtapa 6 - Análise de Resíduos Out-of-Sample")
    resid_out = calcular_residuos_out_of_sample(y_test, forecasts)
    plot_residuos(resid_out, "Resíduos Out-of-Sample")
    diagnostico_residuos(resid_out)
    plot_acf_residuos(resid_out, "Resíduos Out-of-Sample")

    fim = time.time()
    duracao_segundos = fim - inicio
    duracao_minutos = duracao_segundos / 60
    
    print("\nPipeline finalizado!")
    print(f"Tempo total de execução: {duracao_minutos:.2f} minutos")
    print("="*80)

    return {
        "modelo": modelo,
        "metricas": metricas,
        "residuos_in": resid_in,
        "residuos_out": resid_out,
        "tempo_execucao_minutos": duracao_minutos
    }


