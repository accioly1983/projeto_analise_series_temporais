from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)

def calcular_metricas_erro(y_true, y_pred):
    """
    Calcula MSE e MAE entre os valores reais e previstos.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred)
    }

def print_metricas_erro(metricas, titulo="Avaliação Out-of-Sample"):
    print(f"{titulo}")
    print(f"MAE: {metricas['mae']:.4f}")
    print(f"MAPE: {metricas['mape']:.4f}")
    print(f"MSE: {metricas['mse']:.4f}")