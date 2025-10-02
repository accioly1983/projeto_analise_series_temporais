import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

def calc_residuo_mm(serie, window: int, center: bool = False):
    """
    Calcula a média móvel e o resíduo para uma janela específica.

    Parâmetros:
    - y (pd.Series): Série temporal original.
    - window (int): Tamanho da janela da média móvel.
    - center (bool): Se True, centraliza a janela.

    Retorna:
    - Tuple[pd.Series, pd.Series]: (media_movel, residuo)
    """
    mm = serie.rolling(window=window, center=center).mean()
    residuo = serie - mm
    return mm.dropna(), residuo.dropna()



def plot_mm_e_residuo(y, media_movel, residuo, window: int):
    """
    Plota a série real, a média móvel e o resíduo correspondente.

    Parâmetros:
    - y (pd.Series): Série original.
    - media_movel (pd.Series): Série com a média móvel.
    - residuo (pd.Series): Série com o resíduo.
    - window (int): Tamanho da janela usada.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    # Série real + média móvel
    axs[0].plot(y, label="Série Real")
    axs[0].plot(media_movel, label=f"Média Móvel ({window})")
    axs[0].set_title(f"Série Real e Média Móvel (janela={window})")
    axs[0].legend()
    axs[0].grid(True)

    # Resíduo
    axs[1].plot(residuo, label="Resíduo", color="tab:orange")
    axs[1].set_title(f"Resíduo (janela={window})")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def calc_residuo_ewm(serie, alpha: float):
    """
    Aplica suavização exponencial (EWM) e retorna: (suavizada, residuo)
    """
    suavizada = serie.ewm(alpha=alpha).mean()
    residuo = serie - suavizada
    return suavizada.dropna(), residuo.dropna()


def plot_ewm_e_residuo(serie, suavizada, residuo, alpha: float):
    """
    Plota a série real, a suavização exponencial (EWM) e o resíduo.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    axs[0].plot(serie, label="Série Real")
    axs[0].plot(suavizada, label=f"EWM (α={alpha})", color="tab:green")
    axs[0].set_title(f"Série Real e EWM (α={alpha})")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(residuo, label=f"Resíduo (α={alpha})", color="tab:red")
    axs[1].set_title(f"Resíduo (α={alpha})")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def diagnostico_residuos(resid):
    resid = np.asarray(resid).flatten()
    
    media, desvio = resid.mean(), resid.std(ddof=1)
    print(f"Média: {media:.4f}, Desvio-padrão: {desvio:.4f}")

    max_lags = min(20, len(resid)-1)
    lb = acorr_ljungbox(resid, lags=[10, max_lags], return_df=True)
    
    print("\nTeste Ljung-Box (autocorrelação):")
    for lag, stat, pval in zip(lb.index, lb["lb_stat"], lb["lb_pvalue"]):
        conclusao = "Sem autocorrelação" if pval > 0.05 else "Autocorrelação detectada"
        print(f"Lag {lag}: estatística={stat:.4f}, p-valor={pval:.4f} → {conclusao}")

def plot_acf_residuos(resid, titulo="Resíduos"):
    resid = np.asarray(resid).flatten()
    max_lags = min(20, len(resid)-1)
    
    plt.figure(figsize=(10, 4))
    
    sm.graphics.tsa.plot_acf(resid, lags=max_lags)
    
    plt.title(f"ACF - {titulo}")
    plt.tight_layout()
    plt.show()

def plot_residuos(resid, titulo):
    plt.figure(figsize=(12, 6))
    plt.plot(resid.index, resid.values, label="Resíduo", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--")
    plt.legend()
    plt.title(titulo)
    plt.show()