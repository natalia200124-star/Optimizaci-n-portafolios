import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# DISEÑO PROFESIONAL
# =========================
st.set_page_config(
    page_title="Optimización de Portafolios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    h1 {
        color: #00d9ff !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #66d9ff !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input {
        background-color: #1e2433 !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d9ff !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
    }

    .stSlider > div > div > div > div {
        background-color: #00d9ff !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.3) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.5) !important;
    }

    .stDataFrame {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }

    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 217, 255, 0.25) !important;
        color: #00d9ff !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #151c28 !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
        border-radius: 0 0 10px 10px !important;
    }

    .stChatMessage {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        margin: 0.5rem 0 !important;
    }
    .stChatInputContainer {
        background-color: #1e2433 !important;
        border-radius: 15px !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
    }

    .stAlert {
        background-color: rgba(0, 217, 255, 0.08) !important;
        border-left: 4px solid #00d9ff !important;
        border-radius: 10px !important;
    }

    hr {
        border-color: rgba(0, 217, 255, 0.3) !important;
        margin: 2rem 0 !important;
    }

    p, li, span, label { color: #e1e7ed !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1a1f2e; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d9ff, #0099cc);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE - INICIALIZACIÓN
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

st.title("Optimización de Portafolios – Modelo de Markowitz")

st.markdown("""
### ¿Qué es un ticker?

Un **ticker** es el código con el que se identifica una acción en la bolsa de valores.
Cada empresa cotizada tiene un ticker único que permite acceder a su información de mercado.

**Ejemplos comunes:**
- **AAPL** → Apple Inc.
- **MSFT** → Microsoft Corporation
- **GOOGL** → Alphabet (Google)

Estos códigos se utilizan para descargar automáticamente los precios históricos
y realizar el análisis financiero del portafolio.
""")

tickers_input = st.text_input(
    "Ingrese los tickers separados por comas (ejemplo: AAPL, MSFT, GOOGL)",
    help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma."
)

years = st.slider(
    "Seleccione el horizonte temporal (años)",
    min_value=3,
    max_value=10,
    value=6
)

if st.button("Ejecutar optimización"):
    st.session_state.run_analysis = True
    st.session_state.analysis_done = False

if st.session_state.run_analysis and not st.session_state.analysis_done:

        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        if len(tickers) < 2:
            st.error("Ingrese al menos 2 tickers.")
            st.stop()

        try:

            # =====================================================================
            # 1.5) DESCARGA Y DEPURACIÓN DE DATOS (SIN LOOK-AHEAD BIAS)
            # =====================================================================
            end_date = datetime.today()
            start_date = end_date.replace(year=end_date.year - years)

            raw_data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )

            # Usar precios ajustados (corrige splits y dividendos)
            data = raw_data["Adj Close"]

            # En caso de MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(0, axis=1)

            data = data[tickers]

            # Ordenar por fecha (seguridad)
            data = data.sort_index()

            # Rellenar valores faltantes SOLO hacia adelante
            data = data.ffill()

            # Eliminar filas que sigan incompletas (inicio de la serie)
            data = data.dropna()

            st.subheader("Precios ajustados depurados (primeras filas)")
            st.dataframe(data.head())

            # =====================================================================
            # 2) RETORNOS Y MATRICES
            # =====================================================================
            returns = data.pct_change().dropna()
            mean_returns_daily = returns.mean()
            cov_daily = returns.cov()

            trading_days = 252
            mean_returns_annual = mean_returns_daily * trading_days
            cov_annual = cov_daily * trading_days

            # =====================================================================
            # 3) FUNCIONES DE OPTIMIZACIÓN
            # =====================================================================
            def performance(weights, mean_ret, cov):
                ret = np.dot(weights, mean_ret)
                vol = np.sqrt(weights.T @ cov @ weights)
                sharpe = ret / vol if vol > 0 else 0
                return ret, vol, sharpe

            def neg_sharpe(weights):
                r, v, _ = performance(weights, mean_returns_annual, cov_annual)
                return -(r / v) if v > 0 else 1e6

            def vol(weights):
                return np.sqrt(weights.T @ cov_annual @ weights)

            def max_drawdown(series):
                cumulative_max = series.cummax()
                drawdown = (series / cumulative_max) - 1
                return drawdown.min()

            n = len(tickers)
            x0 = np.repeat(1 / n, n)
            bounds = tuple((0, 1) for _ in range(n))
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

            # =====================================================================
            # 4) OPTIMIZACIONES
            # =====================================================================
            res_sharpe = minimize(neg_sharpe, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            weights_sharpe = res_sharpe.x
            ret_sharpe, vol_sharpe, sharpe_sharpe = performance(
                weights_sharpe, mean_returns_annual, cov_annual
            )

            res_minvol = minimize(vol, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            weights_minvol = res_minvol.x
            ret_minvol, vol_minvol, sharpe_minvol = performance(
                weights_minvol, mean_returns_annual, cov_annual
            )

            weights_equal = np.repeat(1 / n, n)
            ret_equal, vol_equal, sharpe_equal = performance(
                weights_equal, mean_returns_annual, cov_annual
            )

            # =====================================================================
            # 5) RENDIMIENTOS DE CADA ESTRATEGIA
            # =====================================================================
            cumulative_assets = (1 + returns).cumprod()

            daily_sharpe = returns.dot(weights_sharpe)
            daily_minvol = returns.dot(weights_minvol)
            daily_equal = returns.dot(weights_equal)

            cum_sharpe = (1 + daily_sharpe).cumprod()
            cum_minvol = (1 + daily_minvol).cumprod()
            cum_equal = (1 + daily_equal).cumprod()

            dd_sharpe = max_drawdown(cum_sharpe)
            dd_minvol = max_drawdown(cum_minvol)
            dd_equal = max_drawdown(cum_equal)

            # =====================================================================
            # 5.1) DESCARGA DE BENCHMARKS DE MERCADO
            # =====================================================================

            benchmarks = {
                "S&P 500 (SPY)": "SPY",
                "Nasdaq 100 (QQQ)": "QQQ",
                "MSCI World (URTH)": "URTH"
            }

            benchmark_data = yf.download(
                list(benchmarks.values()),
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )["Adj Close"]

            # Asegurar formato correcto
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data = benchmark_data.droplevel(0, axis=1)

            benchmark_data = benchmark_data.ffill().dropna()

            benchmark_returns = benchmark_data.pct_change().dropna()
            benchmark_cum = (1 + benchmark_returns).cumprod()


            # =====================================================================
            # 6) FRONTERA EFICIENTE
            # =====================================================================
            target_returns = np.linspace(
                mean_returns_annual.min(),
                mean_returns_annual.max(),
                50
            )

            efficient_vols, efficient_rets = [], []

            for targ in target_returns:
                cons = (
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                    {"type": "eq",
                     "fun": lambda w, targ=targ: np.dot(w, mean_returns_annual) - targ}
                )
                res = minimize(vol, x0, method="SLSQP",
                               bounds=bounds, constraints=cons)
                if res.success:
                    r, v, _ = performance(res.x, mean_returns_annual, cov_annual)
                    efficient_rets.append(r)
                    efficient_vols.append(v)

            # =====================================================================
            # 7) PRECIOS 2025 Y TENDENCIA
            # =====================================================================
            st.subheader("Precios relevantes del año 2025 (últimas 10 filas)")
            precios_2025 = data[data.index.year == 2025].tail(10)
            st.dataframe(precios_2025 if not precios_2025.empty else "No hay datos de 2025.")

            st.subheader(f"Tendencia de precios (últimos {years} años)")
            st.line_chart(data)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Tendencia de precios"):
                st.markdown(
                    """
                    **Interpretación:**

                    Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
                    durante el horizonte temporal seleccionado.

                    - Tendencias crecientes indican periodos de apreciación del activo.
                    - Periodos de alta pendiente reflejan fases de crecimiento acelerado.
                    - Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
                      o episodios de alta volatilidad.

                    Este análisis permite identificar activos con comportamientos más estables
                    frente a otros con mayor variabilidad en el tiempo.
                    """
                )

            # =====================================================================
            # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
            # =====================================================================
            st.subheader("Comparación sistemática de estrategias")

            df_compare = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Retorno Anual": [ret_sharpe, ret_minvol, ret_equal],
                "Volatilidad": [vol_sharpe, vol_minvol, vol_equal],
                "Sharpe": [sharpe_sharpe, sharpe_minvol, sharpe_equal],
                "Retorno Acumulado": [
                    cum_sharpe.iloc[-1] - 1,
                    cum_minvol.iloc[-1] - 1,
                    cum_equal.iloc[-1] - 1
                ],
                "Máx Drawdown": [dd_sharpe, dd_minvol, dd_equal]
            })

            st.dataframe(df_compare)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Comparación de estrategias"):
                st.markdown(
                    """
                    **Cómo interpretar esta tabla:**
                    - **Retorno acumulado:** cuánto creció el capital total en el periodo.
                    - **Volatilidad:** magnitud de las fluctuaciones (riesgo).
                    - **Sharpe:** eficiencia riesgo–retorno.
                    - **Máx Drawdown:** peor caída histórica desde un máximo.

                    **Interpretación analítica de la comparación de estrategias:**

                    Esta tabla sintetiza el desempeño de las distintas estrategias
                    de construcción de portafolios bajo un enfoque riesgo–retorno,
                    permitiendo una evaluación integral y comparativa.

                    - La estrategia de **Sharpe Máximo** tiende a ofrecer el mayor
                      retorno ajustado por riesgo, aunque suele presentar niveles
                      más elevados de volatilidad y drawdowns en periodos adversos.
                    - La estrategia de **Mínima Volatilidad** prioriza la estabilidad
                      del capital, reduciendo la exposición a caídas pronunciadas,
                      a costa de un menor potencial de crecimiento.
                    - La estrategia de **Pesos Iguales** actúa como referencia neutral,
                      proporcionando una diversificación básica sin optimización explícita.

                    La combinación de métricas como retorno anual, volatilidad,
                    Ratio de Sharpe y máximo drawdown permite identificar no solo
                    la estrategia más rentable, sino también la más resiliente
                    frente a escenarios de estrés de mercado.

                    Este análisis respalda decisiones de asignación de activos
                    alineadas con el horizonte temporal y el perfil de riesgo del inversor.
                    """
                )

            # =====================================================================
            # 8.1) VOLATILIDAD HISTÓRICA ROLLING (RIESGO DINÁMICO)
            # =====================================================================
            st.subheader("Volatilidad histórica móvil")

            rolling_vol = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
            })

            st.line_chart(rolling_vol)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Volatilidad histórica móvil"):
                st.markdown(
                    """
                    **Interpretación:**
                    Esta gráfica muestra cómo el riesgo **cambia en el tiempo**.
                    - Picos altos suelen coincidir con periodos de crisis.
                    - Estrategias más estables presentan curvas más suaves.

                    La volatilidad histórica móvil permite analizar cómo
                    evoluciona el riesgo del portafolio a lo largo del tiempo,
                    capturando cambios estructurales en el comportamiento del mercado.

                    - Incrementos abruptos de la volatilidad suelen coincidir
                      con periodos de crisis financiera o incertidumbre macroeconómica.
                    - Curvas más suaves indican estrategias con mayor estabilidad
                      y menor sensibilidad a shocks de mercado.

                    En el análisis comparativo:
                    - El portafolio de **Sharpe Máximo** presenta picos de
                      volatilidad más elevados, reflejando una mayor exposición
                      al riesgo en escenarios adversos.
                    - La estrategia de **Mínima Volatilidad** mantiene un perfil
                      de riesgo más controlado a lo largo del tiempo.
                    - La asignación de **Pesos Iguales** muestra un comportamiento
                      intermedio, replicando parcialmente la dinámica del mercado.

                    Este enfoque dinámico del riesgo complementa las métricas
                    estáticas tradicionales y aporta una visión más realista
                    del comportamiento del portafolio.
                    """
                )

            # =====================================================================
            # 8.2) RATIO CALMAR
            # =====================================================================
            calmar_sharpe = ret_sharpe / abs(dd_sharpe)
            calmar_minvol = ret_minvol / abs(dd_minvol)
            calmar_equal = ret_equal / abs(dd_equal)

            st.subheader("Ratio Calmar (retorno vs drawdown)")

            df_calmar = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Calmar": [calmar_sharpe, calmar_minvol, calmar_equal]
            })

            st.dataframe(df_calmar)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Ratio Calmar"):
                st.markdown(
                    """
                    **Interpretación analítica del Ratio Calmar:**

                    El Ratio Calmar relaciona el **retorno anual esperado** con el
                    **máximo drawdown histórico**, ofreciendo una medida directa
                    de la capacidad del portafolio para generar rentabilidad
                    sin incurrir en pérdidas extremas prolongadas.

                    - Un **Ratio Calmar elevado** indica que la estrategia logra
                      retornos atractivos manteniendo caídas relativamente
                      controladas.
                    - Valores bajos sugieren que el retorno obtenido no compensa
                      adecuadamente las pérdidas máximas sufridas.
                    - Esta métrica resulta especialmente relevante para
                      inversionistas con enfoque conservador o con restricciones
                      estrictas de preservación de capital.

                    A diferencia del Ratio de Sharpe, el Calmar se centra en el
                    **riesgo extremo observado**, lo que lo convierte en un
                    indicador complementario para evaluar la resiliencia del
                    portafolio en periodos de crisis o alta volatilidad.

                    En el contexto del presente análisis, el Ratio Calmar permite
                    identificar qué estrategia ofrece un **mejor equilibrio entre
                    crecimiento del capital y control de pérdidas severas**,
                    reforzando la robustez del proceso de selección de portafolios.
                    """
                )

            # =====================================================================
            # 8.3) SORTINO RATIO
            # =====================================================================
            downside = returns.copy()
            downside[downside > 0] = 0
            downside_std = downside.std() * np.sqrt(252)

            sortino_sharpe = ret_sharpe / downside_std.dot(weights_sharpe)
            sortino_minvol = ret_minvol / downside_std.dot(weights_minvol)
            sortino_equal = ret_equal / downside_std.dot(weights_equal)

            st.subheader("Ratio Sortino")

            df_sortino = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Sortino": [sortino_sharpe, sortino_minvol, sortino_equal]
            })

            st.dataframe(df_sortino)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Ratio Sortino"):
                st.markdown(
                    """
                    **Interpretación analítica del Ratio Sortino:**

                    El Ratio Sortino evalúa el desempeño del portafolio considerando
                    exclusivamente la **volatilidad negativa**, es decir, aquellas
                    fluctuaciones que representan pérdidas para el inversor.

                    - Un **valor más alto de Sortino** indica que la estrategia genera
                      mayor retorno por cada unidad de riesgo a la baja asumida.
                    - A diferencia del Ratio de Sharpe, este indicador **no penaliza
                      la volatilidad positiva**, lo que lo convierte en una métrica
                      más alineada con la percepción real del riesgo por parte del inversor.
                    - Estrategias con Sortino elevado suelen ser más adecuadas para
                      escenarios de mercado inciertos o para perfiles que priorizan
                      la protección frente a caídas.

                    En el contexto del análisis comparativo, el Ratio Sortino permite
                    identificar qué estrategia ofrece una **mejor compensación entre
                    retorno y riesgo negativo**, aportando una visión complementaria
                    y más conservadora al proceso de toma de decisiones.
                    """
                )

            # =====================================================================
            # 8.4) PERIODOS DE CRISIS (COVID 2020)
            # =====================================================================
            st.subheader("Comportamiento en periodo de crisis (COVID 2020)")

            crisis = (cum_sharpe.index.year == 2020)

            st.line_chart(pd.DataFrame({
                "Sharpe Máximo": cum_sharpe[crisis],
                "Mínima Volatilidad": cum_minvol[crisis],
                "Pesos Iguales": cum_equal[crisis]
            }))

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Comportamiento en crisis (COVID 2020)"):
                st.markdown(
                    """
                    **Interpretación del comportamiento en periodo de crisis:**

                    Esta visualización muestra el desempeño de las distintas
                    estrategias durante un periodo de estrés sistémico,
                    caracterizado por alta volatilidad y caídas abruptas del mercado.

                    El análisis permite evaluar:
                    - La **profundidad de la caída** inicial (drawdown).
                    - La **velocidad de recuperación** tras el shock.
                    - La **resiliencia relativa** de cada estrategia ante eventos extremos.

                    Los resultados evidencian que:
                    - Las estrategias optimizadas para maximizar el retorno
                      (como Sharpe Máximo) tienden a experimentar caídas más
                      pronunciadas en el corto plazo.
                    - Las estrategias orientadas a la reducción de riesgo
                      (Mínima Volatilidad) presentan una mayor capacidad de
                      contención de pérdidas.

                    Este análisis refuerza la idea de que la eficiencia
                    riesgo–retorno debe evaluarse no solo en condiciones normales,
                    sino también bajo escenarios adversos.
                    """
                )

            # =====================================================================
            # 8.5) COMPARACIÓN CON BENCHMARKS DE MERCADO
            # =====================================================================

            st.subheader("Comparación con benchmarks de mercado")

            def annualized_return(series):
                return (series.iloc[-1]) ** (252 / len(series)) - 1

            def annualized_vol(series):
                return series.std() * np.sqrt(252)

            benchmark_summary = []

            for name, ticker in benchmarks.items():
                ret = annualized_return(benchmark_cum[ticker])
                vol = annualized_vol(benchmark_returns[ticker])
                dd = max_drawdown(benchmark_cum[ticker])

                benchmark_summary.append({
                    "Benchmark": name,
                    "Retorno Anual": ret,
                    "Volatilidad": vol,
                    "Retorno Acumulado": benchmark_cum[ticker].iloc[-1] - 1,
                    "Máx Drawdown": dd
                })

            df_benchmarks = pd.DataFrame(benchmark_summary)
            st.dataframe(df_benchmarks)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 ¿Qué es un benchmark? – S&P 500, MSCI y NASDAQ explicados"):
                st.markdown("""
                ### ¿Qué es un benchmark?

                Un **benchmark** es un **punto de referencia** que se utiliza para evaluar si una estrategia de inversión es buena o mala.
                Funciona de forma similar a una *regla de medición*: permite comparar los resultados obtenidos con una alternativa estándar y ampliamente utilizada en los mercados financieros.

                En este trabajo, los benchmarks representan **formas simples y comunes de invertir**, frente a las cuales se comparan las estrategias optimizadas desarrolladas en la aplicación.

                ### ¿Qué representa el S&P 500?

                El **S&P 500** es uno de los índices bursátiles más conocidos del mundo. Agrupa a aproximadamente **500 de las empresas más grandes de Estados Unidos**, como Apple, Microsoft o Google.
                Invertir en el S&P 500 se considera una aproximación al comportamiento general del mercado y suele utilizarse como referencia básica para evaluar el desempeño de cualquier portafolio.

                Si una estrategia no logra superar al S&P 500 en el largo plazo, resulta difícil justificar su complejidad frente a una inversión pasiva en el mercado.

                ### ¿Qué es el MSCI?

                **MSCI** (Morgan Stanley Capital International) es una empresa internacional que elabora **índices bursátiles** utilizados como referencia en todo el mundo.
                Un índice MSCI representa el comportamiento de un conjunto amplio de empresas de una región o del mercado global.

                Por ejemplo:
                - **MSCI World** agrupa empresas grandes y medianas de países desarrollados.
                - **MSCI Emerging Markets** representa mercados emergentes.

                Estos índices se utilizan como benchmark porque reflejan el desempeño promedio de mercados completos y permiten evaluar si una estrategia supera o no una inversión diversificada a nivel internacional.

                ### ¿Qué es el NASDAQ?

                El **NASDAQ** es una bolsa de valores estadounidense caracterizada por una **alta concentración de empresas tecnológicas y de innovación**, como Apple, Microsoft, Amazon o Google.
                El índice NASDAQ suele mostrar mayores crecimientos en periodos de expansión económica, pero también presenta **mayor volatilidad** en momentos de crisis.

                Por esta razón, el NASDAQ se utiliza como benchmark para comparar estrategias con un perfil más dinámico y orientado al crecimiento, especialmente en sectores tecnológicos.

                ### ¿Por qué se incluyen estos índices como benchmarks?

                La inclusión del **S&P 500, MSCI y NASDAQ** permite comparar los portafolios optimizados con:
                - El comportamiento general del mercado estadounidense (S&P 500),
                - Una referencia de diversificación global (MSCI),
                - Un mercado de alto crecimiento y mayor riesgo (NASDAQ).

                De esta forma, se obtiene una evaluación más completa del desempeño relativo de las estrategias desarrolladas en la aplicación.

                ### ¿Por qué se comparan varias estrategias?

                Además del S&P 500, se incluyen otras estrategias como:
                - **Pesos iguales**, donde todos los activos reciben la misma proporción.
                - **Portafolio de mínima volatilidad**, orientado a reducir el riesgo.
                - **Portafolio de Sharpe máximo**, que busca el mejor retorno ajustado por riesgo.

                La comparación con estos benchmarks permite responder una pregunta clave:
                **¿La optimización realmente mejora los resultados frente a alternativas simples y ampliamente utilizadas?**
                """)

            # =====================================================================
            # 8.6) RENDIMIENTO ACUMULADO: ESTRATEGIAS VS BENCHMARKS
            # =====================================================================

            st.subheader("Rendimiento acumulado: estrategias vs benchmarks")

            comparison_cum = pd.DataFrame({
                "Sharpe Máximo": cum_sharpe,
                "Mínima Volatilidad": cum_minvol,
                "Pesos Iguales": cum_equal,
                "S&P 500 (SPY)": benchmark_cum["SPY"],
                "Nasdaq 100 (QQQ)": benchmark_cum["QQQ"],
                "MSCI World (URTH)": benchmark_cum["URTH"]
            })

            st.line_chart(comparison_cum)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Rendimiento acumulado vs benchmarks"):
                st.markdown("""
                **Cómo interpretar la gráfica de rendimiento acumulado**

                Esta gráfica muestra cómo habría evolucionado una inversión inicial a lo largo del tiempo bajo cada estrategia.

                - La línea que termina **más arriba** representa la estrategia con **mayor crecimiento acumulado**.
                - Las curvas más **suaves y estables** indican menor volatilidad y menor exposición a crisis.
                - Caídas pronunciadas reflejan periodos de estrés de mercado; una recuperación rápida indica mayor resiliencia.
                - Si una estrategia optimizada supera de forma consistente a los benchmarks, se confirma que el modelo aporta valor frente a una inversión pasiva.

                La interpretación conjunta del gráfico permite evaluar no solo cuánto se gana, sino **cómo se gana**, identificando estrategias más robustas frente a escenarios adversos.
                """)

            # =====================================================================
            # 9) SÍNTESIS ANALÍTICA PARA EL ASISTENTE (PERSISTENTE)
            # =====================================================================

            asset_summary = {}

            for ticker in tickers:
                asset_summary[ticker] = {
                    "retorno_anual": mean_returns_annual[ticker],
                    "volatilidad": np.sqrt(cov_annual.loc[ticker, ticker]),
                    "contribucion_riesgo": cov_annual.loc[ticker].dot(weights_sharpe)
                }

            strategy_summary = {
                "Sharpe Máximo": {
                    "retorno": ret_sharpe,
                    "volatilidad": vol_sharpe,
                    "sharpe": sharpe_sharpe,
                    "drawdown": dd_sharpe
                },
                "Mínima Volatilidad": {
                    "retorno": ret_minvol,
                    "volatilidad": vol_minvol,
                    "sharpe": sharpe_minvol,
                    "drawdown": dd_minvol
                },
                "Pesos Iguales": {
                    "retorno": ret_equal,
                    "volatilidad": vol_equal,
                    "sharpe": sharpe_equal,
                    "drawdown": dd_equal
                }
            }


            # =====================================================================
            # 10) RENDIMIENTOS ACUMULADOS
            # =====================================================================
            st.subheader("Rendimiento acumulado por acción")
            st.line_chart(cumulative_assets)

            st.subheader("Comparación de rendimientos de estrategias")
            st.line_chart(
                pd.DataFrame({
                    "Sharpe Máximo": cum_sharpe,
                    "Mínima Volatilidad": cum_minvol,
                    "Pesos Iguales": cum_equal
                })

            )

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Rendimiento acumulado por acción"):
                st.markdown(
                    """
                    **Interpretación:**

                    El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
                    en cada activo si se hubiera mantenido durante todo el periodo de análisis.

                    - Curvas más empinadas indican mayor crecimiento del capital.
                    - Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.
                    - Diferencias significativas entre curvas evidencian distintos perfiles
                      de riesgo y rentabilidad.

                    Este gráfico facilita la comparación directa del desempeño histórico
                    entre los activos analizados.
                    """
                )

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS ACUMULADOS
            # =====================================================================
            st.subheader("Retornos diarios de los activos")
            st.line_chart(returns)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Retornos diarios de los activos"):
                st.markdown(
                    """
                    **Interpretación:**

                    Este gráfico muestra los retornos porcentuales diarios de cada activo,
                    evidenciando la volatilidad de corto plazo.

                    - Picos positivos o negativos representan movimientos abruptos del mercado.
                    - Mayor dispersión implica mayor riesgo.
                    - Periodos de alta concentración de picos suelen coincidir con crisis financieras
                      o eventos macroeconómicos relevantes.

                    Este análisis es clave para evaluar el riesgo diario asumido por el inversor.
                    """
                )

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS POR ACTIVO
            # =====================================================================

            st.subheader("Retornos diarios por activo")

            for ticker in returns.columns:
                  st.markdown(f"### {ticker}")
                  st.line_chart(returns[[ticker]])

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Retornos diarios por activo individual"):
                st.markdown(
                    """
                    **Interpretación:**

                    Este gráfico muestra el comportamiento diario del retorno del activo,
                    permitiendo identificar:

                    - Frecuencia e intensidad de pérdidas y ganancias.
                    - Presencia de volatilidad asimétrica (más caídas que subidas).
                    - Episodios de estrés específicos para el activo.

                    Resulta útil para evaluar el riesgo individual antes de integrarlo
                    dentro de un portafolio diversificado.
                    """
                )

            # =====================================================================
            # 11) FRONTERA EFICIENTE (MEJORADA CON ETIQUETAS)
            # =====================================================================
            st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

            # ── DISEÑO: columnas para centrar y reducir tamaño de la gráfica ──
            _col1, _col2, _col3 = st.columns([0.5, 2, 0.5])
            with _col2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))   # antes: (8, 6)

                # Frontera eficiente
                ax2.plot(
                        efficient_vols,
                        efficient_rets,
                        linestyle="-",
                        linewidth=2,
                        label="Frontera eficiente"
                )
                # Portafolios destacados
                ax2.scatter(
                        vol_sharpe,
                        ret_sharpe,
                        s=90,
                        marker="o",
                        label="Sharpe Máximo"
                )

                ax2.scatter(
                        vol_minvol,
                        ret_minvol,
                        s=90,
                        marker="^",
                        label="Mínima Volatilidad"
                )
                ax2.scatter(
                        vol_equal,
                        ret_equal,
                        s=90,
                        marker="s",
                        label="Pesos Iguales"
                )
                # Etiquetas de los puntos
                ax2.annotate(
                        "Sharpe Máximo",
                        (vol_sharpe, ret_sharpe),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontweight="bold"
                )
                ax2.annotate(
                        "Mínima Volatilidad",
                        (vol_minvol, ret_minvol),
                        xytext=(8, -12),
                        textcoords="offset points",
                        fontweight="bold"
                )
                ax2.annotate(
                        "Pesos Iguales",
                        (vol_equal, ret_equal),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontweight="bold"
                )
                # Ejes y título
                ax2.set_xlabel("Volatilidad anual (riesgo)")
                ax2.set_ylabel("Retorno anual esperado")
                ax2.set_title("Frontera eficiente y estrategias comparadas")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Frontera eficiente de Markowitz"):
                st.markdown(
                    """
                    **Interpretación analítica de la frontera eficiente:**

                    La frontera eficiente representa el conjunto de portafolios
                    óptimos que maximizan el retorno esperado para cada nivel
                    de riesgo asumido, de acuerdo con la teoría media–varianza
                    de Markowitz.

                    - Cada punto de la curva corresponde a una combinación
                      distinta de activos que no puede ser mejorada simultáneamente
                      en términos de mayor retorno y menor riesgo.
                    - Los portafolios situados por debajo de la frontera son
                      ineficientes, ya que existe al menos una alternativa
                      con mejor desempeño riesgo–retorno.

                    La ubicación de las estrategias analizadas sobre la frontera
                    permite identificar su perfil:
                    - El portafolio de **Sharpe Máximo** se sitúa en una zona de
                      mayor eficiencia, priorizando la rentabilidad ajustada
                      por riesgo.
                    - La estrategia de **Mínima Volatilidad** se posiciona en el
                      extremo de menor riesgo, sacrificando retorno esperado.
                    - La asignación de **Pesos Iguales** actúa como referencia
                      neutral, sin optimización explícita.

                    Esta visualización facilita la comprensión del trade-off
                    riesgo–retorno y constituye una herramienta central para
                    la toma de decisiones de inversión.
                    """
                )

            # =====================================================================
            # INTERPRETACIÓN FINAL – COMPORTAMIENTO REAL PONDERADO EN EL TIEMPO
            # =====================================================================
            st.subheader("Interpretación automática del mejor portafolio")

            df_strategies = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe,
                "Mínima Volatilidad": daily_minvol,
                "Pesos Iguales": daily_equal
            })

            # Ponderación temporal (años recientes pesan más)
            years_index = df_strategies.index.year
            unique_years = np.sort(years_index.unique())

            year_weights = {
                year: (i + 1) / len(unique_years)
                for i, year in enumerate(unique_years)
            }

            weights_series = years_index.map(year_weights)

            # Retorno real ponderado
            weighted_performance = (
                (1 + df_strategies).cumprod()
                .mul(weights_series, axis=0)
                .iloc[-1]
            )

            best = weighted_performance.idxmax()

            st.dataframe(weighted_performance.rename("Desempeño_Ponderado"))

            # Interpretación
            if best == "Pesos Iguales":
                st.markdown(
                    "### Mejor portafolio: Pesos Iguales\n\n"
                    "El análisis del **comportamiento real del portafolio en el tiempo**, "
                    "ponderando más los años recientes, muestra que esta estrategia ha sido "
                    "la **más robusta y consistente**.\n\n"
                    "- Menor dependencia de supuestos estadísticos.\n"
                    "- Mejor desempeño agregado a lo largo del tiempo.\n"
                    "- Alta estabilidad frente a cambios de mercado."
                )

            elif best == "Sharpe Máximo":
                st.markdown(
                    "### Mejor portafolio: Sharpe Máximo\n\n"
                    "La evaluación temporal indica que esta estrategia ofrece el mejor "
                    "equilibrio riesgo–retorno en el comportamiento histórico reciente."
                )

            else:
                st.markdown(
                    "### Mejor portafolio: Mínima Volatilidad\n\n"
                    "Esta estrategia destaca por su estabilidad, aunque sacrifica retorno "
                    "frente a las demás."
                )

            st.success(f"Portafolio recomendado según comportamiento real ponderado: {best}")

            # =====================================================================
            # 9) PESOS ÓPTIMOS SEGÚN PORTAFOLIO RECOMENDADO
            # =====================================================================
            st.subheader("Pesos óptimos del portafolio recomendado")

            n_assets = len(tickers)

            if best == "Sharpe Máximo":
                final_weights = weights_sharpe
                metodo = "Optimización por Ratio de Sharpe"

            elif best == "Mínima Volatilidad":
                final_weights = weights_minvol
                metodo = "Optimización por Mínima Volatilidad"

            else:  # Pesos Iguales
                final_weights = np.array([1 / n_assets] * n_assets)
                metodo = "Asignación Equitativa (Pesos Iguales)"

            df_weights = pd.DataFrame({
                "Ticker": tickers,
                "Peso": final_weights,
                "Peso (%)": final_weights * 100
            })

            st.dataframe(df_weights)

            # ── DISEÑO: columnas para centrar y reducir tamaño de la gráfica ──
            _cw1, _cw2, _cw3 = st.columns([0.5, 2, 0.5])
            with _cw2:
                fig, ax = plt.subplots(figsize=(6, 3.5))   # antes: sin figsize
                ax.barh(df_weights["Ticker"], df_weights["Peso"])
                ax.set_title(f"Composición del portafolio recomendado\n({metodo})")
                st.pyplot(fig)

            # ── DISEÑO: expander para la interpretación ──
            with st.expander("📖 Interpretación – Pesos óptimos del portafolio recomendado"):
                st.markdown(
                    f"""
                    ### Interpretación de los pesos

                    Los pesos mostrados corresponden **exclusivamente** al portafolio
                    recomendado por el modelo (**{best}**).

                    - Cada peso indica qué proporción del capital debe asignarse a cada activo.
                    - La suma total de los pesos es del **100%**.
                    - Esta asignación refleja el comportamiento histórico del portafolio
                      bajo el criterio seleccionado.

                    ### Explicación extendida de los pesos óptimos

                    Los **pesos óptimos** indican cómo distribuir el capital para obtener
                    el mejor balance entre **riesgo y retorno**, según el modelo de Markowitz.

                    - Un **peso del 40%** significa que **40 de cada 100 unidades monetarias**
                      se asignan a ese activo.
                    - **Pesos altos** reflejan activos que aportan mayor eficiencia al portafolio.
                    - **Pesos bajos** indican activos que añaden más riesgo que beneficio relativo.

                    Para personas sin experiencia previa,
                    esta tabla funciona como una **guía práctica de asignación de capital**,
                    evitando decisiones intuitivas o emocionales.
                    """
                )

            # ======================================================
            # GUARDAR RESULTADOS PARA EL CHAT
            # ======================================================
            st.session_state["analysis_results"] = {
                "tickers": tickers,
                "best": best,

                # Comparación general
                "comparison": df_compare,

                # Pesos del portafolio recomendado (tabla)
                "weights_recommended": df_weights,

                # Pesos óptimos por estrategia (clave para el chat)
                "weights": {
                    "Sharpe Máximo": dict(zip(tickers, weights_sharpe)),
                    "Mínima Volatilidad": dict(zip(tickers, weights_minvol)),
                    "Pesos Iguales": dict(zip(tickers, [1 / len(tickers)] * len(tickers)))
                },

                # Retornos esperados
                "retornos": {
                    "Sharpe Máximo": ret_sharpe,
                    "Mínima Volatilidad": ret_minvol,
                    "Pesos Iguales": ret_equal
                },

                # Volatilidades
                "volatilidades": {
                    "Sharpe Máximo": vol_sharpe,
                    "Mínima Volatilidad": vol_minvol,
                    "Pesos Iguales": vol_equal
                },
                "asset_summary": asset_summary,
                "strategy_summary": strategy_summary
            }

            # ── Marcar análisis completado y detener el bucle de re-ejecución ──
            st.session_state.analysis_done = True
            st.session_state.run_analysis = False

            st.success("Análisis del portafolio ejecutado correctamente")

        except Exception as e:
            st.error(f"Error: {e}")

# ======================================================
# MOSTRAR RESULTADOS (FUERA DEL BOTÓN)
# ======================================================

if st.session_state.analysis_done:
    results = st.session_state.analysis_results

    st.subheader("Comparación de estrategias")
    st.dataframe(results["comparison"])

    st.subheader("Pesos del portafolio recomendado")
    st.dataframe(results["weights_recommended"])

    df_retornos = pd.DataFrame(
        {
            "Retorno anual esperado": [
                results["retornos"]["Sharpe Máximo"],
                results["retornos"]["Mínima Volatilidad"],
                results["retornos"]["Pesos Iguales"]
            ]
        },
        index=["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"]
    )

    st.subheader("Ratio / retorno esperado por estrategia")
    st.dataframe(df_retornos)

# ======================================================
# ASISTENTE INTELIGENTE DEL PORTAFOLIO (GEMINI)
# ======================================================

st.divider()
st.subheader("🤖 Asistente inteligente del portafolio")

if not st.session_state.analysis_done:
    st.info("Ejecuta primero la optimización para habilitar el asistente.")
else:
    import requests
    import os

    # =========================
    # CONFIGURACIÓN GEMINI
    # =========================
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("El asistente requiere una API Key válida de Gemini.")
        st.stop()

    MODEL = "gemini-2.5-flash-lite"
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    # =========================
    # HISTORIAL DE CHAT
    # =========================
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(
        "Pregunta sobre los tickers, riesgos o el portafolio recomendado"
    )

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question}
        )

        results = st.session_state.analysis_results

        # =========================
        # CONTEXTO FINANCIERO
        # =========================
        best_strategy = results["best"]
        weights_dict = results["weights"][best_strategy]

        weights_text = "\n".join(
            f"- {k}: {v:.2%}" for k, v in weights_dict.items()
        )

        asset_text = "\n".join(
            f"- {k}: retorno anual={v['retorno_anual']:.2%}, "
            f"volatilidad={v['volatilidad']:.2%}"
            for k, v in results["asset_summary"].items()
        )

        strategy_text = "\n".join(
            f"- {k}: retorno={v['retorno']:.2%}, "
            f"volatilidad={v['volatilidad']:.2%}, "
            f"Sharpe={v['sharpe']:.2f}, "
            f"drawdown={v['drawdown']:.2%}"
            for k, v in results["strategy_summary"].items()
        )

        # =========================
        # PROMPT OPTIMIZADO
        # =========================
        system_prompt = f"""
Actúa como un analista financiero profesional.

CONTEXTO (úsalo solo si es necesario):
Activos analizados: {', '.join(results['tickers'])}

Resumen de activos:
{asset_text}

Resumen de estrategias:
{strategy_text}

Estrategia recomendada: {best_strategy}
Pesos del portafolio recomendado:
{weights_text}

INSTRUCCIONES ESTRICTAS:
- Responde ÚNICAMENTE la pregunta del usuario.
- Usa lenguaje claro para personas no técnicas.
- La respuesta DEBE tener al menos 2 párrafos cortos.
- Máximo 4 párrafos en total.
- Cada párrafo debe aportar información distinta (no repetir ideas).
- No expliques teoría financiera innecesaria.
- Si aplica, menciona brevemente riesgo y retorno.
- Si preguntan por cifras, usa números concretos.
- No inventes datos.
- Termina siempre la respuesta.
"""
        # =========================
        # LLAMADA A GEMINI
        # =========================
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": system_prompt
                            + "\n\nPregunta del usuario:\n"
                            + user_question
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 900
            }
        }

        response = requests.post(GEMINI_URL, json=payload)

        if response.status_code != 200:
            answer = "⚠️ Error al generar la respuesta con Gemini."
        else:
            data = response.json()
            answer = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No se obtuvo respuesta.")
            )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

































