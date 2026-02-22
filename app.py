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
# FUNCIONES OPTIMIZADAS (CACHE)
# =========================

@st.cache_data(show_spinner="Descargando datos de activos...")
def get_crypto_data(tickers, years):
    end_date = datetime.today()
    start_date = end_date.replace(year=end_date.year - years)
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )
    if raw_data.empty:
        return None
    data = raw_data["Adj Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    data = data[tickers].sort_index().ffill().dropna()
    return data

@st.cache_data(show_spinner="Descargando benchmarks...")
def get_benchmark_data(years):
    end_date = datetime.today()
    start_date = end_date.replace(year=end_date.year - years)
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
    if isinstance(benchmark_data.columns, pd.MultiIndex):
        benchmark_data = benchmark_data.droplevel(0, axis=1)
    benchmark_data = benchmark_data.ffill().dropna()
    return benchmark_data, benchmarks

def performance(weights, mean_ret, cov):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(weights.T @ cov @ weights)
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def max_drawdown(series):
    cumulative_max = series.cummax()
    drawdown = (series / cumulative_max) - 1
    return drawdown.min()

@st.cache_data(show_spinner="Calculando optimizaciones...")
def run_portfolio_optimizations(returns, tickers):
    mean_returns_daily = returns.mean()
    cov_daily = returns.cov()
    trading_days = 252
    mean_returns_annual = mean_returns_daily * trading_days
    cov_annual = cov_daily * trading_days

    n = len(tickers)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(weights):
        r, v, _ = performance(weights, mean_returns_annual, cov_annual)
        return -(r / v) if v > 0 else 1e6

    def vol_func(weights):
        return np.sqrt(weights.T @ cov_annual @ weights)

    res_sharpe = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights_sharpe = res_sharpe.x
    ret_sharpe, vol_sharpe, sharpe_sharpe = performance(weights_sharpe, mean_returns_annual, cov_annual)

    res_minvol = minimize(vol_func, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights_minvol = res_minvol.x
    ret_minvol, vol_minvol, sharpe_minvol = performance(weights_minvol, mean_returns_annual, cov_annual)

    weights_equal = np.repeat(1 / n, n)
    ret_equal, vol_equal, sharpe_equal = performance(weights_equal, mean_returns_annual, cov_annual)

    return {
        "sharpe": (weights_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe),
        "minvol": (weights_minvol, ret_minvol, vol_minvol, sharpe_minvol),
        "equal": (weights_equal, ret_equal, vol_equal, sharpe_equal),
        "mean_returns_annual": mean_returns_annual,
        "cov_annual": cov_annual
    }

@st.cache_data(show_spinner="Calculando frontera eficiente...")
def get_efficient_frontier(mean_returns_annual, cov_annual, tickers):
    n = len(tickers)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, 1) for _ in range(n))
    
    target_returns = np.linspace(mean_returns_annual.min(), mean_returns_annual.max(), 30) # Reduced to 30 for speed
    efficient_vols, efficient_rets = [], []

    def vol_func(weights):
        return np.sqrt(weights.T @ cov_annual @ weights)

    for targ in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns_annual) - targ}
        )
        res = minimize(vol_func, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            r, v, _ = performance(res.x, mean_returns_annual, cov_annual)
            efficient_rets.append(r)
            efficient_vols.append(v)
    
    return efficient_vols, efficient_rets

def render_ui(results, years):
    data = results["data"]
    tickers = results["tickers"]
    returns = results["returns"]
    weights_recommended = results["weights_recommended"]
    best = results["best"]
    
    st.subheader(f"Tendencia de precios (últimos {years} años)")
    st.line_chart(data)
    with st.expander("📖 Interpretación – Tendencia de precios"):
        st.markdown("Este gráfico muestra la evolución histórica de los precios ajustados de cada activo...")

    st.subheader("Comparación sistemática de estrategias")
    st.dataframe(results["comparison"])
    with st.expander("📖 Interpretación – Comparación de estrategias"):
        st.markdown("Esta tabla sintetiza el desempeño de las distintas estrategias...")

    st.subheader("Volatilidad histórica móvil")
    st.line_chart(results["rolling_vol"])
    
    st.subheader("Ratio Calmar (retorno vs drawdown)")
    st.dataframe(results["df_calmar"])

    st.subheader("Ratio Sortino")
    st.dataframe(results["df_sortino"])

    st.subheader("Comportamiento en periodo de crisis (COVID 2020)")
    st.line_chart(results["crisis_data"])

    st.subheader("Comparación con benchmarks de mercado")
    st.dataframe(results["df_benchmarks"])

    st.subheader("Rendimiento acumulado: estrategias vs benchmarks")
    st.line_chart(results["comparison_cum"])

    st.subheader("Interpretación automática del mejor portafolio")
    st.dataframe(results["weighted_performance"].rename("Desempeño_Ponderado"))
    st.success(f"Portafolio recomendado: {best}")

    st.subheader("Pesos del portafolio recomendado")
    st.dataframe(weights_recommended)
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(weights_recommended["Ticker"], weights_recommended["Peso"])
    ax.set_title(f"Composición del portafolio recomendado")
    st.pyplot(fig)


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

with st.expander("❓ ¿Qué es un ticker?"):
    st.markdown("""
    Un **ticker** es el código con el que se identifica una acción en la bolsa de valores.
    **Ejemplos:** AAPL (Apple), MSFT (Microsoft), GOOGL (Google).
    """)

tickers_input = st.text_input(
    "Ingrese los tickers separados por comas (ejemplo: AAPL, MSFT, GOOGL)",
    help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma."
)

years = st.slider("Seleccione el horizonte temporal (años)", 3, 10, 6)

if st.button("Ejecutar optimización"):
    st.session_state.run_analysis = True
    st.session_state.analysis_done = False

if st.session_state.run_analysis and not st.session_state.analysis_done:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("Ingrese al menos 2 tickers.")
    else:
        try:
            data = get_crypto_data(tickers, years)
            if data is None or data.empty:
                st.error("No hay datos suficientes.")
            else:
                returns = data.pct_change().dropna()
                opt_results = run_portfolio_optimizations(returns, tickers)
                
                weights_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe = opt_results["sharpe"]
                weights_minvol, ret_minvol, vol_minvol, sharpe_minvol = opt_results["minvol"]
                weights_equal, ret_equal, vol_equal, sharpe_equal = opt_results["equal"]
                mean_returns_annual = opt_results["mean_returns_annual"]
                cov_annual = opt_results["cov_annual"]

                daily_sharpe, daily_minvol, daily_equal = returns.dot(weights_sharpe), returns.dot(weights_minvol), returns.dot(weights_equal)
                cum_sharpe, cum_minvol, cum_equal = (1 + daily_sharpe).cumprod(), (1 + daily_minvol).cumprod(), (1 + daily_equal).cumprod()
                dd_sharpe, dd_minvol, dd_equal = max_drawdown(cum_sharpe), max_drawdown(cum_minvol), max_drawdown(cum_equal)

                benchmark_data, benchmarks = get_benchmark_data(years)
                benchmark_returns = benchmark_data.pct_change().dropna()
                benchmark_cum = (1 + benchmark_returns).cumprod()

                efficient_vols, efficient_rets = get_efficient_frontier(mean_returns_annual, cov_annual, tickers)

                rolling_vol = pd.DataFrame({
                    "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                    "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                    "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
                })

                df_calmar = pd.DataFrame({
                    "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                    "Calmar": [ret_sharpe / abs(dd_sharpe), ret_minvol / abs(dd_minvol), ret_equal / abs(dd_equal)]
                })

                downside = returns.copy()
                downside[downside > 0] = 0
                downside_std = downside.std() * np.sqrt(252)
                df_sortino = pd.DataFrame({
                    "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                    "Sortino": [ret_sharpe / downside_std.dot(weights_sharpe), ret_minvol / downside_std.dot(weights_minvol), ret_equal / downside_std.dot(weights_equal)]
                })

                crisis = (cum_sharpe.index.year == 2020)
                crisis_data = pd.DataFrame({"Sharpe Máximo": cum_sharpe[crisis], "Mínima Volatilidad": cum_minvol[crisis], "Pesos Iguales": cum_equal[crisis]})

                def ann_ret(s): return (s.iloc[-1]) ** (252 / len(s)) - 1
                def ann_vol(s): return s.std() * np.sqrt(252)
                
                df_benchmarks = pd.DataFrame([{
                    "Benchmark": n, "Retorno Anual": ann_ret(benchmark_cum[t]), "Volatilidad": ann_vol(benchmark_returns[t]),
                    "Retorno Acumulado": benchmark_cum[t].iloc[-1] - 1, "Máx Drawdown": max_drawdown(benchmark_cum[t])
                } for n, t in benchmarks.items()])

                comparison_cum = pd.DataFrame({
                    "Sharpe Máximo": cum_sharpe, "Mínima Volatilidad": cum_minvol, "Pesos Iguales": cum_equal,
                    "S&P 500 (SPY)": benchmark_cum["SPY"], "Nasdaq 100 (QQQ)": benchmark_cum["QQQ"], "MSCI World (URTH)": benchmark_cum["URTH"]
                })

                df_strategies = pd.DataFrame({"Sharpe Máximo": daily_sharpe, "Mínima Volatilidad": daily_minvol, "Pesos Iguales": daily_equal})
                year_w = {y: (i + 1) for i, y in enumerate(np.sort(df_strategies.index.year.unique()))}
                w_series = df_strategies.index.year.map(year_w) / sum(year_w.values())
                weighted_perf = ((1 + df_strategies).cumprod().mul(w_series, axis=0).iloc[-1])
                best = weighted_perf.idxmax()

                final_w = weights_sharpe if best == "Sharpe Máximo" else (weights_minvol if best == "Mínima Volatilidad" else weights_equal)
                df_weights = pd.DataFrame({"Ticker": tickers, "Peso": final_w.round(2), "Peso (%)": (final_w * 100).round(2)})

                asset_summary = {t: {"retorno_anual": mean_returns_annual[t], "volatilidad": np.sqrt(cov_annual.loc[t, t])} for t in tickers}
                strategy_summary = {
                    "Sharpe Máximo": {"retorno": ret_sharpe, "volatilidad": vol_sharpe, "sharpe": sharpe_sharpe, "drawdown": dd_sharpe},
                    "Mínima Volatilidad": {"retorno": ret_minvol, "volatilidad": vol_minvol, "sharpe": sharpe_minvol, "drawdown": dd_minvol},
                    "Pesos Iguales": {"retorno": ret_equal, "volatilidad": vol_equal, "sharpe": sharpe_equal, "drawdown": dd_equal}
                }

                st.session_state.analysis_results = {
                    "data": data, "tickers": tickers, "returns": returns,
                    "comparison": pd.DataFrame({
                        "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                        "Retorno Anual": [ret_sharpe, ret_minvol, ret_equal], "Volatilidad": [vol_sharpe, vol_minvol, vol_equal],
                        "Sharpe": [sharpe_sharpe, sharpe_minvol, sharpe_equal], "Máx Drawdown": [dd_sharpe, dd_minvol, dd_equal]
                    }),
                    "rolling_vol": rolling_vol, "df_calmar": df_calmar, "df_sortino": df_sortino,
                    "crisis_data": crisis_data, "df_benchmarks": df_benchmarks, "comparison_cum": comparison_cum,
                    "weighted_performance": weighted_perf, "best": best, "weights_recommended": df_weights,
                    "weights": {"Sharpe Máximo": dict(zip(tickers, weights_sharpe)), "Mínima Volatilidad": dict(zip(tickers, weights_minvol)), "Pesos Iguales": dict(zip(tickers, weights_equal))},
                    "retornos": {"Sharpe Máximo": ret_sharpe, "Mínima Volatilidad": ret_minvol, "Pesos Iguales": ret_equal},
                    "asset_summary": asset_summary, "strategy_summary": strategy_summary,
                    "efficient_data": (efficient_vols, efficient_rets)
                }
                st.session_state.analysis_done = True
                st.session_state.run_analysis = False
                st.success("Análisis completado.")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.analysis_done:
    render_ui(st.session_state.analysis_results, years)


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
                # 🔹 NUEVO — NO BORRES NADA DE ARRIBA
                "asset_summary": asset_summary,
                "strategy_summary": strategy_summary
            }

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
        with st.chat_message("user"):        # ← LÍNEA NUEVA
            st.markdown(user_question)       # ← LÍNEA NUEVA

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
""".format(
    tickers=", ".join(results["tickers"]),
    asset_text=asset_text,
    strategy_text=strategy_text,
    best_strategy=best_strategy,
    weights_text=weights_text
)
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




































































