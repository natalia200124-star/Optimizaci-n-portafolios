import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

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

# =========================
# SELECTORES (SIMULADOR)
# =========================
tickers_input = st.text_input(
    "Ingrese los tickers separados por comas (ej: AAPL,MSFT,GOOGL):"
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
            # 1) DESCARGA DE DATOS – HORIZONTE SELECCIONADO
            # =====================================================================
            end_date = datetime.today()
            start_date = end_date.replace(year=end_date.year - years)

            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True
            )["Close"]

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(0, axis=1)

            data = data.ffill()

            # =====================================================================
            # 1.5) DESCARGA Y DEPURACIÓN DE DATOS (SIN LOOK-AHEAD BIAS)
            # =====================================================================
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
                return -(r / v)

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

            st.markdown(
                """
                **Cómo interpretar esta tabla:**
                - **Retorno acumulado:** cuánto creció el capital total en el periodo.
                - **Volatilidad:** magnitud de las fluctuaciones (riesgo).
                - **Sharpe:** eficiencia riesgo–retorno.
                - **Máx Drawdown:** peor caída histórica desde un máximo.
                """
            )

            # =====================================================================
            # 8.1) VOLATILIDAD HISTÓRICA ROLLING (RIESGO DINÁMICO)
            # =====================================================================
            st.subheader("Volatilidad histórica móvil (12 meses)")

            rolling_vol = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
            })

            st.line_chart(rolling_vol)

            st.markdown(
                """
                **Interpretación:**
                Esta gráfica muestra cómo el riesgo **cambia en el tiempo**.
                - Picos altos suelen coincidir con periodos de crisis.
                - Estrategias más estables presentan curvas más suaves.
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

            st.markdown(
                """
                **Interpretación:**
                - Un **Calmar alto** indica buen retorno con caídas controladas.
                - Muy utilizado por perfiles conservadores y análisis profesional.
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

            st.markdown(
                """
                **Interpretación:**
                El Sortino penaliza solo la volatilidad negativa,
                ofreciendo una visión más realista del riesgo para el inversor.
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

            st.markdown(
                """
                **Interpretación:**
                Permite observar:
                - Qué estrategia cayó menos.
                - Cuál se recuperó más rápido tras la crisis.
                """
            )

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

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS ACUMULADOS
            # =====================================================================
            st.subheader("Retornos diarios de los activos")
            st.line_chart(returns)

            st.markdown(
                """
                **Interpretación:**
                - Cada línea representa el retorno porcentual diario de una acción.
                - Picos indican periodos de alta volatilidad.
                """
            )
            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS POR ACTIVO
            # =====================================================================

            st.subheader("Retornos diarios por activo")

            for ticker in returns.columns:
                  st.markdown(f"### {ticker}")
                  st.line_chart(returns[[ticker]])
            # =====================================================================
            # 11) FRONTERA EFICIENTE (MEJORADA CON ETIQUETAS)
            # =====================================================================
            st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

            fig2, ax2 = plt.subplots(figsize=(8, 6))

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

            # --- Gráfico ---
            fig, ax = plt.subplots()
            ax.barh(df_weights["Ticker"], df_weights["Peso"])
            ax.set_title(f"Composición del portafolio recomendado\n({metodo})")
            st.pyplot(fig)

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

            st.session_state.analysis_done = True

            st.success("Análisis del portafolio ejecutado correctamente")

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
                }
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


st.divider()
st.subheader("🤖 Asistente inteligente del portafolio")

if not st.session_state.analysis_done:
    st.info("Ejecuta primero la optimización para habilitar el asistente.")
else:
    import os
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("El asistente requiere una API Key válida de OpenAI.")
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

            # Obtener pesos óptimos del portafolio recomendado
            best_strategy = results["best"]
            weights_dict = results["weights"][best_strategy]

            weights_text = "\n".join(
                [f"- {ticker}: {weight:.2%}"
                for ticker, weight in weights_dict.items()]
            )

            system_prompt = f"""
            Eres un analista financiero profesional.

            Activos analizados:
            {', '.join(results['tickers'])}

            Portafolio recomendado:
            {results['best']}

            Pesos óptimos del portafolio recomendado:
            {weights_text}

            Reglas estrictas:
            - Usa EXCLUSIVAMENTE esta información.
            - Si el usuario pregunta cómo invertir un monto específico, 
              calcula los valores usando estos pesos.
            - No inventes datos.
            - Explica de forma clara para usuarios no técnicos.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.chat_messages
                ],
                temperature=0.3
            )

            answer = response.choices[0].message.content

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)









