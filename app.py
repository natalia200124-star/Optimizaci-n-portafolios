import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

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

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("Ingrese al menos 2 tickers.")
    else:
        try:
            # =====================================================================
            # 1) DESCARGA DE DATOS – HORIZONTE SELECCIONADO
            # =====================================================================
            end_date = datetime.today()
            start_date = end_date.replace(year=end_date.year - years)

            data = yf.download(tickers, start=start_date, end=end_date)["Close"]

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(0, axis=1)

            data = data[tickers]

            st.subheader("Precios de cierre (primeras filas)")
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
            # 8.5) RANKING MULTICRITERIO
            # =====================================================================
            st.subheader("Ranking multicriterio de estrategias")

            
            ranking = df_compare.copy()
            ranking["Score"] = (
                    ranking["Sharpe"].rank(ascending=False, method="average") +
                    ranking["Retorno Acumulado"].rank(ascending=False, method="average") +
                    ranking["Máx Drawdown"].rank(ascending=True, method="average")
            )

            st.dataframe(ranking[["Estrategia", "Score"]])

            st.markdown(
            """
                ### ¿Qué significa el **Score** del ranking multicriterio?

                El **Score** es un indicador **comparativo**, no un valor financiero directo.

                - A cada estrategia se le asigna una **posición relativa (ranking)** en:
                  - Retorno ajustado por riesgo (Sharpe)
                  - Retorno acumulado
                  - Máxima caída histórica (drawdown)
                - Luego, estas posiciones se **suman** para obtener el Score final.

                **Un Score más bajo indica mejor desempeño global**, porque la estrategia:
                - Rinde bien,
                - Asume un riesgo razonable,
                - Y controla mejor las caídas.

                Este ranking sirve para:
                - Validar la decisión,
                - Comparar perfiles conservadores vs agresivos,
                - Y justificar la elección desde múltiples criterios.
                """
            )

            # =====================================================================
            # 9) PESOS ÓPTIMOS
            # =====================================================================
            st.subheader("Pesos óptimos (Sharpe Máximo)")

            df_weights = pd.DataFrame({
                "Ticker": tickers,
                "Peso": weights_sharpe,
                "Peso (%)": weights_sharpe * 100
            })
            st.dataframe(df_weights)

            fig1, ax1 = plt.subplots()
            ax1.barh(df_weights["Ticker"], df_weights["Peso"])
            ax1.set_title("Composición del portafolio óptimo")
            st.pyplot(fig1)
            st.markdown(
                """
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
            # 12) INTERPRETACIÓN FINAL (AJUSTADA POR ESTRATEGIA)
            # =====================================================================
            st.subheader("Interpretación automática del mejor portafolio")

            best = df_compare.sort_values("Sharpe", ascending=False).iloc[0]["Estrategia"]

            if best == "Sharpe Máximo":
                st.markdown(
                    "### Mejor portafolio: Sharpe Máximo\n\n"
                    "Este portafolio ofrece el **mejor equilibrio entre riesgo y retorno**.\n\n"
                    "- Maximiza el retorno por cada unidad de riesgo asumido.\n"
                    "- Presenta el mejor desempeño ajustado por riesgo.\n"
                    "- Adecuado para horizontes de inversión largos."
                )
            elif best == "Mínima Volatilidad":
                st.markdown(
                    "### Mejor portafolio: Mínima Volatilidad\n\n"
                    "Prioriza estabilidad y preservación del capital.\n\n"
                    "- Reduce la exposición a grandes caídas."
                )
            else:
                st.markdown(
                    "### Mejor portafolio: Pesos Iguales\n\n"
                    "Estrategia simple y robusta."
                )

            st.success(f"Portafolio recomendado según el modelo: {best}")

            # =====================================================================
            # 13) EXPORTACIÓN A EXCEL
            # =====================================================================
            export_df = pd.DataFrame({"Fecha": data.index})

            for t in tickers:
                export_df[f"Precio_{t}"] = data[t].values
                export_df[f"Retorno_{t}"] = returns[t].reindex(export_df["Fecha"]).values
                export_df[f"Acumulado_{t}"] = cumulative_assets[t].reindex(
                    export_df["Fecha"]
                ).values

            export_df["Sharpe"] = cum_sharpe.reindex(export_df["Fecha"]).values
            export_df["MinVol"] = cum_minvol.reindex(export_df["Fecha"]).values
            export_df["Equal"] = cum_equal.reindex(export_df["Fecha"]).values

            with pd.ExcelWriter("resultados_portafolio.xlsx", engine="openpyxl") as writer:
                export_df.to_excel(writer, sheet_name="Datos_Completos", index=False)
                df_weights.to_excel(writer, sheet_name="Pesos", index=False)
                df_compare.to_excel(writer, sheet_name="Comparación", index=False)
                pd.DataFrame({
                    "Retorno": efficient_rets,
                    "Volatilidad": efficient_vols
                }).to_excel(writer, sheet_name="Frontera", index=False)

            st.success("Archivo Excel generado correctamente")

        except Exception as e:
            st.error(f"Error: {e}")









