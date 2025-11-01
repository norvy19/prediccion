# =====================================================
# 🩺 APP STREAMLIT - PREDICCIÓN DE DIABETES
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Título y descripción general
# =====================================================
st.title("🧠 Predicción de Diabetes")


# =====================================================
# CARGA DE MODELOS
# =====================================================
modelo_lr = joblib.load("modelo_diabetes.pkl")
modelo_rf = joblib.load("modelo_rf_diabetes.joblib")
modelo_xgb = joblib.load("modelo_xgb_diabetes.pkl")

# =====================================================
# LÍMITES DE VARIABLES
# =====================================================
limites = {
    'Edad': (19, 90),
    'Glucosa_en_ayunas': (75, 147),
    'Glucosa_postprandial': (76, 244),
    'Hemoglobina_glicosilada_(HbA1c)': (4.32, 8.72),
    'Puntaje_riesgo_diabetes': (6.1, 53.3)
}

vars_minmax = ['Glucosa_en_ayunas', 'Glucosa_postprandial', 'Hemoglobina_glicosilada_(HbA1c)']
vars_std = ['Edad', 'Puntaje_riesgo_diabetes']

# =====================================================
# INFORMACIÓN DE LOS MODELOS
# =====================================================
st.sidebar.header("📘 Información de los Modelos")

model_info = {
    "Regresión Logística": "📊 Modelo estadístico interpretable que estima la probabilidad de diabetes según la relación lineal entre las variables.",
    "Random Forest": "🌳 Conjunto de árboles de decisión que mejora la precisión combinando múltiples predictores y reduciendo el sobreajuste.",
    "XGBoost": "⚡ Modelo avanzado de boosting que optimiza errores previos y ofrece alta precisión en clasificación médica."
}

modelo_seleccionado = st.selectbox(
    "🧩 Selecciona el modelo para la predicción:",
    list(model_info.keys())
)

st.sidebar.info(model_info[modelo_seleccionado])

# =====================================================
# ENTRADAS DEL USUARIO
# =====================================================
st.markdown("### 🧍‍ Datos del paciente")

entrada = {}

entrada['Edad'] = st.number_input("Ingrese Edad [19 - 90]", 19, 90, 50, step=1)

antecedentes = st.selectbox("¿Tiene antecedentes familiares de diabetes?", ("No", "Sí"))
entrada['Antecedentes_familiares_diabetes'] = 0 if antecedentes == "No" else 1

entrada['Glucosa_en_ayunas'] = st.number_input("Ingrese Glucosa en ayunas [75 - 147]", 75.0, 147.0, 111.0, step=0.1, format="%.2f")
entrada['Glucosa_postprandial'] = st.number_input("Ingrese Glucosa postprandial [76 - 244]", 76.0, 244.0, 160.0, step=0.1, format="%.2f")
entrada['Hemoglobina_glicosilada_(HbA1c)'] = st.number_input("Ingrese Hemoglobina glicosilada (HbA1c) [4.32 - 8.72]", 4.32, 8.72, 6.52, step=0.01, format="%.2f")
entrada['Puntaje_riesgo_diabetes'] = st.number_input("Ingrese Puntaje de riesgo de diabetes [6.1 - 53.3]", 6.1, 53.3, 30.2, step=0.1, format="%.2f")

# =====================================================
# BOTÓN DE PREDICCIÓN
# =====================================================
if st.button("🔍 Predecir Diabetes"):
    df_input = pd.DataFrame([entrada])

    # Escalado MinMax
    min_max_values = {
        'Glucosa_en_ayunas': (75, 147),
        'Glucosa_postprandial': (76, 244),
        'Hemoglobina_glicosilada_(HbA1c)': (4.32, 8.72)
    }
    for var in vars_minmax:
        df_input[var] = (df_input[var] - min_max_values[var][0]) / (min_max_values[var][1] - min_max_values[var][0])

    # Escalado estándar
    mean_std_values = {
        'Edad': (50.19, 15.49),
        'Puntaje_riesgo_diabetes': (30.20, 9.00)
    }
    for var in vars_std:
        df_input[var] = (df_input[var] - mean_std_values[var][0]) / mean_std_values[var][1]

    # Selección del modelo
    if modelo_seleccionado == "Regresión Logística":
        modelo = modelo_lr
    elif modelo_seleccionado == "Random Forest":
        modelo = modelo_rf
    else:
        modelo = modelo_xgb

    # Predicción
    pred = modelo.predict(df_input)[0]
    prob = modelo.predict_proba(df_input)[0][1]

    # =====================================================
    # RESULTADO DE LA PREDICCIÓN
    # =====================================================
    st.subheader("📈 Resultado de la predicción")
    st.write(f"**Probabilidad estimada de diabetes:** {prob:.2f}")

    if prob > 0.6:
        st.error("⚠️ **Riesgo alto de diabetes.** Se recomienda una revisión médica y control clínico.")
    elif prob > 0.3:
        st.warning("🟠 **Riesgo moderado.** Mantén hábitos saludables y chequeos regulares.")
    else:
        st.success("🟢 **Riesgo bajo.** Mantén un estilo de vida saludable.")

    # =====================================================
    # 📊 COMPARACIÓN DE VALORES
    # =====================================================
    st.markdown("### 📊 Comparación de tus valores con los promedios del conjunto de datos")

    promedios = {
        "Edad": 50.19,
        "Glucosa_en_ayunas": 111.12,
        "Glucosa_postprandial": 160.0,
        "Hemoglobina_glicosilada_(HbA1c)": 6.52,
        "Puntaje_riesgo_diabetes": 30.20
    }

    etiquetas = list(promedios.keys())
    valores_usuario = [
        entrada["Edad"],
        entrada["Glucosa_en_ayunas"],
        entrada["Glucosa_postprandial"],
        entrada["Hemoglobina_glicosilada_(HbA1c)"],
        entrada["Puntaje_riesgo_diabetes"]
    ]
    valores_promedio = list(promedios.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(etiquetas, valores_promedio, alpha=0.4, label='Promedio población')
    ax.bar(etiquetas, valores_usuario, alpha=0.7, label='Tus valores')
    ax.legend()
    ax.set_ylabel('Valor')
    plt.xticks(rotation=20)
    st.pyplot(fig)

    # =====================================================
    # 🩺 MEDIDOR DE RIESGO
    # =====================================================
    st.markdown("### 🩺 Nivel de riesgo estimado")

    fig, ax = plt.subplots(figsize=(6, 1))
    color = "red" if prob > 0.6 else "orange" if prob > 0.3 else "green"
    ax.barh(0, prob, color=color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Probabilidad de Diabetes")
    st.pyplot(fig)

    # =====================================================
    # 🩻 INTERPRETACIÓN FINAL
    # =====================================================
    st.markdown("### 🩻 Interpretación del resultado")
    if pred == 1:
        st.write("El modelo indica un **riesgo elevado** de diabetes. Se recomienda realizar una evaluación médica detallada, "
                 "controlar los niveles de glucosa y mejorar los hábitos alimenticios y de ejercicio.")
    else:
        st.write("El modelo sugiere un **bajo riesgo** de diabetes, aunque es recomendable mantener chequeos médicos regulares "
                 "y un estilo de vida saludable.")
