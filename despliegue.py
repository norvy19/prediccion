import streamlit as st
import pandas as pd
import joblib

# 🔹 Cargar modelos previamente entrenados
modelo_lr = joblib.load('modelo_diabetes.pkl')
modelo_rf = joblib.load('modelo_rf_diabetes.joblib')
modelo_xgb = joblib.load('modelo_xgb_diabetes.pkl')

# 🔹 Definir los límites de cada variable según tu dataset
limites = {
    'Edad': (19, 90),
    'Glucosa_en_ayunas': (75, 147),
    'Glucosa_postprandial': (76, 244),
    'Hemoglobina_glicosilada_(HbA1c)': (4.32, 8.72),
    'Puntaje_riesgo_diabetes': (6.1, 53.3)
}

# 🔹 Variables por tipo de escalado
vars_minmax = ['Glucosa_en_ayunas', 'Glucosa_postprandial', 'Hemoglobina_glicosilada_(HbA1c)']
vars_std = ['Edad', 'Puntaje_riesgo_diabetes']

st.title("Predicción de Diabetes")

# 🔹 Selección del modelo
modelo_seleccionado = st.selectbox(
    "Selecciona el modelo para la predicción:",
    ("Regresión Logística", "Random Forest", "XGBoost")
)

# 🔹 Inputs del usuario
entrada = {}

# 🔹 Inputs del usuario
entrada = {}

# Edad (int)
entrada['Edad'] = st.number_input(
    "Ingrese Edad [19 - 90]",
    min_value=19,
    max_value=90,
    value=50,  # valor inicial
    step=1
)

# Antecedentes familiares de diabetes (0 = No, 1 = Sí)
antecedentes = st.selectbox(
    "¿Tiene antecedentes familiares de diabetes?",
    ("No", "Sí")
)
entrada['Antecedentes_familiares_diabetes'] = 0 if antecedentes == "No" else 1

# Glucosa en ayunas (float)
entrada['Glucosa_en_ayunas'] = st.number_input(
    "Ingrese Glucosa en ayunas [75 - 147]",
    min_value=75.0,
    max_value=147.0,
    value=111.0,
    step=0.1,
    format="%.2f"
)

# Glucosa postprandial (float)
entrada['Glucosa_postprandial'] = st.number_input(
    "Ingrese Glucosa postprandial [76 - 244]",
    min_value=76.0,
    max_value=244.0,
    value=160.0,
    step=0.1,
    format="%.2f"
)

# Hemoglobina glicosilada (HbA1c) (float)
entrada['Hemoglobina_glicosilada_(HbA1c)'] = st.number_input(
    "Ingrese Hemoglobina glicosilada (HbA1c) [4.32 - 8.72]",
    min_value=4.32,
    max_value=8.72,
    value=6.52,
    step=0.01,
    format="%.2f"
)

# Puntaje de riesgo de diabetes (float)
entrada['Puntaje_riesgo_diabetes'] = st.number_input(
    "Ingrese Puntaje de riesgo de diabetes [6.1 - 53.3]",
    min_value=6.1,
    max_value=53.3,
    value=30.2,
    step=0.1,
    format="%.2f"
)

# 🔹 Botón para predecir
if st.button("Predecir Diabetes"):
    df_input = pd.DataFrame([entrada])

    # 🔹 Escalado
    # MinMax
    min_max_values = {
        'Glucosa_en_ayunas': (75, 147),
        'Glucosa_postprandial': (76, 244),
        'Hemoglobina_glicosilada_(HbA1c)': (4.32, 8.72)
    }
    for var in vars_minmax:
        df_input[var] = (df_input[var] - min_max_values[var][0]) / (min_max_values[var][1] - min_max_values[var][0])

    # Standard
    mean_std_values = {
        'Edad': (50.19, 15.49),
        'Puntaje_riesgo_diabetes': (30.20, 9.00)
    }
    for var in vars_std:
        df_input[var] = (df_input[var] - mean_std_values[var][0]) / mean_std_values[var][1]

    # 🔹 Selección del modelo
    if modelo_seleccionado == "Regresión Logística":
        modelo = modelo_lr
    elif modelo_seleccionado == "Random Forest":
        modelo = modelo_rf
    else:
        modelo = modelo_xgb

    # 🔹 Predicción
    pred = modelo.predict(df_input)[0]
    prob = modelo.predict_proba(df_input)[0][1]

    # 🔹 Mostrar resultado
    if pred == 1:
        st.error(f"Predicción: La persona tiene riesgo de diabetes. Probabilidad: {prob:.2f}")
    else:
        st.success(f"Predicción: La persona NO tiene riesgo de diabetes. Probabilidad: {prob:.2f}")
