import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Supervivencia - Titanic",
    page_icon="🚢",
    layout="centered"
)

# Título de la aplicación
st.title("🚢 ¿Sobrevivirías al hundimiento del Titanic?")
st.write("Introduce tus datos y averigua si habrías sobrevivido usando tus modelos entrenados.")

# Función para cargar los modelos de manera segura usando caché
@st.cache_resource
def load_model(model_name):
    try:
        with open(model_name, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{model_name}'. Asegúrate de subirlo a tu repositorio de GitHub.")
        return None

# Selector de modelo en la barra lateral
st.sidebar.header("Configuración del Modelo")
model_option = st.sidebar.selectbox(
    "Selecciona el modelo de Machine Learning:",
    ("Logistic Regression", "Random Forest")
)

# Asignar archivo según selección
model_file = "logistic_regression_model.pkl" if model_option == "Logistic Regression" else "random_forest_model.pkl"
model = load_model(model_file)

# Formulario de entrada de datos del usuario
st.header("📝 Introduce tus Datos de Pasajero")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Género", ["Masculino", "Femenino"])
    age = st.slider("Edad", min_value=1, max_value=100, value=25)
    pclass = st.selectbox("Clase de Boleto (Pclass)", [1, 2, 3], index=2, 
                          help="1 = Primera clase (Alta), 2 = Segunda clase (Media), 3 = Tercera clase (Baja)")

with col2:
    sibsp = st.number_input("Número de hermanos / cónyuges a bordo (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Número de padres / hijos a bordo (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Precio del boleto pagado (Fare)", min_value=0.0, max_value=600.0, value=30.0)

embarked = st.selectbox("Puerto de Embarque", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)

# --- Preprocesamiento de los datos ingresados ---
# Convertir Género a Sex_male
sex_male = 1 if sex == "Masculino" else 0

# Convertir Puerto de Embarque a variables One-Hot (Embarked_Q, Embarked_S)
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0

# Crear el DataFrame con las características exactas que los modelos esperan
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_q,
    'Embarked_S': embarked_s
}])

st.markdown("---")

# Botón para realizar la predicción
if st.button("🔮 Calcular Supervivencia"):
    if model is not None:
        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        
        # Obtener las probabilidades (si el modelo lo permite)
        try:
            probabilities = model.predict_proba(input_data)[0]
            prob_survival = probabilities[1] * 100
            prob_death = probabilities[0] * 100
        except AttributeError:
            prob_survival = None

        # Mostrar resultados
        st.subheader("Resultado de la predicción:")
        if prediction == 1:
            st.success("🎉 ¡Felicidades! Según el modelo, hubieras **SOBREVIVIDO** al desastre.")
            if prob_survival is not None:
                st.write(f"Probabilidad de supervivencia: **{prob_survival:.2f}%**")
        else:
            st.error("💔 Lamentablemente, según el modelo, **NO HUBIERAS SOBREVIVIDO**.")
            if prob_survival is not None:
                st.write(f"Probabilidad de fallecimiento: **{prob_death:.2f}%**")
                
        # Detalle técnico opcional
        with st.expander("Ver datos enviados al modelo"):
            st.dataframe(input_data)
