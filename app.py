import streamlit as st
import pickle
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Predicción Titanic",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 ¿Sobrevivirías al Titanic?")

# Función de carga nativa con control de errores total
@st.cache_resource
def load_model(model_name):
    try:
        with open(model_name, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # Esto evitará que la app colapse y te dirá exactamente qué falla en pantalla
        st.error(f"Error al cargar {model_name}: {e}")
        return None

# Barra lateral
st.sidebar.header("Modelo")
model_option = st.sidebar.selectbox("Selecciona el modelo:", ("Logistic Regression", "Random Forest"))
model_file = "logistic_regression_model.pkl" if model_option == "Logistic Regression" else "random_forest_model.pkl"

model = load_model(model_file)

# Formulario
st.header("📝 Tus Datos")
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Género", ["Masculino", "Femenino"])
    age = st.slider("Edad", 1, 100, 25)
    pclass = st.selectbox("Clase", [1, 2, 3], index=2)
with col2:
    sibsp = st.number_input("Hermanos/Cónyuges (SibSp)", 0, 10, 0)
    parch = st.number_input("Padres/Hijos (Parch)", 0, 10, 0)
    fare = st.number_input("Precio Boleto (Fare)", 0.0, 600.0, 30.0)

embarked = st.selectbox("Puerto", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)

# Preprocesamiento
sex_male = 1 if sex == "Masculino" else 0
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0

input_data = pd.DataFrame([{
    'Pclass': pclass, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 
    'Fare': fare, 'Sex_male': sex_male, 'Embarked_Q': embarked_q, 'Embarked_S': embarked_s
}])

st.markdown("---")

if st.button("🔮 Calcular Supervivencia"):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.success("🎉 ¡Sobreviviste!")
            else:
                st.error("💔 No sobreviviste.")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
    else:
        st.warning("El modelo no está cargado.")
