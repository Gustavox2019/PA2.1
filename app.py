import streamlit as st
import joblib
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Predicción Titanic",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 ¿Sobrevivirías al Titanic?")
st.write("Introduce tus datos y averigua si habrías sobrevivido usando tus modelos entrenados.")

# Función de carga definitiva utilizando joblib de forma nativa
@st.cache_resource
def load_model(model_name):
    try:
        return joblib.load(model_name)
    except Exception as e:
        st.error(f"⚠️ Error al cargar {model_name}: {e}")
        return None

# Barra lateral para seleccionar el modelo
st.sidebar.header("Configuración del Modelo")
model_option = st.sidebar.selectbox("Selecciona el modelo:", ("Logistic Regression", "Random Forest"))
model_file = "logistic_regression_model.pkl" if model_option == "Logistic Regression" else "random_forest_model.pkl"

model = load_model(model_file)

# Formulario de entrada de datos
st.header("📝 Tus Datos de Pasajero")
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Género", ["Masculino", "Femenino"])
    age = st.slider("Edad", 1, 100, 25)
    pclass = st.selectbox("Clase de Boleto (Pclass)", [1, 2, 3], index=2)

with col2:
    sibsp = st.number_input("Hermanos/Cónyuges a bordo (SibSp)", 0, 10, 0)
    parch = st.number_input("Padres/Hijos a bordo (Parch)", 0, 10, 0)
    fare = st.number_input("Precio del Boleto (Fare)", 0.0, 600.0, 30.0)

embarked = st.selectbox("Puerto de Embarque", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)

# Preprocesamiento de variables
sex_male = 1 if sex == "Masculino" else 0
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0

# Crear DataFrame estructurado
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

# Botón para ejecutar la predicción
if st.button("🔮 Calcular Supervivencia"):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            
            st.subheader("Resultado:")
            if prediction == 1:
                st.success("🎉 ¡Felicidades! Según el modelo, hubieras **SOBREVIVIDO** al desastre.")
            else:
                st.error("💔 Lamentablemente, según el modelo, **NO HUBIERAS SOBREVIVIDO**.")
                
            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Hubo un problema al procesar los datos con el modelo: {e}")
    else:
        st.warning("El modelo no está disponible actualmente debido al error de carga de arriba.")

# ==========================================
# SECCIÓN DEL PIE DE PÁGINA (FOOTER)
# ==========================================
st.markdown("<br><br><br>", unsafe_allow_html=True) # Espaciado para empujar el pie de página hacia abajo
st.markdown("---")

# Centramos el botón del enlace usando columnas
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.link_button(
        label="🚀 Ir a Google Colab", 
        url="https://colab.research.google.com/drive/1qjsKGqGPF5bIKJa_9n3DVZ3em6LyMEth?usp=sharing",
        use_container_width=True
    )
