import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# =============================================================================
# Configuración general
# =============================================================================
st.set_page_config(
    page_title="Clasificador de Diabetes - Solemne 2",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .small-note {
        font-size: 0.88rem;
        color: #666666;
    }
    .box {
        padding: 1rem;
        border-radius: 0.75rem;
        background-color: #f6f8fa;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Funciones auxiliares
# =============================================================================
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Outcome"

COLS_CEROS_INVALIDOS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

NOMBRES_VARIABLES = {
    "Pregnancies": "Embarazos",
    "Glucose": "Glucosa",
    "BloodPressure": "Presión arterial",
    "SkinThickness": "Grosor de piel",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Función pedigree diabetes",
    "Age": "Edad",
}


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """Carga el archivo diabetes.csv desde la misma carpeta de la app."""
    ruta = Path(__file__).parent / "diabetes.csv"
    if not ruta.exists():
        st.error(
            "No se encontró el archivo diabetes.csv. "
            "Déjalo en la misma carpeta que app_solemne2.py."
        )
        st.stop()

    df = pd.read_csv(ruta)
    return df


@st.cache_resource
def entrenar_modelo(df: pd.DataFrame):
    """Entrena el clasificador usado para visualizar y probar el modelo."""
    df_model = df.copy()

    # En el dataset Pima, algunos ceros no tienen sentido clínico.
    # Se tratan como datos faltantes, igual que en el notebook de la Solemne 1.
    for col in COLS_CEROS_INVALIDOS:
        df_model[col] = df_model[col].replace(0, np.nan)

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    modelo = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob),
    }

    matriz = confusion_matrix(y_test, y_pred)

    clf = modelo.named_steps["clf"]
    importancias = pd.DataFrame(
        {
            "Variable": [NOMBRES_VARIABLES[col] for col in FEATURES],
            "Importancia": clf.feature_importances_,
        }
    ).sort_values("Importancia", ascending=True)

    return modelo, metricas, matriz, importancias, X_test, y_test


def preparar_caso_usuario(datos: dict) -> pd.DataFrame:
    """Convierte los datos ingresados por pantalla a DataFrame para predecir."""
    caso = pd.DataFrame([datos], columns=FEATURES)

    # Si el usuario ingresa 0 en variables donde no tiene sentido clínico,
    # se trata como dato faltante y el pipeline lo imputará con mediana.
    for col in COLS_CEROS_INVALIDOS:
        caso[col] = caso[col].replace(0, np.nan)

    return caso


# =============================================================================
# Carga y entrenamiento
# =============================================================================
df = cargar_datos()
modelo, metricas, matriz, importancias, X_test, y_test = entrenar_modelo(df)

# =============================================================================
# Encabezado
# =============================================================================
st.title("Clasificador de riesgo de diabetes")

st.caption(
    "Solemne 2 · Taller de Aplicaciones · Visualización interactiva del clasificador"
)

st.markdown(
    """
    Esta aplicación presenta los principales resultados del clasificador desarrollado sobre el dataset de diabetes
    y permite probar el modelo ingresando manualmente los datos de una persona.
    """
)

st.info(
    """
    Modelo utilizado: **Árbol de Decisión**  

    El clasificador se presenta mediante métricas de desempeño, matriz de confusión,
    variables relevantes y una prueba interactiva con nuevos datos.
    """
)

st.warning(
    "Nota: esta aplicación tiene fines académicos. La predicción corresponde a una salida del modelo y no constituye diagnóstico médico."
)

# =============================================================================
# Sección 1: Métricas del clasificador
# =============================================================================
st.header("1. Resultados principales del clasificador")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{metricas['Accuracy']:.2%}")
col2.metric("Precision", f"{metricas['Precision']:.2%}")
col3.metric("Recall", f"{metricas['Recall']:.2%}")
col4.metric("F1-score", f"{metricas['F1-score']:.2%}")
col5.metric("AUC", f"{metricas['AUC']:.2%}")

st.markdown(
    """
    <div class="small-note">
    Estas métricas resumen las bondades del clasificador. En este problema, el <b>Recall</b> es especialmente relevante,
    porque indica qué proporción de casos reales de diabetes fueron identificados correctamente por el modelo.
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Sección 2: Matriz de confusión e importancia de variables
# =============================================================================
st.header("2. Matriz de confusión e importancia de variables")

col_matriz, col_importancia = st.columns([1, 1])

with col_matriz:
    st.subheader("Matriz de confusión")

    labels = ["No diabetes", "Diabetes"]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=["Predicción: No diabetes", "Predicción: Diabetes"],
            y=["Real: No diabetes", "Real: Diabetes"],
            text=matriz,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig_cm.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    tn, fp, fn, tp = matriz.ravel()
    st.markdown(
        f"""
        - Verdaderos negativos: **{tn}**
        - Falsos positivos: **{fp}**
        - Falsos negativos: **{fn}**
        - Verdaderos positivos: **{tp}**
        """
    )

with col_importancia:
    st.subheader("Variables más relevantes")

    fig_imp = px.bar(
        importancias,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables en el Árbol de Decisión",
    )
    fig_imp.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Importancia",
        yaxis_title="Variable",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# Sección 3: Probar el clasificador
# =============================================================================
st.header("3. Probar el modelo con nuevos datos")

st.markdown(
    "Ingrese los datos de una persona y presione **Clasificar** para obtener la predicción del modelo."
)

st.caption(
    "El formulario utiliza distintos controles interactivos para probar el modelo. "
    "Las variables clínicas que no estén disponibles pueden dejarse en 0 o marcarse como desconocidas; "
    "el modelo las tratará como valores faltantes cuando corresponda. "
    "El índice de antecedentes familiares se controla mediante un deslizador y utiliza 0.45 como valor de referencia."
)

with st.container(border=True):
    col_general, col_principal, col_opcional = st.columns(3)

    with col_general:
        st.markdown("**Datos generales**")

        pregnancies = st.selectbox(
            "Embarazos (cantidad)",
            options=list(range(0, 16)),
            index=2,
            help="Cantidad de embarazos registrados. Seleccione 0 si no ha tenido embarazos.",
        )

        age = st.slider(
            "Edad (años)",
            min_value=18,
            max_value=100,
            value=38,
            step=1,
            help="Edad de la persona en años.",
        )

        dpf = st.slider(
            "Índice antecedentes familiares",
            min_value=0.01,
            max_value=3.0,
            value=0.45,
            step=0.01,
            help="Corresponde a DiabetesPedigreeFunction. Es un índice del dataset asociado a antecedentes familiares/genéticos. Ejemplo de referencia: 0.45.",
        )

    with col_principal:
        st.markdown("**Indicadores principales**")

        glucose = st.slider(
            "Glucosa (mg/dL aprox.)",
            min_value=0,
            max_value=250,
            value=120,
            step=1,
            help="Glucosa plasmática. Si no se conoce, puede dejar 0 para que sea imputada por el modelo.",
        )

        bmi = st.slider(
            "IMC (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=30.0,
            step=0.1,
            help="Índice de masa corporal. Si no se conoce, puede dejar 0 para imputar.",
        )

    with col_opcional:
        st.markdown("**Variables clínicas opcionales**")
    
        blood_pressure_input = st.number_input(
            "Presión arterial diastólica (mmHg)",
            min_value=0,
            max_value=140,
            value=70,
            step=1,
            disabled=st.session_state.get("bp_desconocida", False),
            help="Presión arterial diastólica. Si no se conoce, marque la casilla inferior.",
            key="blood_pressure_input",
        )
        bp_desconocida = st.checkbox(
            "No conozco presión arterial", 
            key="bp_desconocida",
        )
        blood_pressure = 0 if bp_desconocida else blood_pressure_input
    
        skin_thickness_input = st.number_input(
            "Grosor de piel (mm)",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
            disabled=st.session_state.get("skin_desconocido", False),
            help="Pliegue cutáneo del tríceps en mm. Si no se conoce, marque la casilla inferior.",
            key="skin_thickness_input",
        )
        skin_desconocido = st.checkbox(
            "No conozco grosor de piel", 
            key="skin_desconocido",
        )
        skin_thickness = 0 if skin_desconocido else skin_thickness_input
    
        insulin_input = st.number_input(
            "Insulina sérica",
            min_value=0,
            max_value=900,
            value=80,
            step=1,
            disabled=st.session_state.get("insulin_desconocida", False),
            help="Insulina sérica medida a las 2 horas. Si no se conoce, marque la casilla inferior.",
            key="insulin_input",
        )
        insulin_desconocida = st.checkbox(
            "No conozco insulina", 
            key="insulin_desconocida",
        )
        insulin = 0 if insulin_desconocida else insulin_input

    clasificar = st.button("Clasificar")

if clasificar:
    datos_usuario = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    caso_usuario = preparar_caso_usuario(datos_usuario)
    prediccion = int(modelo.predict(caso_usuario)[0])
    probabilidad = float(modelo.predict_proba(caso_usuario)[0, 1])

    st.subheader("Resultado de la clasificación")

    if prediccion == 1:
        st.error(
            f"""
            **Predicción del modelo:** Diabetes / mayor riesgo estimado 
    
            **Recomendación:** este caso debería ser priorizado para evaluación preventiva o revisión clínica,
            especialmente si presenta valores altos de glucosa, IMC o antecedentes familiares.
            """,
            icon="⚠️"
        )
    else:
        st.success(
            f"""
            **Predicción del modelo:** No diabetes / menor riesgo estimado 
    
            **Recomendación:** el modelo no clasifica este caso como de alto riesgo. De todas formas,
            se recomienda mantener seguimiento preventivo y monitoreo de factores de riesgo.
            """,
            icon="✅"
        )
        
    st.caption(
        "La predicción corresponde a una estimación del modelo académico y no debe interpretarse como diagnóstico médico."
    )

    with st.expander("Ver datos ingresados"):
        st.dataframe(caso_usuario.rename(columns=NOMBRES_VARIABLES), use_container_width=True)

# =============================================================================
# Sección 4: Datos usados
# =============================================================================
with st.expander("Ver resumen del dataset utilizado"):
    st.write("Dimensión del dataset:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    distribucion = df[TARGET].value_counts().rename(index={0: "No diabetes", 1: "Diabetes"})
    st.write("Distribución de la variable objetivo:")
    st.dataframe(distribucion.to_frame("Cantidad"), use_container_width=True)
