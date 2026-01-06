import streamlit as st
from pathlib import Path
from src.dsproject.predict import load_model, predict_single
import pandas as pd

MODEL_PATH = Path("artifacts/model/model.joblib")

# ----- Page Configuration -----
st.set_page_config(
    page_title="🌸 Iris Classifier",
    page_icon="🌸",
    layout="wide",
)

# ----- Header -----
st.title("🌸 **Iris Classifier**")
st.markdown(
    "<p style='color:gray'>Train the model with <code>python scripts/train.py</code> "
    "and predict below with live visuals.</p>",
    unsafe_allow_html=True,
)

# ----- Input Section -----
st.markdown("### 🔢 Input Flower Measurements")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 0.0, 10.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  0.0, 10.0, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 0.0, 10.0, 1.4, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.0, 10.0, 0.2, 0.1)

# Image mapping for each species
image_map = {
    0: "images/setosa.png",
    1: "images/versicolor.png",
    2: "images/virginica.png",
}

# ----- Prediction -----
if not MODEL_PATH.exists():
    st.warning("⚠️ Model not found. Train first: `python scripts/train.py`")
else:
    model = load_model(str(MODEL_PATH))
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        out = predict_single(model, [sepal_length, sepal_width, petal_length, petal_width])

        label_map = {0: "Setosa 🌱", 1: "Versicolor 🌿", 2: "Virginica 🌺"}
        pred_idx = out["prediction"]
        pred_label = label_map[pred_idx]

        st.success(f"### ✅ Prediction: **{pred_label}**")

        # ----- Show Flower Image -----
        img_path = image_map.get(pred_idx)
        if img_path and Path(img_path).exists():
            st.image(img_path, caption=f"Predicted: {pred_label}", use_container_width=True)
        else:
            st.info("Add images in the 'images' folder to display flower photos.")

        # ----- Probability Chart -----
        if out.get("proba"):
            st.markdown("### 📊 Class Probabilities")
            probs = pd.DataFrame(
                [out["proba"]],
                columns=["Setosa", "Versicolor", "Virginica"]
            )
            st.bar_chart(probs.T)

# ----- Footer -----
st.markdown(
    "<hr style='border:1px solid #eee'>"
    "<p style='text-align:center;color:gray'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
