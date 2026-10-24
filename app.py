
import streamlit as st
import requests

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎗️ Breast Cancer Prediction")
st.write(
    "Enter the cell measurements below and submit them "
    "to your trained Machine Learning model."
)

st.info(
    "Educational project only. Predictions are not a medical "
    "diagnosis and should not guide medical decisions."
)

# -----------------------------
# Feature Names
# Keep these identical to main.py
# -----------------------------
FEATURES = [
    "radius1", "texture1", "perimeter1", "area1",
    "smoothness1", "compactness1", "concavity1",
    "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2",
    "smoothness2", "compactness2", "concavity2",
    "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3",
    "smoothness3", "compactness3", "concavity3",
    "concave_points3", "symmetry3", "fractal_dimension3"
]

# Example values from a Malignant-class sample.
# These are for software testing, not diagnosis.
DEFAULTS = {
    "radius1": 17.99,
    "texture1": 10.38,
    "perimeter1": 122.8,
    "area1": 1001.0,
    "smoothness1": 0.1184,
    "compactness1": 0.2776,
    "concavity1": 0.3001,
    "concave_points1": 0.1471,
    "symmetry1": 0.2419,
    "fractal_dimension1": 0.07871,
    "radius2": 1.095,
    "texture2": 0.9053,
    "perimeter2": 8.589,
    "area2": 153.4,
    "smoothness2": 0.006399,
    "compactness2": 0.04904,
    "concavity2": 0.05373,
    "concave_points2": 0.01587,
    "symmetry2": 0.03003,
    "fractal_dimension2": 0.006193,
    "radius3": 25.38,
    "texture3": 17.33,
    "perimeter3": 184.6,
    "area3": 2019.0,
    "smoothness3": 0.1622,
    "compactness3": 0.6656,
    "concavity3": 0.7119,
    "concave_points3": 0.2654,
    "symmetry3": 0.4601,
    "fractal_dimension3": 0.1189
}

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):
    st.subheader("Cell Measurements")

    st.caption("Group 1: Mean values")
    st.caption("Group 2: Standard error values")
    st.caption("Group 3: Worst/largest values")

    columns = st.columns(3)
    input_values = {}

    for group_index, col in enumerate(columns):
        start = group_index * 10
        group_features = FEATURES[start:start + 10]

        with col:
            st.markdown(f"### Feature Group {group_index + 1}")

            for feature in group_features:
                input_values[feature] = st.number_input(
                    label=feature,
                    min_value=0.0,
                    value=float(DEFAULTS[feature]),
                    format="%.6f",
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Send Data to FastAPI
# -----------------------------
if submitted:
    try:
        with st.spinner("Connecting to prediction API..."):
            response = requests.post(
                API_URL,
                json=input_values,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            label = result.get("prediction", "")
            result_name = result.get("result", "Unknown")

            if label == "B":
                st.success(f"Prediction: {result_name} (B)")
            elif label == "M":
                st.warning(f"Prediction: {result_name} (M)")
            else:
                st.write(f"Prediction: {result_name}")

            st.write(result.get("message", ""))

            probabilities = result.get("probabilities", {})

            if probabilities:
                st.subheader("Model Probabilities")

                col1, col2 = st.columns(2)

                p0 = float(probabilities.get("0", 0))
                p1 = float(probabilities.get("1", 0))

                col1.metric("Class 0", f"{p0 * 100:.2f}%")
                col2.metric("Class 1", f"{p1 * 100:.2f}%")

                st.caption(
                    "Class 0 = Benign and Class 1 = Malignant, "
                    "assuming your training labels used this mapping."
                )

            with st.expander("View complete API response"):
                st.json(result)

        else:
            st.error(f"API returned status {response.status_code}")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI se connection nahi ho saka. "
            "Pehle backend start karein."
        )

    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")

    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")

    except (ValueError, TypeError, KeyError) as e:
        st.error(f"Response processing error: {e}")
