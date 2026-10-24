import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# Label encodings (same as the notebook's LabelEncoder: alphabetical order)
PCLASS = {"First": 0, "Second": 1, "Third": 2}
GENDER = {"Female": 0, "Male": 1}
SIBLING = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
EMBARKED = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

st.title("🚢 Titanic Survival Predictor")
st.write("Would you have survived the Titanic? Pick the passenger's details below.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings/spouses aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of embarkation",
                            ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict", type="primary"):
    features = pd.DataFrame({
        "PClass": [PCLASS[pclass]],
        "Gender": [GENDER[gender]],
        "Sibling": [SIBLING[sibling]],
        "Embarked": [EMBARKED[embarked]],
    })
    prediction = model.predict(features)[0]
    margin = abs(float(model.decision_function(features)[0]))
    confidence = "High" if margin > 0.8 else "Medium" if margin > 0.3 else "Low"

    if prediction == 1:
        st.success("Prediction: SURVIVED ✅")
    else:
        st.error("Prediction: NOT SURVIVED ❌")
    st.caption(f"Model confidence: **{confidence}**")

    st.table(pd.DataFrame(
        {"Passenger class": [pclass], "Gender": [gender],
         "Siblings/spouses": [sibling], "Embarked": [embarked]}))

st.divider()
st.caption("Support Vector Classifier trained on a small 100-row sample of "
           "Titanic passengers, so treat results as a learning demo, not history.")
