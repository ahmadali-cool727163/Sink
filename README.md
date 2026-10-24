# Titanic Survival Predictor

Streamlit app that predicts Titanic passenger survival from class, gender,
siblings/spouses aboard and port of embarkation, using a Support Vector Classifier.

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Note: `svc_trained_model.pkl` was saved with scikit-learn 1.7.2, so keep that
version pinned in `requirements.txt`.
