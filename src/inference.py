import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

Cluster_descriptions = {
    0: "The Loyal Spenders: Long-time customers, high bills, 1-2 year contracts, rarely leave.",
    1: "The At-Risk Group: Newer customers, high bills, month-to-month plans, quitting fast.",
    2: "The Average Users: Basic internet users, average bills, normal cancellation rate.",
    3: "The Phone-Only Group: No internet, tiny bills, very stable, rarely leave.",
    4: "The Internet-Only Group: No phone service, just internet, normal cancellation rate."
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path

@st.cache_resource
def load_kmeans_pipeline(model_path='models/kmeans_pipeline.pkl'):
    path = _resolve_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"KMeans model not found: {path}")
    return joblib.load(path)

@st.cache_resource
def load_rf_model(model_path='models/rf_model.pkl'):
    path = _resolve_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Random Forest model not found: {path}")
    return joblib.load(path)

@st.cache_resource
def load_shap_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)


def identify_user_cluster(user_df):

    pipeline = load_kmeans_pipeline()
    cluster_id = pipeline.predict(user_df)[0]
    description = Cluster_descriptions.get(cluster_id, "Unknown Archetype")
    
    return cluster_id, description

def random_forest_inference(user_df):

    rf_model = load_rf_model()
    prediction = rf_model.predict(user_df)[0]
    probability = rf_model.predict_proba(user_df)[0][1]
    return prediction, probability

def display_prediction_results(prediction, probability):
    """
    Displays the prediction result in Streamlit using the user-provided logic.
    """
    if prediction == 1:
        st.error(f"**High Risk**: This customer is likely to CHURN.")
    else:
        st.success(f"**Low Risk**: This customer is likely to STAY.")
    
    st.metric("Churn Probability", f"{probability:.2%}")
    st.progress(float(probability))

def get_top_contributors(user_df, top_n=3):
    """
    Returns the names of the top N features contributing to the churn prediction.
    """
    rf_model = load_rf_model()
    explainer = load_shap_explainer(rf_model)
    shap_values = explainer(user_df)

    if len(shap_values.shape) == 3:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]

    feature_names = user_df.columns.tolist()
    
    # Create a list of (feature_name, absolute_shap_value)
    contributions = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)
    
    return [c[0] for c in contributions[:top_n]]

def rf_feature_contribution_to_churn(user_df):
    rf_model = load_rf_model()

    explainer = load_shap_explainer(rf_model)
    shap_values = explainer(user_df)

    if len(shap_values.shape) == 3:
        churn_explanation = shap_values[0, :, 1]
    else:
        churn_explanation = shap_values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(churn_explanation, show=False)
    plt.tight_layout()
    
    return fig