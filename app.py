import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --- Configuration and Model Loading ---
st.set_page_config(layout="wide", page_title="Subscription Predictor")


MODEL_PATH = Path(__file__).parent / "final_subscription_predictor_pipeline.pkl"

@st.cache_resource
def load_model(path):
    """Loads the final trained model pipeline from the specified path."""
    try:
        with open(path, 'rb') as file:
            loaded_pipeline = pickle.load(file)
        return loaded_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure 'deploy_model.py' was run.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model(MODEL_PATH)

# --- Feature Definitions (Based on EDA) ---
JOB_OPTIONS = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
MARITAL_OPTIONS = ['married', 'single', 'divorced', 'unknown']
EDUCATION_OPTIONS = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
CONTACT_OPTIONS = ['cellular', 'telephone']
POUTCOME_OPTIONS = ['failure', 'nonexistent', 'success']
DEFAULT_OPTIONS = ['no', 'unknown']

# --- SCENARIO DEFAULTS (Corrected for Consistency) ---
SCENARIOS = {
    "Low Likelihood (No)": {
        'age': 65, 'job': 'retired', 'marital': 'married', 'education': 'university.degree', 'default': 'no',
        'campaign': 1, 'previous': 2, 'pdays': 10, 'poutcome': 'success', 'contact': 'telephone',
        'euribor3m': 1.30, 'cons_price_idx': 92.80, 'cons_conf_idx': -46.2, 'month': 'mar',
    },
    "High Likelihood (Yes)": {
        'age': 35, 'job': 'blue-collar', 'marital': 'married', 'education': 'basic.9y', 'default': 'no',
        'campaign': 5, 'previous': 0, 'pdays': 999, 'poutcome': 'nonexistent', 'contact': 'cellular',
        'euribor3m': 4.90, 'cons_price_idx': 93.99, 'cons_conf_idx': -36.4, 'month': 'may',
    },
}

# --- State Management Function ---
def set_default_values(scenario_key):
    """Updates Streamlit session state with values from the selected scenario."""
    for key, value in SCENARIOS[scenario_key].items():
        st.session_state[key] = value
    st.session_state['month_select'] = SCENARIOS[scenario_key]['month']
    st.session_state['poutcome_select'] = SCENARIOS[scenario_key]['poutcome']
    st.session_state['contact_select'] = SCENARIOS[scenario_key]['contact']


# --- Helper Functions for Feature Engineering (MUST MATCH TRAINING) ---
def engineer_features(data):
    """Performs the manual feature engineering steps on raw inputs."""
    data['campaign_log'] = np.log(data['campaign'])
    data['age_capped'] = np.where(data['age'] > 70, 70, data['age'])
    data['was_contacted'] = np.where(data['pdays'] != 999, 1, 0)
    data['has_default'] = np.where(data['default'] == 'no', 0, 1)
    data['favorable_conditions'] = np.where(
        (data['euribor3m'] < 2.0) & (data['was_contacted'] == 1), 
        1, 
        0
    )
    data = data.drop(columns=['campaign', 'age', 'pdays', 'default'])
    
    return data

# --- Streamlit UI and Input Collection ---

# Use custom classes for title and subtitle
st.markdown("<h1 class='main-title'>🎯 Term Deposit Subscription Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p class='subtitle'>
    This application leverages a <b>fine-tuned XGBoost Classifier</b> to evaluate client profiles and economic indicators, 
    predicting the likelihood of term deposit subscriptions with high precision. 
    It helps optimize <b>marketing ROI</b> by focusing efforts on clients with the highest conversion potential.
</p>
""", unsafe_allow_html=True)



if pipeline:
    
    # Initialize state with a default high-likelihood scenario if not already set
    if 'month' not in st.session_state:
        set_default_values("High Likelihood (Yes)")

    # --- Scenario Buttons (Professional Styling) ---
    st.markdown("### Quick Demonstration Scenarios")
    st.markdown("")
    
    
    col_scenario_1, col_scenario_2, col_scenario_3 = st.columns([1, 1, 1])

    with col_scenario_1:
        st.button("High Likelihood (YES)", type="primary", on_click=set_default_values, 
                  args=("High Likelihood (Yes)",), 
                  help="Sets optimal values: Successful prior contact, Low interest rates, Retirement age.")
    with col_scenario_2:
        st.button("Low Likelihood (NO)", type="secondary", on_click=set_default_values, 
                  args=("Low Likelihood (No)",), 
                  help="Sets poor values: No prior contact, High interest rates, Blue-collar job, May contact.")
    with col_scenario_3:
        # Prediction Threshold Slider (Crucial for decision-making)
        prediction_threshold = st.slider(
            "Decision Threshold (F1-Optimized)", 
            min_value=0.01, 
            max_value=0.99, 
            value=0.35, 
            step=0.01,
            help=(
    "Defines the probability threshold for classifying a client as a potential subscriber (1) or non-subscriber (0). "
    "Default = 0.35, offering a balanced Recall–Precision trade-off. "
    "If missing a subscriber is costly → lower the threshold (focus on Recall). "
    "If contacting a non-subscriber is costly → raise the threshold (focus on Precision)."
)

        )
    
    st.markdown("---")


    with st.form("prediction_form"):
        st.markdown("""
        <h2 style="font-weight:700; color:#ffffff; margin-bottom:0.5rem;">
        Client and Campaign Data Inputs
        </h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # COLUMN 1: Demographics and Core Contact
        with col1:
            st.markdown("<h2 style='font-weight:600;'>Client Profile</h2>", unsafe_allow_html=True)
            age = st.number_input("Age (Years)", min_value=18, max_value=99, key='age')
            job = st.selectbox("Job Title", options=JOB_OPTIONS, key='job')
            marital = st.selectbox("Marital Status", options=MARITAL_OPTIONS, key='marital')
            education = st.selectbox("Education Level", options=EDUCATION_OPTIONS, key='education')
            default = st.selectbox("Has Credit Default?", options=DEFAULT_OPTIONS, key='default')
            
        # COLUMN 2: Campaign and History (Includes strongest categorical predictors)
        with col2:
            
            st.markdown("<h2 style='font-weight:600;'>Campaign History</h2>", unsafe_allow_html=True)
            campaign = st.number_input("Campaign Contacts (Current)", min_value=1, max_value=50, key='campaign')
            previous = st.number_input("Previous Contacts (Total)", min_value=0, max_value=20, key='previous')
            pdays = st.number_input("Days Since Last Contact", min_value=0, max_value=999, key='pdays')
            
            poutcome = st.selectbox("Previous Outcome", options=POUTCOME_OPTIONS, key='poutcome_select')
            contact = st.selectbox("Contact Type", options=CONTACT_OPTIONS, key='contact_select')
            
        # COLUMN 3: Economic Indicators (Final Scaled Features, including strongest numerical predictor)
        with col3:
            
            st.markdown("<h2 style='font-weight:600;'>Economic Context / Timing</h2>", unsafe_allow_html=True)
            month = st.selectbox("Last Contact Month (Strong Predictor)", options=['mar', 'may', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jun', 'apr'], key='month_select', index=0)
            st.markdown("<br>", unsafe_allow_html=True) # Spacer for better alignment
            euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.0, max_value=5.1, step=0.01, key='euribor3m')
            cons_price_idx = st.slider("Consumer Price Index", min_value=92.0, max_value=95.0, step=0.01, key='cons_price_idx')
            cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-55.0, max_value=-25.0, step=0.1, key='cons_conf_idx')

        submitted = st.form_submit_button("Predict Subscription Likelihood", type="primary")

    if submitted:
        # Create a dictionary of the raw inputs using current session state values
        raw_input_data = {
            'age': st.session_state.age,
            'job': st.session_state.job,
            'marital': st.session_state.marital,
            'education': st.session_state.education,
            'default': st.session_state.default,
            'campaign': st.session_state.campaign,
            'previous': st.session_state.previous,
            'pdays': st.session_state.pdays,
            'poutcome': st.session_state.poutcome_select,
            'contact': st.session_state.contact_select,
            'euribor3m': st.session_state.euribor3m,
            'cons.price.idx': st.session_state.cons_price_idx,
            'cons.conf.idx': st.session_state.cons_conf_idx
        }

        # Convert to DataFrame (crucial step for pipeline)
        input_df = pd.DataFrame([raw_input_data])
        
        # 1. Perform client-side feature engineering
        engineered_df = engineer_features(input_df)

        # 2. Make Prediction (Pipeline handles final scaling/encoding)
        prediction_proba = pipeline.predict_proba(engineered_df)[:, 1]
        
        # 3. Apply custom threshold for final prediction
        prediction = 1 if prediction_proba[0] >= prediction_threshold else 0
        
        # 4. Display Results
        st.subheader("Prediction Result")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.success("High Likelihood of Subscription! ✅")
            else:
                st.warning("Low Likelihood of Subscription. ❌")

        with col_res2:
            st.metric(
                label=f"Subscription Probability (Threshold: {prediction_threshold:.2f})",
                value=f"{prediction_proba[0]*100:.2f}%"
            )

        # Final recommendation box (Updated to use global styles for colors)
        st.markdown(f"""
            <div style="
                border: 1px solid var(--primary-color); 
                padding: 10px; 
                border-radius: 5px; 
                /* Using Streamlit warning/success classes for background color for better theme integration */
                background-color: var(--secondary-background-color);">
                <strong>MODEL RECOMMENDATION:</strong> 
                {
                    '🔴 Prioritize Contact for maximum reach.' if prediction == 1 else 
                    '⚪ Do Not Prioritize Contact (Low ROI).'
                }
            </div>
            """, unsafe_allow_html=True)
