# The 5-Week Machine Learning Engineer Challenge - Week 5 - Classification Deployment

# Importing the libraries
import joblib
import streamlit as st

# Configuring the Streamlit page
st.set_page_config(
    page_title="Alzheimer's Disease Predictor",
    page_icon="🧠",
    layout="wide"
)

# Applying the H-Tech visual identity
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #002060;
    }

    div.stButton > button {
        background-color: #002060;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #001746;
        color: white;
        border: none;
    }

    [data-testid="stCaptionContainer"] {
        color: #5F6B7A;
    }

    hr {
        border-color: #DCE3EE;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Loading the trained model
model = joblib.load("ensemble_classifier.joblib")

# Displaying the H-Tech logo
st.image("htech_logo.jpg", width=420)

# Creating the Streamlit app title
st.title("🧠 Alzheimer's Disease Predictor")

# Adding the authoring
st.markdown(
    "### An AI-powered Machine Learning application by **H-Tech Education**"
)

# Adding a short description of the app
st.write(
    "This predictor is powered by an advanced ensemble Machine Learning model "
    "that combines three of the most powerful gradient boosting algorithms: "
    "XGBoost, LightGBM, and CatBoost. By bringing together the predictive strengths "
    "of these state-of-the-art boosting models, the ensemble delivers a robust and "
    "high-performance approach to Alzheimer's Disease prediction. Enter your "
    "information below to generate the prediction."
)

st.divider()

# Gathering the personal information
st.subheader("👤 Personal Information")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age",
            min_value=60,
            max_value=90,
            value=75
        )

        ethnicity_label = st.selectbox(
            "Ethnicity",
            ["Caucasian", "African American", "Asian", "Other"]
        )

        ethnicity_mapping = {
            "Caucasian": 0,
            "African American": 1,
            "Asian": 2,
            "Other": 3
        }

        ethnicity = ethnicity_mapping[ethnicity_label]

        bmi = st.slider(
            "BMI",
            min_value=15.0,
            max_value=40.0,
            value=25.0,
            step=0.1
        )

        st.caption(
            "BMI = weight (kg) / height² (m²). "
            "Example: 75 kg and 1.80 m → BMI = 75 / 1.80² = 23.1."
        )

    with col2:
        gender_label = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        gender = 0 if gender_label == "Male" else 1

        education_label = st.selectbox(
            "Education Level",
            ["None", "High School", "Bachelor's", "Higher"]
        )

        education_mapping = {
            "None": 0,
            "High School": 1,
            "Bachelor's": 2,
            "Higher": 3
        }

        education_level = education_mapping[education_label]

# Gathering the lifestyle information
st.subheader("🌿 Lifestyle")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        smoking = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        physical_activity = st.slider(
            "Weekly Physical Activity (hours)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        sleep_quality = st.slider(
            "How would you rate your sleep quality? (4 = poor, 10 = excellent)",
            min_value=4.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    with col2:
        alcohol_consumption = st.slider(
            "Weekly Alcohol Consumption (units)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        )

        diet_quality = st.slider(
            "How would you rate your diet quality? (0 = very poor, 10 = excellent)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

# Gathering the medical history
st.subheader("🩺 Medical History")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        family_history_alzheimers = st.selectbox(
            "Family History of Alzheimer's",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        head_injury = st.selectbox(
            "History of Head Injury",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        cardiovascular_disease = st.selectbox(
            "Cardiovascular Disease",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        depression = st.selectbox(
            "Depression",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

# Gathering the cognitive and behavioral information
st.subheader("🧠 Cognitive and Behavioral Signs")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        memory_complaints = st.selectbox(
            "Memory Complaints",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        confusion = st.selectbox(
            "Confusion",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        personality_changes = st.selectbox(
            "Personality Changes",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        forgetfulness = st.selectbox(
            "Forgetfulness",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        behavioral_problems = st.selectbox(
            "Behavioral Problems",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        disorientation = st.selectbox(
            "Disorientation",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        difficulty_completing_tasks = st.selectbox(
            "Difficulty Completing Tasks",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

# Preparing the input data
input_data = [[
    age,
    gender,
    ethnicity,
    education_level,
    bmi,
    smoking,
    alcohol_consumption,
    physical_activity,
    diet_quality,
    sleep_quality,
    family_history_alzheimers,
    cardiovascular_disease,
    diabetes,
    depression,
    head_injury,
    hypertension,
    memory_complaints,
    behavioral_problems,
    confusion,
    disorientation,
    personality_changes,
    difficulty_completing_tasks,
    forgetfulness
]]

# Creating the prediction section
st.divider()

st.subheader("🔍 Prediction")

predict_button = st.button(
    "Predict Alzheimer's Disease",
    use_container_width=True
)

# Making and displaying the prediction
if predict_button:
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.warning(
            "The model predicts that the patient is likely to have Alzheimer's Disease."
        )
    else:
        st.success(
            "The model predicts that the patient is unlikely to have Alzheimer's Disease."
        )

# Adding the medical disclaimer
st.caption(
    "This application is an educational Machine Learning demonstration "
    "and is not intended to provide medical diagnosis or medical advice."
)