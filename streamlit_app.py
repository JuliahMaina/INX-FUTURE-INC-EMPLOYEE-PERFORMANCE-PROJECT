import streamlit as st

st.title("My App")

import streamlit as st
import pandas as pd
import pickle
import gzip

# Feature Columns
feature_cols = ['EmpDepartment', 'EmpEnvironmentSatisfaction',
                'EmpLastSalaryHikePercent', 'EmpWorkLifeBalance',
                'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Department Encoding
department_encoding = {
    "Data Science": 0,
    "Development": 1,
    "Finance": 2,
    "Human Resources": 3,
    "Research & Development": 4,
    "Sales": 5
}

# Model Loading Function
@st.cache_resource # Caching the model for faster loading
def load_model():
    try:
        with gzip.open('model_rf.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'model_rf.pkl.gz' not found in the current directory. Please ensure it is uploaded or placed correctly.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Streamlit App
def main():
    st.title("Employee Performance Rating Prediction App")

    uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            # Read the first worksheet of the Excel file
            df = pd.read_excel(uploaded_file, sheet_name=0)

            # Validate Feature Columns
            if not all(col in df.columns for col in feature_cols):
                missing_cols = [col for col in feature_cols if col not in df.columns]
                st.error(f"The first worksheet is missing the following required columns: {', '.join(missing_cols)}")
            else:
                st.success("Worksheet validated successfully. Required columns found!")

                # Make a copy to avoid SettingWithCopyWarning if you intend to modify df later
                df_processed = df.copy()

                # Encode 'EmpDepartment'
                if 'EmpDepartment' in df_processed.columns:
                    df_processed['EmpDepartment'] = df_processed['EmpDepartment'].replace(department_encoding)
                else:
                    st.warning("'EmpDepartment' column not found in the uploaded file, encoding step skipped for this column.")

                # Load Model
                model = load_model()
                if model:
                    # Prepare features for prediction
                    X = df_processed[feature_cols].copy() # Keep the feature columns data

                    # Predict Performance Rating
                    predictions = model.predict(X)

                    # Display Results - Include Feature Columns
                    results_df = X.copy() # Start with the feature columns data
                    results_df['Predicted Performance Rating'] = predictions # Add the prediction column
                    st.subheader("Prediction Results with Input Features")
                    st.dataframe(results_df)

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

if _name_ == "_main_":
    main()
