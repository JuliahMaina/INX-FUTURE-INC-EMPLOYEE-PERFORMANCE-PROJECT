import streamlit as st

st.title("My App")

import streamlit as st
import pandas as pd
import pickle
import gzip
import io

# Load the model from GitHub (replace with your raw GitHub URL)
model_rf = "model_rf.pkl.gz"  # Replace with your actual URL
try:
    response = pd.read_csv(model_url, compression='gzip', header=None, sep='\t', quoting=3, error_bad_lines=False)
    model_data = gzip.decompress(response.iloc[0, 0].encode('latin1'))
    model_rf = pickle.loads(model_data)

except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Feature Columns
feature_columns = ['EmpDepartment', 'EmpEnvironmentSatisfaction',
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

def preprocess_data(df):
    """Preprocesses the dataframe for prediction."""
    df = df[feature_columns]
    df['EmpDepartment'] = df['EmpDepartment'].replace(department_encoding)
    return df

def predict_performance(df, model):
    """Predicts performance rating using the loaded model."""
    predictions = model.predict(df)
    return predictions

st.title("Employee Performance Prediction")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        df = excel_file.parse(excel_file.sheet_names[0])

        # Validate feature columns
        if not all(col in df.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in df.columns]
            st.error(f"Error: The uploaded file is missing the following required columns: {missing_cols}")
        else:
            # Preprocess the data
            preprocessed_df = preprocess_data(df.copy())

            # Make predictions
            predictions = predict_performance(preprocessed_df, model_rf)

            # Display results
            result_df = pd.DataFrame({'Predicted Performance Rating': predictions})
            st.write("Prediction Results:")
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
