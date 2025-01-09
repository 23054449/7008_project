import streamlit as st
import pandas as pd
import subprocess
import tempfile

# Title header
st.title("WQD7008 Project")

def run_preprocess(filepath):
    status_text = st.empty()
    status_text.warning("Preprocessing ongoing.")
    try:
        subprocess.run([".\venv\bin\python.exe", "Preprocessing_7008.py", "--source", filepath], check=True)
        status_text.success("Preprocessing completed successfully!")
    except subprocess.CalledProcessError as e:
        status_text.error(f"Error during preprocessing: {e}")

# File uploader (accepts both .xlsx and .csv files)
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["csv"])

if "data" not in st.session_state:
    st.session_state.data = None

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            st.session_state.data = pd.read_csv(uploaded_file, index_col=[0])

        else:
            st.write("Unsupported file type. Please upload a .xlsx or .csv file.")
            st.stop()
    else:
        st.write("Please upload an Excel or CSV file first.")

# Create a container for displaying data
ori_data_container = st.container()

with ori_data_container:
    if st.session_state.data is not None:
        st.write("Here is the data from the file you uploaded:")
        st.write(st.session_state.data)
        # Display the total number of rows
        num_rows = st.session_state.data.shape[0]
        st.write(f"Total number of rows: {num_rows}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_filepath = temp_file.name
            
            # Run preprocessing
            run_preprocess(temp_filepath)