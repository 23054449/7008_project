import streamlit as st
import pandas as pd
import subprocess
import os
import time

st.set_page_config(layout="wide")

# Title header
st.title("WQD7008 Project")

# Initialize session state
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None

if "clean_data" not in st.session_state:
    st.session_state.clean_data = None

if 'raw_data_run' not in st.session_state:
    st.session_state.raw_data_run = False

if "clean_data_run" not in st.session_state:
    st.session_state.clean_data_run = False

if "submit_run" not in st.session_state:
    st.session_state.submit_run = False

if "training_run" not in st.session_state:
    st.session_state.training_run = False

if "DT_run" not in st.session_state:
    st.session_state.DT_run = False

if "KNN_run" not in st.session_state:
    st.session_state.KNN_run = False

if "RF_run" not in st.session_state:
    st.session_state.RF_run = False

if "LR_run" not in st.session_state:
    st.session_state.LR_run = False

if "NB_run" not in st.session_state:
    st.session_state.NB_run = False

if "status_DT" not in st.session_state:
    st.session_state.status_DT = None

if "status_KNN" not in st.session_state:
    st.session_state.status_KNN = None

if "status_RF" not in st.session_state:
    st.session_state.status_RF = None

if "status_LR" not in st.session_state:
    st.session_state.status_LR = None

if "status_NB" not in st.session_state:
    st.session_state.status_NB = None

if "DT_PKL" not in st.session_state:
    st.session_state.DT_PKL = None

if "KNN_PKL" not in st.session_state:
    st.session_state.KNN_PKL = None

if "RF_PKL" not in st.session_state:
    st.session_state.RF_PKL = None

if "LR_PKL" not in st.session_state:
    st.session_state.LR_PKL = None

if "NB_PKL" not in st.session_state:
    st.session_state.NB_PKL = None

venv_path = os.path.join(".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join("venv", "bin", "python")
preprocessing_path = os.path.join("scripts", "preprocessing.py")
clean_data_path = os.path.join('temp', 'cleanData.csv')
raw_data_path = os.path.join('temp', 'rawData.csv')
training_path = os.path.join("scripts", "training.py")
trainData_path = os.path.join("temp", "cleanTrain.csv")
testData_path = os.path.join("temp", "cleanTest.csv")

# Define paths for each model
model_paths = {
    'dt': os.path.join("temp", "fraud_detection_dt_model.pkl"),
    'knn': os.path.join("temp", "fraud_detection_knn_model.pkl"),
    'rf': os.path.join("temp", "fraud_detection_rf_model.pkl"),
    'lr': os.path.join("temp", "fraud_detection_lr_model.pkl"),
    'nb': os.path.join("temp", "fraud_detection_nb_model.pkl")
}

model_name = {
    'dt': 'Decesion Tree',
    'knn': 'K Nearest Neighbour',
    'rf': 'Random Forest',
    'lr': 'Logistic Regression',
    'nb': 'Naive Bayes'
}

models = ['dt', 'knn']#, 'rf', 'lr', 'nb']

def check_processing_complete():
    try:
        with open(os.path.join('preprocessing_outputs', 'job.out'), "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-2].strip()
                if last_line == "PROCESSING COMPLETE":
                    return False
                else:
                    return True
            else:
                return True
    except FileNotFoundError:
        print("Error: The file 'job.out' was not found.")
        return True
    except PermissionError:
        print("Error: Permission denied when accessing 'job.out'.")
        return True
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return True

def check_training_complete(model):
    try:
        with open(os.path.join('training_outputs', f'{model}.out'), "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-2].strip()
                if last_line == "TRAINING COMPLETE":
                    st.session_state[f'{model.upper()}_run'] = True
                    return False
                else:
                    return True
            else:
                return True
    except FileNotFoundError:
        print(f"Error: The file '{model}.out' was not found.")
        return True
    except PermissionError:
        print(f"Error: Permission denied when accessing '{model}.out'.")
        return True
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return True

def run_preprocess(filepath):
    try:
        if os.name == "nt":
            subprocess.run([venv_path, preprocessing_path, "--source", filepath], check=True)
        else:
            subprocess.run(["condor_submit", os.path.join('preprocessing.sub')], check=True)
            while check_processing_complete():
                time.sleep(2)
    except subprocess.CalledProcessError as e:
        status_text.error(f"Error during preprocessing: {e}")

# File uploader (accepts both .xlsx and .csv files)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Submit button
if st.session_state.submit_run is False:
    submit_button_placeholder = st.empty()
    if submit_button_placeholder.button("Submit"):
        if uploaded_file is not None:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                st.session_state.raw_data = pd.read_csv(uploaded_file, index_col=[0])
                st.session_state.submit_run = True
                submit_button_placeholder.empty()  # Clear the submit button

            else:
                st.write("Unsupported file type. Please upload a .csv file.")
                st.stop()
        else:
            st.write("Please upload a CSV file first.")

# Create a container for displaying raw data
ori_data_container = st.container()

# Create a container for displaying clean data
clean_data_container = st.container()

# Create a container for displaying model interface
model_button_container = st.container()

# Create a container for displaying model interface
model_container = st.container()

# Create a container for displaying model interface
download_container = st.container()

with ori_data_container:
    if st.session_state.raw_data is not None:
        status_text = st.empty()
        if not st.session_state.raw_data_run:
            status_text.warning("Reading input file.... Please wait.")

            # Ensure the 'temp' directory exists
            os.makedirs('temp', exist_ok=True)
            
            # Save the CSV file
            st.session_state.raw_data.to_csv(raw_data_path)

            st.write("Here is a preview of the data you uploaded:")
            st.dataframe(st.session_state.raw_data.head(100))
            # Display the total number of rows
            num_rows = st.session_state.raw_data.shape[0]
            st.write(f"Total number of rows: {num_rows}")
            status_text.success("Finish reading input file!")
            time.sleep(1)
            st.session_state.raw_data_run = True

        else:
            st.write("Here is a preview of the data you uploaded:")
            st.dataframe(st.session_state.raw_data.head(100))
            # Display the total number of rows
            num_rows = st.session_state.raw_data.shape[0]
            st.write(f"Total number of rows: {num_rows}")
            status_text.success("Finish reading input file!")


with clean_data_container:
    if st.session_state.raw_data_run is True:
        status_text = st.empty()
        status_text1 = st.empty()
        if not st.session_state.clean_data_run:
            # Run preprocessing
            status_text.warning("Preprocessing ongoing.... Please wait.")
            run_preprocess(raw_data_path)
            status_text.success("Preprocessing completed successfully!")

            time.sleep(1)
            status_text1.warning("Loading cleaned data.... Please wait.")
            st.session_state.clean_data = pd.read_csv(clean_data_path, index_col=[0])
            st.write("Here is a preview of the cleaned data:")
            st.dataframe(st.session_state.clean_data.head(100))
            # Display the total number of rows
            num_rows = st.session_state.clean_data.shape[0]
            st.write(f"Total number of rows: {num_rows}")
            status_text1.success("Finish loading clean data!")
            st.session_state.clean_data_run = True

        else:
            status_text.success("Preprocessing completed successfully!")
            st.write("Here is a preview of the cleaned data:")
            st.dataframe(st.session_state.clean_data.head(100))
            # Display the total number of rows
            num_rows = st.session_state.clean_data.shape[0]
            st.write(f"Total number of rows: {num_rows}")
            status_text1.success("Finish loading clean data!")

def print_line():
    st.markdown("<hr>", unsafe_allow_html=True)

def run_training(model):
    if os.name == "nt":
        return subprocess.run([venv_path, training_path, trainData_path, testData_path, model], capture_output=True, text=True)
    else:
        subprocess.run(["condor_submit", os.path.join(f'training_{model}.sub')], check=True)

def save_pkl():
    for model, path in model_paths.items():
        with open(path, "rb") as file:
            st.session_state[f'{model.upper()}_PKL'] = file.read()

def results_print():
    for model, name in model_name.items():
        if st.session_state[f'{model.upper()}_run'] is True:
            st.header(name)
            if os.name == "nt":
                st.code(st.session_state[f'status_{model.upper()}'].stdout)
            else:
                with open(os.path.join('training_results', f'{model}.txt'), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        st.code(line)
            st.download_button(label=f"Download {name} Model", data=st.session_state[f'{model.upper()}_PKL'], file_name=f"fraud_detection_{model}_model.pkl", mime="application/octet-stream")

    # if st.session_state.DT_run is True:
    #     print_line()
    #     st.header("Decision Tree")
    #     st.code(st.session_state.status_DT.stdout)
    #     st.download_button(label="Download Decision Tree Model", data=st.session_state.DT_PKL, file_name="fraud_detection_dt_model.pkl", mime="application/octet-stream")

    # if st.session_state.KNN_run is True:
    #     print_line()
    #     st.header("K Nearest Neighbour")
    #     st.code(st.session_state.status_KNN.stdout)
    #     st.download_button(label="Download K Nearest Neighbour Model", data=st.session_state.KNN_PKL, file_name="fraud_detection_knn_model.pkl", mime="application/octet-stream")

    # if st.session_state.RF_run is True:
    #     print_line()
    #     st.header("Random Forest")
    #     st.code(st.session_state.status_RF.stdout)
    #     st.download_button(label="Download Random Forest Model", data=st.session_state.RF_PKL, file_name="fraud_detection_rf_model.pkl", mime="application/octet-stream")

    # if st.session_state.LR_run is True:   
    #     print_line()
    #     st.header("Logistic Regression")
    #     st.code(st.session_state.status_LR.stdout)
    #     st.download_button(label="Download Logistic Regression Model", data=st.session_state.LR_PKL, file_name="fraud_detection_lr_model.pkl", mime="application/octet-stream")

    # if st.session_state.NB_run is True:
    #     print_line()
    #     st.header("Naive Bayes")
    #     st.code(st.session_state.status_NB.stdout)
    #     st.download_button(label="Download Naive Bayes Model", data=st.session_state.NB_PKL, file_name="fraud_detection_nb_model.pkl", mime="application/octet-stream")

with model_button_container:
    if st.session_state.clean_data_run is True:
        if st.session_state.training_run is False:
            training_button_placeholder = st.empty()
            if training_button_placeholder.button("Start Training!"):
                training_button_placeholder.empty()
                st.session_state.training_run = True

with model_container:
    if st.session_state.training_run is True:
        status_text = st.empty()
        if (st.session_state.DT_run or st.session_state.KNN_run or st.session_state.RF_run or st.session_state.LR_run or st.session_state.NB_run) is False:
            status_text.warning("Training in progress.... Please wait.")
            if os.name == "nt":
                for model in models:
                    st.session_state[f'status_{model.upper()}'] = run_training(model)
                    st.session_state[f'{model.upper()}_run'] = True
            else:
                for model in models:
                    run_training(model)
                while (st.session_state.DT_run or st.session_state.KNN_run or st.session_state.RF_run or st.session_state.LR_run or st.session_state.NB_run) is False:
                    for model in models:
                        check_training_complete(model)
                    time.sleep(2)
            save_pkl()
            status_text.success("Training completed successfully!")
            results_print()
        else:
            status_text.success("Training completed successfully!")
            results_print()
