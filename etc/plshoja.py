import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from logparser import Drain
from imblearn.over_sampling import SMOTE

# Hardcoded model paths
MODEL_PATHS = {
    "linux": [
        "models/cnn_model_linux_20.h5",
        "models/rnn_model_linux_20.h5",
        "models/lstm_model_linux_20.h5",
    ],
    "bgl": [
        "models/lstm_model_bgl.h5",
    ],
    "hdfs": [
        "models/cnn_log_classifier.h5",
        "models/rnn_log_classifier.h5",
        "models/lstm_log_classifier.h5",
    ],
}

# Function to detect log type
def detect_log_type(log_file):
    """
    Detects the log type based on the first line of the uploaded log file.
    Defaults to 'linux' if no match is found.
    """
    first_line = log_file[0].decode("ISO-8859-1", errors='ignore').strip()
    print(f"First line: {first_line}")
    if (len(first_line) > 15 and first_line[0:3].isalpha() and first_line[4:6].isdigit() and
        first_line[7:9].count(':') == 2 and first_line[10:].startswith(" ")):
        print("Matched Linux log pattern")
        return "linux"
    elif first_line.startswith("-") or first_line.strip().isdigit():
        print("Matched BGL log pattern")
        return "bgl"
    elif re.match(r'^\d{6}\s+\d{6}\s+\d+\s+[A-Z]+\s+\S+:.*', first_line):
        print("Matched HDFS log pattern")
        return "hdfs"
    print("Defaulting to Linux log type")
    return "linux"

# Function to parse logs
def parse_logs(log_file, log_type):
    """
    Parses the uploaded log file based on the detected log type.
    """
    if log_type == "linux":
        log_pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\w+)\s+(\w+):\s+(.*)'
        data = []
        for line in log_file:
            match = re.match(log_pattern, line.decode('ISO-8859-1'))
            if match:
                data.append({
                    "Timestamp": match.group(1),
                    "Host": match.group(2),
                    "Log_Level": match.group(3),
                    "Message": match.group(4)
                })
        return pd.DataFrame(data)
    elif log_type == "bgl":
        logs = [line.decode('utf-8').strip() for line in log_file]
        return pd.DataFrame(logs, columns=["Message"])
    elif log_type == "hdfs":
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
        regex = [
            r"(?<=blk_)[-\d]+",  # block_id
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
            r"(/[-\w]+)+",  # file paths
        ]
        indir = "./temp_logs"
        outdir = "./temp_logs"
        os.makedirs(indir, exist_ok=True)
        parser = Drain.LogParser(
            log_format, indir=indir, outdir=outdir, depth=5, st=0.5, rex=regex, keep_para=False
        )
        temp_log_path = os.path.join(indir, "temp_hdfs.log")
        with open(temp_log_path, "wb") as f:
            f.writelines(log_file)
        try:
            parser.parse("temp_hdfs.log")
            structured_file = os.path.join(outdir, "temp_hdfs.log_structured.csv")
            df = pd.read_csv(structured_file)
            df["Block_ID"] = df["Content"].str.extract(r'blk_([-\d]+)')
            return df
        except FileNotFoundError:
            st.error("No structured file generated. Ensure the log file format matches the expected HDFS format.")
            return pd.DataFrame()
    return pd.DataFrame()

# Function to preprocess Linux logs
def preprocess_linux_logs(linux_data, max_sequence_length=100):
    """
    Preprocess Linux logs by tokenizing, padding, and applying SMOTE for class balancing.
    """
    print("Preprocessing Linux logs...")
    linux_data["Message"] = linux_data["Message"].fillna('').astype(str)
    linux_tokenizer = Tokenizer()
    linux_tokenizer.fit_on_texts(linux_data["Message"])
    linux_sequences = linux_tokenizer.texts_to_sequences(linux_data["Message"])
    linux_padded = pad_sequences(linux_sequences, maxlen=max_sequence_length)
    if "Label" not in linux_data:
        raise ValueError("Linux data must contain a 'Label' column for SMOTE.")
    linux_labels = linux_data["Label"]
    smote = SMOTE(random_state=42)
    linux_inputs, linux_labels = smote.fit_resample(linux_padded, linux_labels)
    print(f"Preprocessed Linux data shape: {linux_inputs.shape}")
    print(f"Preprocessed Linux labels shape: {linux_labels.shape}")
    return linux_inputs, linux_labels

# Function to preprocess logs
def preprocess_logs(df, log_type, expected_sequence_length=100):
    """
    Tokenizes and preprocesses the logs for the prediction models.
    """
    tokenizer = Tokenizer(num_words=1000)
    column = "Message" if log_type in ["linux", "bgl"] else "Content"
    tokenizer.fit_on_texts(df[column])
    sequences = tokenizer.texts_to_sequences(df[column])
    return pad_sequences(sequences, padding='post', maxlen=expected_sequence_length)

# Function to retrieve model input shape
def get_model_input_shape(model):
    """
    Retrieves the expected input shape of a model.
    """
    return model.input_shape[1]  # Assuming the model is sequential

# Function to load models
def load_models(model_paths):
    """
    Loads models for the detected log type from predefined paths.
    """
    models = []
    for path in model_paths:
        try:
            models.append(load_model(path))
        except Exception as e:
            st.error(f"Failed to load model at {path}: {e}")
            raise
    return models

# Function to generate ensemble predictions
def ensemble_predict(models, inputs):
    """
    Generates predictions using an ensemble of models.
    """
    predictions = [model.predict(inputs) for model in models]
    return np.mean(predictions, axis=0)

# Streamlit app
def main():
    st.title("Log File Anomaly Detection")
    uploaded_file = st.file_uploader("Upload a log file", type=["log", "txt"])
    if uploaded_file:
        log_file = uploaded_file.readlines()
        log_type = detect_log_type(log_file)
        st.write(f"Detected Log Type: **{log_type}**")
        df = parse_logs(log_file, log_type)
        if df.empty:
            st.error("No data parsed from the log file.")
            st.stop()
        st.write("Parsed Data Sample:")
        st.dataframe(df.head())
        if log_type == "linux":
            if "Label" not in df:
                st.error("Linux logs must include a 'Label' column for preprocessing.")
                st.stop()
            linux_inputs, linux_labels = preprocess_linux_logs(df)
            st.write("Linux data preprocessed successfully.")
        else:
            expected_sequence_length = 100  # Adjust based on your model
            processed_data = preprocess_logs(df, log_type, expected_sequence_length)
            st.write(f"Processed Data Shape: {processed_data.shape}")

if __name__ == "__main__":
    main()
