import streamlit as st
import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Preprocessing Function
def parse_and_preprocess_linux_logs(log_file, model_input_length):
    """
    Parse and preprocess Linux log files for prediction.
    """
    # Define regex pattern for Linux logs
    log_pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\w+)\s+(\w+):\s+(.*)'

    data = []
    for line in log_file:
        match = re.match(log_pattern, line.decode('ISO-8859-1', errors='ignore'))
        if match:
            data.append({
                "Timestamp": match.group(1),
                "Host": match.group(2),
                "Log_Level": match.group(3),
                "Message": match.group(4)
            })

    # Convert parsed data to a DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        st.error("No valid log lines detected. Please check the log file format.")
        return None, None

    # Encode categorical features
    df['Host_Encoded'] = LabelEncoder().fit_transform(df['Host'])
    df['Log_Level_Encoded'] = LabelEncoder().fit_transform(df['Log_Level'])

    # Tokenize and pad the 'Message' column
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df['Message'])
    X_message = tokenizer.texts_to_sequences(df['Message'])
    X_message_padded = pad_sequences(X_message, padding='post', maxlen=model_input_length - 2)

    # Combine features
    X_host = np.array(df['Host_Encoded']).reshape(-1, 1)
    X_log_level = np.array(df['Log_Level_Encoded']).reshape(-1, 1)
    X_combined = np.concatenate([X_host, X_log_level, X_message_padded], axis=1)
    X_combined = X_combined.reshape((X_combined.shape[0], X_combined.shape[1], 1))

    return X_combined, df


# Streamlit App
def main():
    st.title("Linux Log File Anomaly Detection")

    # File upload
    uploaded_file = st.file_uploader("Upload a Linux log file (.log)", type=["log"])
    if uploaded_file:
        log_file = uploaded_file.readlines()
        st.success("Log file uploaded successfully!")

        # Load pre-trained CNN model
        model_path = "models/cnn_model_new_linux.h5"
        st.write("Loading pre-trained model...")
        try:
            model = load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        # Parse and preprocess logs
        st.write("Parsing and preprocessing logs...")
        model_input_length = model.input_shape[1]
        X_data, original_df = parse_and_preprocess_linux_logs(log_file, model_input_length)

        if X_data is None or original_df is None:
            st.stop()

        # Perform predictions
        st.write("Performing predictions...")
        try:
            predictions = model.predict(X_data)
            original_df["Prediction"] = (predictions > 0.5).astype(int)

            # Display predictions
            st.write("Predictions:")
            st.dataframe(original_df)

            # Option to download predictions
            output_csv_path = "linux_log_predictions.csv"
            original_df.to_csv(output_csv_path, index=False)
            st.download_button(
                label="Download Predictions",
                data=original_df.to_csv(index=False),
                file_name="linux_log_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.info("Please upload a log file to proceed.")


if __name__ == "__main__":
    main()