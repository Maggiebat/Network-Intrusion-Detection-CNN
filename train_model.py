from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle
import ipaddress

def clean_data(data):
    """
    Convert IP addresses to integers for model compatibility.
    """
    def ip_to_int(ip_address):
        try:
            return int(ipaddress.ip_address(ip_address))
        except ValueError:
            return None  # Handle invalid IPs as needed

    data['src_ip'] = data['src_ip'].apply(ip_to_int)
    # Exclude 'dst_ip' since it's constant and not informative
    return data

def train_model(data):
    """
    Train the IsolationForest model using specified features.
    """
    features = ['src_ip', 'src_port', 'dst_port', 'protocol', 'length']
    model = IsolationForest()
    model.fit(data[features])
    return model

def save_model(model, filename='ids.pkl'):
    """
    Save the trained IsolationForest model to a file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Trained model is now saved as {filename}")

def save_training_data(data, filename='training_data_with_anomaly_scores.csv'):
    """
    Save the training data with anomaly labels and scores for reference.
    """
    data.to_csv(filename, index=False)
    print(f"Training data with predictions saved as {filename}")

# --- MAIN EXECUTION ---

# 1. Read the input CSV
data = pd.read_csv('2day_capture.csv')

# 2. Clean the data
cleaned_data = clean_data(data)

# 3. Train the Isolation Forest model
model = train_model(cleaned_data)

# 4. Predict on training data to inspect performance
features = ['src_ip', 'src_port', 'dst_port', 'protocol', 'length']
predictions = model.predict(cleaned_data[features])
scores = model.decision_function(cleaned_data[features])

# Add prediction results into the cleaned data
cleaned_data['anomaly'] = predictions
cleaned_data['anomaly_score'] = scores

# 5. Save the trained model
save_model(model)

# 6. Save the training data with scores
save_training_data(cleaned_data)
