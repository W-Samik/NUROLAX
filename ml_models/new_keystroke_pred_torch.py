import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ----- PyTorch Imports -----
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# ---------------------------
from sklearn.metrics import accuracy_score, roc_auc_score # For evaluation
import os
import logging
import glob
import joblib 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MAX_LEN = 150
DATA_BASE_DIR = r'C:\Users\thigh\Downloads\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroqwerty-mit-csxpd-dataset-1.0.0\MIT-CS1PD' # Base Path
GROUND_TRUTH_FILE = os.path.join(DATA_BASE_DIR, r'C:\Users\thigh\Downloads\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroqwerty-mit-csxpd-dataset-1.0.0\MIT-CS1PD\GT_DataPD_MIT-CS1PD.csv')
KEYSTROKE_DATA_DIR = os.path.join(DATA_BASE_DIR, r'C:\Users\thigh\Downloads\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroqwerty-mit-csxpd-dataset-1.0.0\MIT-CS1PD\data_MIT-CS1PD')

# --- PyTorch Specific Config ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4 
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'parkinsons_keystroke_cnn_pytorch_corrected.pth'
SCALER_SAVE_PATH = 'parkinsons_keystroke_scaler_corrected.joblib'

logging.info(f"Using device: {DEVICE}")
logging.info(f"Corrected model will be saved to: {MODEL_SAVE_PATH}")
logging.info(f"Associated scaler will be saved to: {SCALER_SAVE_PATH}")


# ====================================
# 1. Data Loading and Preprocessing
# ====================================

def extract_features_from_real_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None, names=['key', 'press_time', 'release_time', 'hold_csv'])
        for col in ['press_time', 'release_time']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['press_time', 'release_time'], inplace=True)
        df = df[df['press_time'] >= 0]
        df = df[df['release_time'] >= 0]
        df = df.sort_values(by='press_time').reset_index(drop=True)
        if df.empty: return None

        features = []
        prev_release_time = None
        for i, row in df.iterrows():
            press_time = row['press_time']
            release_time = row['release_time']
            hold_time = release_time - press_time
            if hold_time < 0: hold_time = 0.0

            if i > 0:
                if prev_release_time is not None:
                    latency = press_time - prev_release_time
                    if latency < 0: latency = 0.0
                else: latency = 0.0
            else: latency = 0.0

            features.append([hold_time, latency])
            prev_release_time = release_time
        if not features: return None
        return np.array(features, dtype=np.float32)
    except pd.errors.EmptyDataError: return None
    except FileNotFoundError: logging.error(f"FNF: {file_path}"); return None
    except Exception as e: logging.error(f"Error {file_path}: {e}"); return None

def pad_sequences_custom(feature_list, max_len):
    padded = np.zeros((len(feature_list), max_len, 2), dtype=np.float32)
    for i, seq in enumerate(feature_list):
        if seq is not None and seq.ndim == 2 and seq.shape[0] > 0 and seq.shape[1] == 2:
             seq_len = min(len(seq), max_len)
             padded[i,:seq_len,:] = seq[:seq_len]
    return padded

def load_parkinsons_data(gt_file, data_dir, max_len=MAX_LEN):
    logging.info(f"Loading ground truth from: {gt_file}")
    try:
        try: gt_df = pd.read_csv(gt_file).set_index('pID')
        except UnicodeDecodeError: gt_df = pd.read_csv(gt_file, encoding='iso-8859-1').set_index('pID')
        logging.info(f"Loaded GT dataframe with shape: {gt_df.shape}")
    except FileNotFoundError: logging.error(f"GT file FNF: {gt_file}"); return None, None
    except Exception as e: logging.error(f"Error loading GT CSV: {e}"); return None, None

    all_features = []
    all_labels = []
    processed_files_count = 0
    logging.info("Processing keystroke files...")
    # Determine file columns dynamically - safer
    file_cols = [col for col in gt_df.columns if col.startswith('file_')]
    if not file_cols:
         logging.error("No columns starting with 'file_' found in GT file.")
         return None, None

    for pID, row in gt_df.iterrows():
        # Check if 'gt' column exists and is valid boolean/int
        if 'gt' not in row or pd.isna(row['gt']):
             logging.warning(f"Skipping pID {pID}: Missing or invalid 'gt' label.")
             continue
        label = 1 if row['gt'] else 0 # Convert True/False/1/0 to 1 or 0

        for file_col in file_cols:
            if file_col in row and pd.notna(row[file_col]):
                file_name = str(row[file_col]).strip() # Ensure it's a string and strip spaces
                if not file_name: continue # Skip empty file names
                file_path = os.path.join(data_dir, file_name)

                if not os.path.exists(file_path):
                    # Reduce verbosity, maybe log only once per pID if needed
                    # logging.warning(f"File not found: {file_path}")
                    continue

                features = extract_features_from_real_csv(file_path)
                if features is not None and features.shape[0] > 0:
                    all_features.append(features)
                    all_labels.append(label)
                    processed_files_count += 1

    if not all_features: logging.error("No valid feature data loaded."); return None, None
    logging.info(f"Successfully processed {processed_files_count} files. Padding {len(all_features)} sequences...")
    X_padded = pad_sequences_custom(all_features, max_len)
    y = np.array(all_labels, dtype=np.float32) # BCELoss needs float labels

    if X_padded.shape[0] == 0: logging.error("Padding resulted in zero sequences."); return None, None
    if X_padded.shape[0] != y.shape[0]: # Sanity check
         logging.error(f"Mismatch between number of sequences ({X_padded.shape[0]}) and labels ({y.shape[0]}) after padding.")
         return None, None

    logging.info(f"Finished processing. Final X shape: {X_padded.shape}, y shape: {y.shape}")
    # Check label distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    logging.info(f"Label distribution: {dict(zip(unique_labels, counts))}")
    if len(unique_labels) < 2:
        logging.warning("Only one class present in the loaded data! Training/Evaluation might not be meaningful.")

    return X_padded, y


# --- Load Real Data ---
X, y = load_parkinsons_data(GROUND_TRUTH_FILE, KEYSTROKE_DATA_DIR, max_len=MAX_LEN)
if X is None or y is None: logging.error("Failed to load data."); exit()
if X.shape[0] < 10: logging.warning(f"Few samples loaded ({X.shape[0]}).")

# --- Split Data ---
if len(np.unique(y)) < 2:
    logging.warning("Only one class present. Cannot stratify split. Splitting without stratify.")
    stratify_param = None
else:
    stratify_param = y

# Check if enough samples exist for the split
if X.shape[0] < 5: # Need at least 1 sample in test set (assuming 0.2 test size)
     logging.error("Not enough samples to perform train/test split.")
     exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_param, random_state=42)
logging.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
if X_train.shape[0] == 0 or (stratify_param is not None and X_test.shape[0] == 0): # Test can be empty if not stratifying small datasets
    logging.error("Empty train or test split."); exit()

# Re-check labels after split
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
logging.info(f"Labels - Train: {dict(zip(unique_train, counts_train))}, Test: {dict(zip(unique_test, counts_test))}")


# --- Normalize features ---
scaler = StandardScaler()
if X_train.size > 0:
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1]) # Reshape to (n_samples * seq_len, n_features)
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train = X_train_scaled_reshaped.reshape(X_train.shape) # Reshape back
    logging.info("Scaler fitted on training data and applied.")

    # Save the scaler using the NEW path
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logging.info(f"Saved fitted scaler to {SCALER_SAVE_PATH}")
else:
    logging.error("X_train empty, cannot fit scaler."); exit()

if X_test.size > 0:
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test = X_test_scaled_reshaped.reshape(X_test.shape)
    logging.info("Scaler applied to test data.")
else:
     logging.warning("X_test is empty after splitting. Evaluation cannot run.")


# ====================================
# 1b. PyTorch Dataset and DataLoader (Functions as provided)
# ====================================
class KeystrokeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        # Permute features for Conv1d: (channels/features, sequence_length)
        return self.features[idx].permute(1, 0), self.labels[idx]

train_dataset = KeystrokeDataset(X_train, y_train)
# Only create test_dataset/loader if X_test is not empty
if X_test.size > 0:
    test_dataset = KeystrokeDataset(X_test, y_test)
else:
    test_dataset = None # Explicitly set to None

num_workers = 0 # Set to 0 for safer debugging, especially on Windows
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE == torch.device('cuda') else False)

if test_dataset:
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE == torch.device('cuda') else False)
else:
    test_loader = None # Explicitly set to None

logging.info("PyTorch Datasets and DataLoaders created.")

# ====================================================================
# 2. Temporal CNN Model (MODIFIED ARCHITECTURE - BASED ON ERROR LOGS)
# ====================================================================
class KeystrokeCNN_PyTorch_Corrected(nn.Module): # Use corrected class name
    def __init__(self, n_features=2, sequence_length=MAX_LEN):
        super().__init__()
        # --- Layer Block 1 ---
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=3, padding=1) # out=128
        self.bn1 = nn.BatchNorm1d(num_features=128) # Added bn1
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3) # Keeping dropout rates from original example

        # --- Layer Block 2 ---
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # in=128, out=256
        self.bn2 = nn.BatchNorm1d(num_features=256) # Added bn2
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        # --- Layer Block 3 (Guessed structure based on error keys 'conv3'/'bn3') ---
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1) # Guessed out=512
        self.bn3 = nn.BatchNorm1d(num_features=512) # Added bn3
        # Assuming no pooling after conv3 based on common architectures for this depth
        self.dropout_conv3 = nn.Dropout(0.3) # Added dropout

        # --- Flatten & Dense Layers ---
        # Calculate flattened size after conv/pool/bn blocks
        final_seq_len = sequence_length // 4 # Two pooling layers with kernel=2
        final_channels = 512 # Output channels from the last conv/bn block (conv3/bn3)
        self.flattened_size = final_channels * final_seq_len
        logging.info(f"Corrected Model: Calculated flattened size = {final_channels} channels * {final_seq_len} length = {self.flattened_size}")

        if self.flattened_size <= 0:
            raise ValueError(f"Calculated flattened size ({self.flattened_size}) is not positive. Check MAX_LEN ({sequence_length}) vs Pooling.")

        self.fc1 = nn.Linear(self.flattened_size, 64) # Adjusted input size
        self.dropout_fc1 = nn.Dropout(0.4) # Using dropout from original Dense block
        self.fc2 = nn.Linear(64, 1) # Final output layer

    def forward(self, x):
        # x shape: (batch, n_features, sequence_length)
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)   # Apply BatchNorm after Conv
        x = F.relu(x)     # Apply Activation after BN
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # No pooling assumed here
        x = self.dropout_conv3(x)

        # Flatten for Dense layers
        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        # Final Activation (Sigmoid for Binary Cross Entropy)
        x = torch.sigmoid(x)

        # Squeeze the last dimension (output is shape [batch_size, 1]) -> [batch_size]
        return x.squeeze(-1)


# --- Initialize model using the CORRECTED class ---
model = KeystrokeCNN_PyTorch_Corrected(n_features=2, sequence_length=MAX_LEN).to(DEVICE)
logging.info("PyTorch Model Initialized (Corrected Architecture).")

# ============================
# 3. Training Loop (Functions as provided)
# ============================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': []}
best_val_loss = float('inf')
patience_counter = 0

logging.info("Starting PyTorch model training (Corrected Architecture)...")
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    history['train_loss'].append(epoch_train_loss)

    # --- Validation Step ---
    epoch_val_loss = float('inf') # Default if no validation occurs
    val_auc = 0.0
    val_accuracy = 0.0

    if test_loader: # Only validate if test_loader exists
        model.eval()
        running_val_loss = 0.0
        all_preds_val = []
        all_labels_val = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss_item = criterion(outputs, labels) # Calculate loss per batch
                running_val_loss += val_loss_item.item() * inputs.size(0)
                all_preds_val.extend(outputs.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        # Calculate average validation loss only if dataset is not empty
        if len(test_loader.dataset) > 0:
             epoch_val_loss = running_val_loss / len(test_loader.dataset)
             history['val_loss'].append(epoch_val_loss) # Append loss if calculated

             # Calculate metrics only if predictions were made
             if len(all_labels_val) > 0 and len(all_preds_val) > 0:
                 try:
                     val_auc = roc_auc_score(all_labels_val, all_preds_val)
                     val_preds_binary = (np.array(all_preds_val) > 0.5).astype(int)
                     val_accuracy = accuracy_score(all_labels_val, val_preds_binary)
                 except ValueError as e: # Handle case of single class in batch/data
                     logging.warning(f"Ep {epoch+1} Val Metrics Warning: {e}")
                     val_auc = 0.0
                     val_accuracy = 0.0 # Assign default value
             else:
                 logging.warning(f"Ep {epoch+1}: No predictions made during validation.")
                 val_auc = 0.0; val_accuracy = 0.0
        else:
             logging.warning(f"Ep {epoch+1}: Test dataset has zero length, skipping validation loss calculation.")
             # Append NaN or previous value to keep history lists aligned if needed, or handle later
             # history['val_loss'].append(float('nan')) # Example

    else:
         logging.info(f"Epoch {epoch+1}/{EPOCHS} => Train Loss: {epoch_train_loss:.4f} | No Validation Data")
         # No validation loss to check for early stopping if no test_loader
         # You might want to save the model based on training loss or just save the last epoch
         # Example: Save last epoch model if no validation
         # torch.save(model.state_dict(), MODEL_SAVE_PATH)
         continue # Skip early stopping check if no validation loss


    # Append metrics whether calculated or default
    history['val_auc'].append(val_auc)
    history['val_accuracy'].append(val_accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} => Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_accuracy:.4f}")

    # --- Early Stopping & Model Saving (Checks if validation loss improved) ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save corrected model state
        logging.info(f"Val loss decreased ({best_val_loss:.4f}). Model saved to {MODEL_SAVE_PATH}")
        patience_counter = 0
    else:
        patience_counter += 1
        logging.info(f"Val loss did not improve for {patience_counter} epochs. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            logging.info("Early stopping triggered.")
            break

logging.info("Model training finished.")

# --- Load best model saved during training ---
# Ensure the model instance here is the same architecture class used for training
final_model = KeystrokeCNN_PyTorch_Corrected(n_features=2, sequence_length=MAX_LEN).to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    try:
        final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        logging.info(f"Loaded best model weights from {MODEL_SAVE_PATH} for final evaluation.")
    except Exception as e:
        logging.error(f"Error loading saved model state_dict: {e}. Evaluating with potentially last epoch weights.")
        final_model = model # Fallback to the model variable from the end of training loop
else:
     logging.warning(f"Model file {MODEL_SAVE_PATH} not found after training. Using weights from end of training.")
     final_model = model # Use model from end of training loop

# ============================
# 4. Evaluation (Functions as provided)
# ============================
if test_loader: # Check if test_loader exists
    logging.info("\nEvaluating final model on test data...")
    final_model.eval() # Ensure eval mode
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = final_model(inputs) # Use the loaded final_model
            all_preds_test.extend(outputs.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    if len(all_labels_test) > 0 and len(all_preds_test) > 0:
        try:
             test_auc = roc_auc_score(all_labels_test, all_preds_test)
             test_preds_binary = (np.array(all_preds_test) > 0.5).astype(int)
             test_accuracy = accuracy_score(all_labels_test, test_preds_binary)
             print(f"\nFinal Test AUC: {test_auc:.4f}")
             print(f"Final Test Accuracy: {test_accuracy:.4f}")
             # Consider adding: from sklearn.metrics import classification_report; print(classification_report(all_labels_test, test_preds_binary))
        except ValueError as e: print(f"Could not calculate final test metrics: {e}")
    else:
        print("No predictions/labels collected during final evaluation.")
else:
    logging.warning("No test data/loader available for final evaluation.")

# ==============================
# 5. Inference (Can be kept for quick test, but best in separate script)
# ==============================
def predict_from_file_pytorch(file_path, fitted_scaler, model_instance, max_len=MAX_LEN, device=DEVICE):
    # (Ensure this function exists as provided before if you keep this section)
    logging.info(f"Predicting from file: {file_path}")
    model_instance.eval()
    features_np = extract_features_from_real_csv(file_path)
    if features_np is None or features_np.shape[0] == 0: return None
    padded_np = pad_sequences_custom([features_np], max_len=max_len)
    if padded_np.shape[0] == 0: return None
    try: normalized_np = fitted_scaler.transform(padded_np.reshape(-1, 2)).reshape(padded_np.shape)
    except Exception as e: logging.error(f"Scaler err predict: {e}"); return None
    input_tensor = torch.tensor(normalized_np, dtype=torch.float32).permute(0, 2, 1).to(device)
    try:
        with torch.no_grad(): prediction = model_instance(input_tensor)
        return round(prediction.item(), 3)
    except Exception as e: logging.error(f"Predict err: {e}"); return None

# --- Optional Inference Example ---
if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
    logging.info(f"\n--- Optional: Loading Final Model and Scaler for Inference Example ---")
    try:
        # Re-initialize the CORRECTED architecture for inference
        inference_model = KeystrokeCNN_PyTorch_Corrected(n_features=2, sequence_length=MAX_LEN).to(DEVICE)
        inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        inference_model.eval()
        inference_scaler = joblib.load(SCALER_SAVE_PATH)
        logging.info("Final model and scaler loaded for inference example.")

        # Find sample files (make sure KEYSTROKE_DATA_DIR is correct and has CSVs)
        example_csv_files = glob.glob(os.path.join(KEYSTROKE_DATA_DIR, '*.csv')) # Check correct dir
        if example_csv_files:
            sample_file_path_to_test = example_csv_files[0]
            logging.info(f"--- Running Inference Example on file: {sample_file_path_to_test} ---")
            risk_score = predict_from_file_pytorch(sample_file_path_to_test, inference_scaler, inference_model, max_len=MAX_LEN, device=DEVICE)
            if risk_score is not None: print(f"Predicted Risk Score: {risk_score:.3f}")
            else: print("Inference prediction failed.")
        else: print("\nNo CSV files found in KEYSTROKE_DATA_DIR for inference example.")
    except Exception as e: print(f"Error loading final model/scaler for inference: {e}")
else: print("\nFinal model or scaler file not found. Skipping inference example.")

print(f"\nScript finished. Corrected model: '{MODEL_SAVE_PATH}', Scaler: '{SCALER_SAVE_PATH}'")

# --- END OF FILE ---