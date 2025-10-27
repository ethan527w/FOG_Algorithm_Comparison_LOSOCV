import pandas as pd
import numpy as np
import os
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# Ensure TensorFlow is imported before TCN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# Import TCN after TensorFlow
try:
    from tcn import TCN
except ImportError:
    print("Error: keras-tcn library not found. Please install it using: pip install keras-tcn")
    exit()

def create_prediction_dataset(file_paths, params):
    all_X, all_y = [], []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1)
            df.columns = [
                'Time', 'Acc_X', 'Acc_Y', 'Acc_Z',
                'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'FoG_Label'
            ]

            for col in params['feature_columns']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            if df.empty:
                continue

            features = df[params['feature_columns']].values
            labels = df[params['label_column']].values

            i = 0
            while i + params['input_window_points'] + params['prediction_window_points'] <= len(features):
                input_end = i + params['input_window_points']
                X_window = features[i:input_end]

                prediction_start = input_end
                prediction_end = prediction_start + params['prediction_window_points']
                prediction_window_labels = labels[prediction_start:prediction_end]

                label = 1 if np.sum(prediction_window_labels) > 0 else 0

                all_X.append(X_window)
                all_y.append(label)

                i += params['step_size_points']
        except Exception as e:
            print(f"\nAn error occurred while processing {os.path.basename(file_path)}: {e}")

    return np.array(all_X), np.array(all_y)

def build_lstm_tcn_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.5),
        TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=False # Make sure this is False if the next layer is Dense(1)
        ),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    params = {
        'sampling_rate': 200,
        'input_window_sec': 4.0,
        'prediction_window_sec': 2.0,
        'step_size_sec': 0.5,
        'feature_columns': ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'label_column': 'FoG_Label',
    }
    params['input_window_points'] = int(params['input_window_sec'] * params['sampling_rate'])
    params['step_size_points'] = int(params['step_size_sec'] * params['sampling_rate'])
    params['prediction_window_points'] = int(params['prediction_window_sec'] * params['sampling_rate'])
    input_shape = (params['input_window_points'], len(params['feature_columns']))

    input_directory = r'C:\Users\ethan\PycharmProjects\FOGDetection\data\dataset'

    input_path = pathlib.Path(input_directory)
    subject_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('S')])

    if not subject_dirs:
        print(f"Error: No subject folders (e.g., S01, S02) found in {input_directory}")
        exit()

    print(f"Found {len(subject_dirs)} subjects. Starting Leave-One-Subject-Out Cross-Validation (LOSO-CV)...")

    all_fold_results = []

    for i, test_subject_dir in enumerate(subject_dirs):
        test_subject_id = test_subject_dir.name
        print("\n" + "="*80)
        print(f"Fold {i+1}/{len(subject_dirs)}: Holding out Subject {test_subject_id}")
        print("="*80)

        train_files = []
        test_files = []

        for subject_dir in subject_dirs:
            files = list(subject_dir.rglob('*.csv'))
            if subject_dir.name == test_subject_id:
                test_files.extend(files)
            else:
                train_files.extend(files)

        if not test_files:
            print(f"Warning: No test files found for subject {test_subject_id}. Skipping this fold.")
            continue
        if not train_files:
            print("Error: No training files found. Cannot proceed.")
            break

        print(f"Loading training data from {len(train_files)} files...")
        X_train, y_train = create_prediction_dataset(train_files, params)

        print(f"Loading test data from {len(test_files)} files...")
        X_test, y_test = create_prediction_dataset(test_files, params)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: No valid samples generated for this fold. Skipping.")
            continue

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
        X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        print("Data scaling complete.")

        model = build_lstm_tcn_model(input_shape)

        # --- THE FIX: Change file extension to .keras ---
        model_save_path = f'best_model_fold_{test_subject_id}.keras'
        # -----------------------------------------------

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            # ModelCheckpoint will now save in the native Keras format
            ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        print("Splitting training data for validation...")
        # Use a slightly larger validation split if memory allows, helps stabilize training
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

        print(f"Starting model training for Fold {i+1}...")
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=30,
            batch_size=64,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=1
        )

        print(f"Evaluating model on held-out subject {test_subject_id}...")
        # Load using custom_objects, although .keras format might not strictly need it, it's safer
        best_model = load_model(model_save_path, custom_objects={'TCN': TCN})

        y_pred_proba = best_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype("int32")

        report = classification_report(y_test, y_pred, target_names=['No FOG', 'Impending FOG'], output_dict=True, zero_division=0)

        fold_metrics = report['Impending FOG']
        fold_metrics['accuracy'] = report['accuracy']
        all_fold_results.append(fold_metrics)

        print(f"\n--- Results for Fold {test_subject_id} ---")
        print(f"Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"Impending FOG Recall: {fold_metrics['recall']:.4f}")
        print(f"Impending FOG Precision: {fold_metrics['precision']:.4f}")
        print(f"Impending FOG F1-Score: {fold_metrics['f1-score']:.4f}")

        # Clean up the saved model file for the fold
        try:
            os.remove(model_save_path)
        except OSError as e:
            print(f"Error removing temporary model file {model_save_path}: {e}")


    print("\n" + "="*80)
    print("      Leave-One-Subject-Out Cross-Validation (LOSO-CV) Final Report      ")
    print(f"                       Model: LSTM-TCN - {len(all_fold_results)} Folds Completed                      ")
    print("="*80)

    if all_fold_results:
        df_results = pd.DataFrame(all_fold_results)

        print("\n--- Aggregated Metrics (Mean ± Std. Deviation) ---")
        print(f"Overall Accuracy: {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
        print(f"FOG Recall:       {df_results['recall'].mean():.4f} ± {df_results['recall'].std():.4f}")
        print(f"FOG Precision:    {df_results['precision'].mean():.4f} ± {df_results['precision'].std():.4f}")
        print(f"FOG F1-Score:     {df_results['f1-score'].mean():.4f} ± {df_results['f1-score'].std():.4f}")

        print("\n--- Full Results per Fold ---")
        # Format the DataFrame output for better readability
        print(df_results.to_string(index=False, float_format="%.4f"))
    else:
        print("No folds were successfully completed. Please check your data and paths.")