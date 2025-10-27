import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_prediction_dataset(file_paths, params):
    all_X, all_y = [], []

    for file_path in tqdm(file_paths, desc="Loading and Processing Files"):
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


def build_deeper_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Dropout(0.5),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\nModel Architecture:")
    model.summary()

    return model


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


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

    input_directory = r'C:\Users\ethan\PycharmProjects\FOGDetection\data\dataset'

    input_path = pathlib.Path(input_directory)
    all_files = sorted(list(input_path.rglob('*.csv')))

    X, y = create_prediction_dataset(all_files, params)

    if len(X) == 0:
        print("\nDataset creation failed. No samples were generated. Please check the data files and paths.")
    else:
        print(f"\nDataset creation complete.")
        print(f"Total samples: {len(X)}")
        print(f"Input data shape (X): {X.shape}")
        print(f"Label data shape (y): {y.shape}")
        print(f"Positive (FOG) class proportion: {np.mean(y):.2%}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)

        X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        print("\nData splitting and scaling complete.")
        print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        input_shape = (params['input_window_points'], len(params['feature_columns']))
        model = build_deeper_cnn_model(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ModelCheckpoint('best_deeper_cnn_model.h5', monitor='val_loss', save_best_only=True, verbose=0)
        ]

        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        print("\n--- Model Evaluation ---")
        plot_training_history(history)

        best_model = tf.keras.models.load_model('best_deeper_cnn_model.h5')

        y_pred_proba = best_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype("int32")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No FOG', 'Impending FOG']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No FOG', 'Impending FOG'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()