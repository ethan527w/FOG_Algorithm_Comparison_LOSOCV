import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from scipy.signal import welch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from tqdm import tqdm


def calculate_psd_and_band_power(window_data, fs, nfft, loco_band, freeze_band):
    freqs, psd = welch(window_data, fs=fs, nperseg=nfft, window='hann')
    loco_idx = np.where((freqs >= loco_band[0]) & (freqs <= loco_band[1]))[0]
    freeze_idx = np.where((freqs >= freeze_band[0]) & (freqs <= freeze_band[1]))[0]
    power_loco = np.sum(psd[loco_idx])
    power_freeze = np.sum(psd[freeze_idx])
    return power_loco, power_freeze


def process_single_file(file_path, params):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'FoG_Label']

        required_columns = [params['signal_column'], params['label_column'], params['time_column']]
        if not all(col in df.columns for col in required_columns):
            return None, None

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        signal = df[params['signal_column']].values
        labels = df[params['label_column']].values
        time = df[params['time_column']].values
    except Exception as e:
        print(f"\nError reading or parsing file {filename}: {e}")
        return None, None

    results = []
    start = 0
    while start + params['window_length_points'] <= len(signal):
        end = start + params['window_length_points']
        window_signal = signal[start:end]
        window_labels = labels[start:end]
        power_loco, power_freeze = calculate_psd_and_band_power(window_signal, params['sampling_rate'], params['nfft'],
                                                                params['loco_band'], params['freeze_band'])
        if power_loco == 0: power_loco = 1e-9
        freeze_index = power_freeze / power_loco
        total_power = power_loco + power_freeze
        true_label = 1 if np.sum(window_labels) > 0 else 0
        results.append({'start_time': time[start], 'end_time': time[end - 1], 'freeze_index': freeze_index,
                        'total_power': total_power, 'true_label': true_label})
        start += params['step_size_points']
    if not results: return None, None
    return pd.DataFrame(results), df


def find_optimal_thresholds(results_df):
    if results_df is None or results_df.empty: return 0, 0, 0, []
    power_th_range = np.linspace(results_df['total_power'].min(), results_df['total_power'].quantile(0.5), 20)
    freeze_th_range = np.linspace(results_df['freeze_index'].min(), results_df['freeze_index'].quantile(0.95), 20)
    best_f1 = -1
    best_thresholds = (0, 0)
    true_labels = results_df['true_label'].values
    for pow_th in power_th_range:
        for fi_th in freeze_th_range:
            predictions = ((results_df['total_power'] > pow_th) & (results_df['freeze_index'] > fi_th)).astype(int)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = (pow_th, fi_th)
    best_pow_th, best_fi_th = best_thresholds
    final_predictions = ((results_df['total_power'] > best_pow_th) & (results_df['freeze_index'] > fi_th)).astype(int)
    return best_pow_th, best_fi_th, best_f1, final_predictions


def plot_and_get_metrics(file_path, original_df, results_df, predictions, best_fi_th, output_dir, params):
    filename = os.path.basename(file_path)
    true_labels = results_df['true_label'].values

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)

    try:
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        if all(true_labels == 0) and all(predictions == 0):
            tn = len(true_labels)
        elif all(true_labels == 1) and all(predictions == 1):
            tp = len(true_labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'FD Algorithm Detection Result: {filename}', fontsize=18)
    ax1.plot(original_df[params['time_column']], original_df[params['signal_column']], label=params['signal_column'],
             color='darkgray', linewidth=1)
    ax1.set_title('Original Acceleration Signal')
    ax1.set_ylabel('Acceleration [m/s^2]')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(results_df['start_time'], results_df['freeze_index'], label='Freeze Index (FI)', color='dodgerblue',
             marker='.', linestyle='-', markersize=4)
    ax2.axhline(y=best_fi_th, color='r', linestyle='--', label=f'Optimal FreezeTH: {best_fi_th:.2f}')
    ax2.set_title('Freeze Index and Detection Outcome')
    ax2.set_ylabel('Freeze Index')
    ax2.set_xlabel('Time [s]')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)

    for i, row in results_df.iterrows():
        start_t, end_t = row['start_time'], row['end_time']
        true = row['true_label'];
        pred = predictions[i]
        if pred == true:
            color, alpha = 'green', 0.15
        elif pred == 1 and true == 0:
            color, alpha = 'red', 0.2
        elif pred == 0 and true == 1:
            color, alpha = 'yellow', 0.3
        ax1.axvspan(start_t, end_t, color=color, alpha=alpha, ec='none')
        ax2.axvspan(start_t, end_t, color=color, alpha=alpha, ec='none')

    ax1.legend(loc='upper right');
    ax2.legend(loc='upper right')

    stats_text = (f"Detection Accuracy: {accuracy:.2%}\n"
                  f"F1-Score: {f1:.3f}\n"
                  f"TP: {tp}, FN (Type II): {fn}, FP (Type I): {fp}, TN: {tn}")

    fig.text(0.5, 0.93, stats_text, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_FD_Result.png")
    plt.savefig(output_path)
    plt.close(fig)

    return {
        'filename': filename, 'accuracy': accuracy, 'f1_score': f1,
        'precision': precision, 'recall': recall,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
    }


if __name__ == '__main__':
    params = {
        'sampling_rate': 200,
        'window_length_sec': 4.0,
        'step_size_sec': 0.5,
        'loco_band': [0.5, 3],
        'freeze_band': [3, 8],
        'signal_column': 'Acc_Y',
        'label_column': 'FoG_Label',
        'time_column': 'Time'
    }
    params['window_length_points'] = int(params['window_length_sec'] * params['sampling_rate'])
    params['step_size_points'] = int(params['step_size_sec'] * params['sampling_rate'])
    params['nfft'] = params['window_length_points']

    input_directory = r'C:\Users\ethan\PycharmProjects\FOGDetection\data\dataset'
    output_directory = r'C:\Users\ethan\PycharmProjects\FOGDetection\data\Exp1_FD_Results'

    input_path = pathlib.Path(input_directory)
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(input_path.rglob('*.csv')))

    all_results_summary = []

    if not csv_files:
        print(f"Error: No .csv files found in the specified directory: {input_directory}")
    else:
        print(f"Experiment 1: FD Algorithm Started. Found {len(csv_files)} files.")

        for file in tqdm(csv_files, desc="Processing Files"):
            results_df, original_df = process_single_file(file, params)
            if results_df is None or results_df.empty: continue

            best_pow_th, best_fi_th, best_f1, final_predictions = find_optimal_thresholds(results_df)

            file_summary = plot_and_get_metrics(file, original_df, results_df, final_predictions, best_fi_th,
                                                output_directory, params)
            if file_summary:
                all_results_summary.append(file_summary)

        print(f"\nExperiment 1 Finished. All plots saved to: {output_directory}")

    if all_results_summary:
        summary_df = pd.DataFrame(all_results_summary)

        avg_accuracy = summary_df['accuracy'].mean()
        std_accuracy = summary_df['accuracy'].std()
        avg_f1 = summary_df['f1_score'].mean()
        std_f1 = summary_df['f1_score'].std()

        total_tp = summary_df['tp'].sum()
        total_fn = summary_df['fn'].sum()
        total_fp = summary_df['fp'].sum()
        total_tn = summary_df['tn'].sum()

        micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (
                                                                                                            total_tp + total_tn + total_fp + total_fn) > 0 else 0
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (
                                                                                                          micro_precision + micro_recall) > 0 else 0

        best_file = summary_df.loc[summary_df['f1_score'].idxmax()]

        print("\n" + "=" * 60)
        print("     Experiment 1: Frequency-Domain (FD) Overall Report")
        print("=" * 60)
        print(f"Successfully processed {len(summary_df)} valid files.")

        print("\n--- Macro Averages (Performance across files) ---")
        print(f"Average Accuracy: {avg_accuracy:.2%} (± {std_accuracy:.2%})")
        print(f"Average F1-Score: {avg_f1:.3f} (± {std_f1:.3f})")

        print("\n--- Micro Averages (Performance across all windows) ---")
        print(f"Overall Accuracy: {micro_accuracy:.2%}")
        print(f"Overall F1-Score: {micro_f1:.3f}")

        print("\n--- Overall Confusion Matrix ---")
        print(f"True Positives (TP):  {total_tp}")
        print(f"False Negatives (FN): {total_fn}")
        print(f"False Positives (FP): {total_fp}")
        print(f"True Negatives (TN):  {total_tn}")

        print("\n" + "*" * 60)
        print("                🏆 Best Performing File 🏆")
        print("*" * 60)
        print(f"Filename: {best_file['filename']}")
        print(f"F1-Score: {best_file['f1_score']:.4f}")
        print(f"Accuracy: {best_file['accuracy']:.4f}")
        print(f"Precision: {best_file['precision']:.4f}")
        print(f"Recall: {best_file['recall']:.4f}")
        print("=" * 60)
    else:
        print("\nNo valid files were processed. Could not generate a performance report.")