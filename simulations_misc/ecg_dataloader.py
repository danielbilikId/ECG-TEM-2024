import os
import scipy.io
import numpy as np
from scipy.signal import find_peaks
from simulations_misc.config import ECG_PATH
np.random.seed(32)

def ecg_dataloader(path=ECG_PATH):
    """
    Loads ECG data from the specified path, filters for specific subjects and conditions.
    """
    # Mapping each condition to its specific index in the filename
    conditions = {
        'resting': 1,
        'Valsalva': 2,
        'TiltUp': 3,
        'TiltDown': 4
    }
    
    # Subjects to load
    subjects = ['GDN0001', 'GDN0005', 'GDN0010']
    
    data = {}
    for subject in subjects:
        subject_path = os.path.join(path, subject)
        if not os.path.exists(subject_path):
            print(f"Path does not exist: {subject_path}")
            continue
            
        data[subject] = {}
        
        for condition, index in conditions.items():
            # Define file path with condition-specific index
            filename = f"{subject}_{index}_{condition}.mat"
            file_path = os.path.join(subject_path, filename)
            
            # Load .mat file if it exists
            if os.path.isfile(file_path):
                mat_data = scipy.io.loadmat(file_path)
                
                # Extract ECG signal from tfm_ecg2
                ecg_data = mat_data.get('tfm_ecg2', None)
                if ecg_data is not None:
                    data[subject][condition] = ecg_data.flatten()  # Flatten to 1D array
                else:
                    print(f"ECG data ('tfm_ecg2') not found in {filename}.")
            else:
                conditions = {
                    'resting': 1,
                    'Valsalva': 2,
                    'Appnea': 3,
                    'TiltUp': 4,
                    'TiltDown': 5
                    }
                for condition, index in conditions.items():
                    filename = f"{subject}_{index}_{condition}.mat"
                    file_path = os.path.join(subject_path, filename)
                        # Load .mat file if it exists
                    if os.path.isfile(file_path):
                        mat_data = scipy.io.loadmat(file_path)
                        
                        # Extract ECG signal from tfm_ecg2
                        ecg_data = mat_data.get('tfm_ecg2', None)
                        if ecg_data is not None:
                            data[subject][condition] = ecg_data.flatten()  # Flatten to 1D array
                        else:
                            print(f"ECG data ('tfm_ecg2') not found in {filename}.")
    return data

def select_pulse(ecg_signal, pulse_length=2000, fs=2000):
    """
    Extracts a pulse centered on the R peak from the ECG signal.
    
    Parameters:
    - ecg_signal: np.ndarray, the ECG signal.
    - pulse_length: int, the total number of samples in the pulse.
    - fs: int, the sampling frequency of the ECG signal (in Hz).
    
    Returns:
    - np.ndarray, the selected ECG pulse centered on an R peak if valid.
    """
    total_length = len(ecg_signal)
    pulse_length = int(pulse_length)  # Ensure pulse_length is an integer
    
    # Check if the pulse length is appropriate
    if total_length < pulse_length:
        raise ValueError(f"ECG signal length {total_length} is shorter than required pulse length {pulse_length}.")

    # Detect peaks in the ECG signal
    # Using a high prominence threshold to isolate the R peaks
    peaks, properties = find_peaks(ecg_signal, distance=0.6 * fs, prominence=0.5)

    if len(peaks) == 0:
        raise ValueError("No prominent R peak found in the ECG signal.")
    
    # Identify the highest peak, assumed to be the R peak
    r_peak = peaks[np.argmax(ecg_signal[peaks])]

    # Calculate window start and end indices
    half_pulse = pulse_length // 2
    start_index = max(0, r_peak - half_pulse)
    end_index = min(total_length, r_peak + half_pulse)

    # Ensure the selected window is exactly pulse_length by adjusting start/end if near edges
    if end_index - start_index < pulse_length:
        if start_index == 0:
            end_index = start_index + pulse_length
        elif end_index == total_length:
            start_index = end_index - pulse_length
    
    # Extract the centered pulse
    pulse = ecg_signal[start_index:end_index]

    # Validate the extracted pulse length
    if len(pulse) != pulse_length:
        raise ValueError("The extracted pulse length does not match the requested pulse length.")

    return pulse

def select_pulse_statistic_approach(ecg_signal, pulse_length=2000, fs=2000):
    """
    Extracts a validated pulse of a specified length from the ECG signal,
    ensuring it holds the typical ECG PQRST pattern.
    
    Parameters:
    - ecg_signal: np.ndarray, the ECG signal.
    - pulse_length: int, number of samples in the pulse.
    - fs: int, sampling frequency of the ECG signal (in Hz).
    
    Returns:
    - np.ndarray, the selected ECG pulse if valid.
    """
    total_length = len(ecg_signal)
    pulse_length = int(pulse_length)  # Ensure pulse_length is an integer
    if total_length < pulse_length:
        raise ValueError(f"ECG signal length {total_length} is shorter than required pulse length {pulse_length}.")
    
    # Loop until a valid PQRST pulse is found
    max_attempts = 10
    for attempt in range(max_attempts):
        # Select a random starting point for pulse extraction
        start_index = np.random.randint(0, total_length - pulse_length)
        pulse = ecg_signal[int(start_index): int(start_index) + pulse_length]  # Extract the pulse
        
        # Set threshold as the median amplitude of the pulse
        threshold = np.median(pulse)
        
        # Detect peaks in the pulse
        peaks, properties = find_peaks(pulse, distance=0.2 * fs, prominence=0.1)
        
        # Filter peaks that cross the median threshold
        threshold_peaks = [p for p in peaks if pulse[p] > threshold]
        
        if len(threshold_peaks) > 1:
            # Find the most central peak in threshold_peaks based on its index distance from the center
            central_peak_index = np.argmin([abs(p - pulse_length // 2) for p in threshold_peaks])
            central_peak = threshold_peaks[central_peak_index]
            
            # Nullify all other peaks and their vicinity, keeping only the central peak
            vicinity_radius = int(0.1 * fs)  # Nullify within 0.1 second radius around each peak
            for peak in threshold_peaks:
                if peak != central_peak:
                    start_vicinity = max(0, peak - vicinity_radius)
                    end_vicinity = min(len(pulse), peak + vicinity_radius)
                    pulse[start_vicinity:end_vicinity] = 0
            
            # Re-check the validity of the pulse for PQRST formation
            # Identify P, R, and T waves based on prominence of remaining peaks
            final_peaks, _ = find_peaks(pulse, distance=0.2 * fs, prominence=0.1)
            if len(final_peaks) >= 3:
                sorted_peaks = sorted(final_peaks, key=lambda i: pulse[i], reverse=True)[:3]
                sorted_peaks = sorted(sorted_peaks)  # Sort to maintain order P, R, T
                
                # Validate the positions and relative heights of P, R, and T waves
                p_wave, r_wave, t_wave = sorted_peaks
                if pulse[r_wave] > pulse[p_wave] and pulse[r_wave] > pulse[t_wave]:
                    print(f"Valid PQRST pulse found on attempt {attempt + 1}")
                    return pulse

    raise ValueError("Failed to find a valid PQRST pulse within maximum attempts")
