Here's a detailed, stylized `README.md` file for your ECG-TEM reconstruction project:

---

# ECG-TEM Reconstruction

A Python-based implementation of ECG signal reconstruction using Time Encoding Machine (TEM) with Integrate-and-Fire (IAF) encoding. This project is designed for high-quality reconstruction of noisy ECG signals using robust sampling, denoising, and reconstruction techniques. The reconstructed results are visualized and saved in a `.mat` format, facilitating easy integration with other MATLAB and Python-based biomedical analysis workflows.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Visualization](#visualization)
- [Output](#output)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **ECG-TEM Reconstruction** project uses Integrate-and-Fire Time Encoding Machine (IAF-TEM) to reconstruct ECG signals, a method known for efficient and noise-resistant signal recovery. It incorporates signal preprocessing, noise addition, Fourier series analysis, and an advanced denoising process using the Cadzow algorithm to achieve high-quality signal restoration.

### Why ECG-TEM?

- **Signal Recovery:** Ideal for reconstructing complex biomedical signals like ECG, where traditional sampling can be prone to noise.
- **Efficient Sampling:** Reduces data load by using sparse sampling and advanced mathematical transformations.
- **Robust to Noise:** Utilizes Gaussian noise injection and denoising algorithms for a resilient reconstruction pipeline.

---

## Features

- **IAF-TEM Signal Encoding**: Captures significant features of ECG signals using time encoding.
- **Cadzow Denoising**: An advanced algorithm for robust noise reduction and accurate reconstruction.
- **User-defined Parameters**: Configure sampling parameters such as sampling rate, bias, noise level, and reconstruction accuracy.
- **Visualization and Export**: High-quality plots and `.mat` output files for easy integration with MATLAB and other signal analysis software.

---

## File Structure

```
ECG-TEM-Reconstruction/
│
├── main.py                     # Main script to execute the reconstruction
├── README.md                   # Project documentation
├── requirements.txt            # Required Python packages
│
├── data/
│   └── u.mat                   # Input data file (Example ECG signal)
│
├── output/
│   └── reconstructed_data.mat  # Output folder to save reconstructed data
│
├── ecg_tem/
│   ├── __init__.py             # Package initialization
│   ├── ecg_loader.py           # ECG loading and preprocessing functions
│   ├── tem_reconstructor.py    # TEM reconstruction functions and core algorithm
│   ├── noise.py                # Noise injection functions
│   ├── cadzow_denoising.py     # Cadzow denoising algorithm
│   └── plotter.py              # Visualization and plotting utilities
```

---

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ECG-TEM-Reconstruction.git
   cd ECG-TEM-Reconstruction
   ```

2. **Install Dependencies**:
   Install the required Python packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   - Ensure you have an ECG signal file (e.g., `u.mat`) in the `data/` directory.
   - Modify the `main.py` parameters as needed to suit your data specifications.

---

## Usage

To run the ECG-TEM reconstruction with default parameters, execute the `main.py` script:

```bash
python main.py
```

### Customizing Parameters
You can modify the reconstruction parameters directly in `main.py`:

```python
K = 7               # Number of Fourier series components
T = 1.0             # Time period
kappa = 0.018       # TEM integrator acceleration factor
delta = 0.9         # TEM threshold
bias = 1.0          # Integrator bias
noise_level = 0.1   # Gaussian noise level
```

Adjust these parameters based on your specific signal characteristics or desired reconstruction quality.

---

## Parameters

The following parameters influence the ECG reconstruction:

- **K (Fourier Components)**: Number of Fourier series components for spectral reconstruction. Higher values offer finer resolution.
- **T (Time Period)**: Time duration for one cycle of reconstruction.
- **kappa (Integrator Acceleration)**: Controls integrator speed; higher values accelerate reconstruction.
- **delta (Threshold)**: Determines the threshold for the IAF encoder.
- **bias**: Controls the baseline offset for encoding.
- **noise_level**: Defines the amount of Gaussian noise injected into the signal for testing noise robustness.

---

## Visualization

The reconstructed signal is plotted against the noisy input signal, with sample points highlighted to visualize the sampling accuracy.

**Example Plot:**

![ECG-TEM Reconstruction Example](./output/reconstruction_plot.png)

- **Original Noisy Signal**: Shown in blue.
- **Reconstructed Signal**: Shown in orange with high accuracy in approximating the original ECG signal.
- **Sample Points**: Marked with red crosses, representing points used in the reconstruction process.

---

## Output

The reconstruction data is saved in the `output/` directory in `.mat` format:

- **`original_signal`**: The noisy input signal array.
- **`reconstructed_signal`**: The denoised and reconstructed ECG signal.
- **`sample_times`**: The sample points used in the reconstruction.

### File Location
Output files are saved by default in `output/reconstructed_data.mat`.

---

## Example

Below is an example of how to run the reconstruction:

```python
# main.py

import numpy as np
from ecg_tem.ecg_loader import load_ecg_signal
from ecg_tem.tem_reconstructor import TEMReconstructor
from ecg_tem.noise import add_gaussian_noise
from ecg_tem.plotter import plot_reconstruction

def main():
    # Define reconstruction parameters
    K = 7
    T = 1.0
    kappa = 0.018
    delta = 0.9
    bias = 1.0
    noise_level = 0.1

    # Load ECG signal and add noise
    signal = load_ecg_signal('data/u.mat', 'signal2')
    noisy_signal = add_gaussian_noise(signal, noise_level)
    
    # Perform TEM reconstruction
    reconstructor = TEMReconstructor(K, T, kappa, delta, bias)
    tn, signal_estimated, spectrum_estimated = reconstructor.reconstruct(noisy_signal)

    # Plot and save results
    plot_reconstruction(noisy_signal, signal_estimated, tn, output_path="output/reconstructed_data.mat")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python main.py
```

---

## Contributing

Contributions to the ECG-TEM Reconstruction project are welcome! If you would like to add improvements or features, please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 

For any questions or issues, please contact [your-email@example.com](mailto:your-email@example.com) or open an issue in this repository.

Happy reconstructing! 🚀