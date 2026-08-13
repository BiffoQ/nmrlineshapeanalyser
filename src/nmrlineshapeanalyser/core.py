import nmrglue as ng
import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional, Union
import warnings
import pandas as pd
import sys
import glob
import os

print(f"nmrglue: {ng.__version__}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"matplotlib: {mpl.__version__}")
print(f"pandas: {pd.__version__}")

class NMRProcessor:
    """
    A comprehensive class for processing and analyzing NMR data.
    Combines data loading, processing, peak fitting, and visualization.
    """
    
    def __init__(self):
        """Initialize the NMR processor with default plot style."""
        self.data = None
        self.number = None
        self.nucleus = None
        self.carrier_freq = None
        self.uc = None
        self.ppm = None
        self.ppm_limits = None
        self.fixed_params = None
        self.set_plot_style()

    @staticmethod
    def set_plot_style() -> None:
        """Set up the matplotlib plotting style."""
        mpl.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2
        mpl.rcParams['xtick.major.size'] = mpl.rcParams['ytick.major.size'] = 8
        mpl.rcParams['xtick.major.width'] = mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['xtick.direction'] = mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['xtick.major.top'] = mpl.rcParams['ytick.major.right'] = False
        mpl.rcParams['xtick.minor.size'] = mpl.rcParams['ytick.minor.size'] = 5
        mpl.rcParams['xtick.minor.width'] = mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['xtick.top'] = mpl.rcParams['ytick.right'] = True

    def load_data(self, filepath: str) -> None:
        """
        Load and process Bruker NMR data from the specified filepath.
        
        Args:
            filepath (str): Path to the Bruker data directory
        """
        # Read the Bruker data
        dic, self.data = ng.bruker.read_pdata(filepath)
        
        # Set the spectral parameters
        udic = ng.bruker.guess_udic(dic, self.data)
        
        nuclei = udic[0]['label']
        carrier_freq = udic[0]['obs']
        self.carrier_freq = carrier_freq
        
        # Extract number and nucleus symbols
        self.number = ''.join(filter(str.isdigit, nuclei))
        self.nucleus = ''.join(filter(str.isalpha, nuclei))
        
        # Create converter and get scales
        self.uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        self.ppm = self.uc.ppm_scale()
        self.ppm_limits = self.uc.ppm_limits()

    def load_csv(self, filepath: str, atomic_no: str, nucleus: str, larmor_freq: float) -> None:
        """
        Load CSV data exported from MNova (columns: 'ppm', 'Intensity').
        
        Args:
            filepath (str): Directory path containing the CSV file
            atomic_no (str): Atomic number of the nucleus
            nucleus (str): Nuclear symbol
            larmor_freq (float): Larmor frequency in MHz
        
        Raises:
            FileNotFoundError: If no CSV files are found in the specified directory
        """
        csv_files = glob.glob(os.path.join(filepath, '*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {filepath}")
        
        data = pd.read_csv(csv_files[0], sep='[,\t]', engine='python')
        
        required_columns = ['ppm', 'Intensity']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        x_data = data['ppm'].values
        y_data = data['Intensity'].values
        
        # Create pseudo-complex data to match the Bruker data format
        y_data = y_data + 1j * np.zeros_like(y_data)
        
        self.ppm = x_data
        self.data = y_data
        self.number = str(atomic_no)
        self.nucleus = nucleus
        self.carrier_freq = float(larmor_freq)

    def select_region(self, ppm_start: float, ppm_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """Select a specific region of the NMR spectrum for analysis."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        region_mask = (self.ppm >= ppm_start) & (self.ppm <= ppm_end)
        x_region = self.ppm[region_mask]
        y_real = self.data.real
        y_region = y_real[region_mask]

        if x_region.size == 0:
            raise ValueError(f"No data found in region {ppm_start} to {ppm_end} ppm.")

        return x_region, y_region

    def normalize_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Normalize the data for processing to 0-1 range."""
        y_ground = np.min(y_data)
        y_normalized = y_data - y_ground
        y_amp = np.max(y_normalized)
        y_normalized = y_normalized / y_amp if y_amp != 0 else np.zeros_like(y_normalized, dtype=float)
        return x_data, y_normalized, y_amp, y_ground

    @staticmethod
    def pseudo_voigt(x: np.ndarray, x0: float, amp: float, width: float, eta: float) -> np.ndarray:
        """Calculate the Pseudo-Voigt function."""
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        gamma = width / 2
        lorentzian = amp * (gamma**2 / ((x - x0)**2 + gamma**2))
        gaussian = amp * np.exp(-0.5 * ((x - x0) / sigma)**2)
        return eta * lorentzian + (1 - eta) * gaussian

    def pseudo_voigt_multiple(self, x: np.ndarray, *params) -> np.ndarray:
        """
        Calculate multiple Pseudo-Voigt peaks summed on top of a single shared offset.
        The shared offset is always the last entry in params.
        """
        n_peaks = len(self.fixed_params)

        expected_len = sum(
            (fixed_x0 is None) + (fixed_amp is None) + (fixed_width is None) + (fixed_eta is None)
            for fixed_x0, fixed_amp, fixed_width, fixed_eta, _ in self.fixed_params
        ) + 1  # +1 for the shared offset
        if len(params) != expected_len:
            raise ValueError(
                f"Expected {expected_len} parameters for {n_peaks} peak(s), got {len(params)}"
            )

        param_idx = 0
        y = np.zeros_like(x)

        for i in range(n_peaks):
            fixed_x0, fixed_amp, fixed_width, fixed_eta, _ = self.fixed_params[i]

            if fixed_x0 is not None:
                x0 = fixed_x0
            else:
                x0 = params[param_idx]
                param_idx += 1

            if fixed_amp is not None:
                amp = fixed_amp
            else:
                amp = params[param_idx]
                param_idx += 1

            if fixed_width is not None:
                width = fixed_width
            else:
                width = params[param_idx]
                param_idx += 1

            if fixed_eta is not None:
                eta = fixed_eta
            else:
                eta = params[param_idx]
                param_idx += 1

            y += self.pseudo_voigt(x, x0, amp, width, eta)
        
        # Single shared offset applied once to the summed peaks
        offset = params[param_idx]
        y += offset
        
        return y

    def fit_peaks(self, x_data: np.ndarray, y_data: np.ndarray,
                 initial_params: List[float], fixed_x0: Optional[List[bool]] = None,
                 fixed_amp: Optional[List[bool]] = None, fixed_width: Optional[List[bool]] = None,
                 fixed_eta: Optional[List[bool]] = None,
                 y_scale: float = 1.0, y_offset: float = 0.0) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Fit multiple Pseudo-Voigt peaks to the data, sharing a single baseline offset.

        Args:
            fixed_x0: per-peak flags to fix the peak position at its initial value
            fixed_amp: per-peak flags to fix the peak amplitude at its initial value
            fixed_width: per-peak flags to fix the peak width at its initial value
            fixed_eta: per-peak flags to fix the Gaussian/Lorentzian mixing parameter
                (eta) at its initial value. Defaults to all False, i.e. eta is fitted
                freely for every peak unless explicitly fixed.

        Note: all peaks share one fitted offset (baseline), seeded from the mean of
        the per-peak offsets given in initial_params. The returned popt/peak_metrics
        report that same shared offset value for every peak.

        IMPORTANT: Initial offsets should be in normalized scale (0-1), e.g., use 0.0.
        """
        if len(initial_params) % 5 != 0:
            raise ValueError("Number of initial parameters must be divisible by 5")

        n_peaks = len(initial_params) // 5

        if fixed_x0 is None:
            fixed_x0 = [False] * n_peaks
        if fixed_amp is None:
            fixed_amp = [False] * n_peaks
        if fixed_width is None:
            fixed_width = [False] * n_peaks
        if fixed_eta is None:
            fixed_eta = [False] * n_peaks

        self.fixed_params = []
        fit_params = []
        lower_bounds = []
        upper_bounds = []
        
        # Process each peak's shape parameters (x0, amp, width, eta).
        # The offset is handled separately below as a single shared parameter.
        for i in range(n_peaks):
            x0, amp, width, eta, offset = initial_params[5*i:5*(i+1)]
            
            self.fixed_params.append((
                x0 if fixed_x0[i] else None,
                amp if fixed_amp[i] else None,
                width if fixed_width[i] else None,
                eta if fixed_eta[i] else None,
                None,
            ))

            if not fixed_x0[i]:
                fit_params.append(x0)
                lower_bounds.append(x0 - width/2)
                upper_bounds.append(x0 + width/2)

            if not fixed_amp[i]:
                fit_params.append(amp)
                lower_bounds.append(0)
                upper_bounds.append(np.inf)

            if not fixed_width[i]:
                fit_params.append(width)
                lower_bounds.append(1)
                upper_bounds.append(np.inf)

            if not fixed_eta[i]:
                fit_params.append(eta)
                lower_bounds.append(0)
                upper_bounds.append(1)
        
        # Single shared offset, appended once at the end, seeded from the
        # mean of the per-peak offsets in initial_params
        shared_offset_init = float(np.mean(initial_params[4::5]))
        fit_params.append(shared_offset_init)
        lower_bounds.append(-1.0)
        upper_bounds.append(2.0)
        
        # Perform the fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            popt, pcov = curve_fit(self.pseudo_voigt_multiple, x_data, y_data,
                                 p0=fit_params, bounds=(lower_bounds, upper_bounds),
                                 maxfev=10000, method='trf')
        
        # Process results
        full_popt = self._process_fit_results(popt, initial_params, fixed_x0, fixed_amp, fixed_width, fixed_eta)
        peak_metrics = self.calculate_peak_metrics(full_popt, pcov, fixed_x0, fixed_amp, fixed_width, fixed_eta)
        fitted_data = self.pseudo_voigt_multiple(x_data, *popt)

        return full_popt, peak_metrics, fitted_data

    def _process_fit_results(self, popt: np.ndarray, initial_params: List[float],
                           fixed_x0: List[bool], fixed_amp: List[bool],
                           fixed_width: List[bool], fixed_eta: List[bool]) -> np.ndarray:
        """Process and organize fitting results, broadcasting the shared offset to every peak."""
        n_peaks = len(initial_params) // 5
        param_idx = 0
        peak_shape_params = []

        for i in range(n_peaks):
            x0_init, amp_init, width_init, eta_init, offset_init = initial_params[5*i:5*(i+1)]

            if fixed_x0[i]:
                x0 = x0_init
            else:
                x0 = popt[param_idx]
                param_idx += 1

            if fixed_amp[i]:
                amp = amp_init
            else:
                amp = popt[param_idx]
                param_idx += 1

            if fixed_width[i]:
                width = width_init
            else:
                width = popt[param_idx]
                param_idx += 1

            if fixed_eta[i]:
                eta = eta_init
            else:
                eta = popt[param_idx]
                param_idx += 1

            peak_shape_params.append((x0, amp, width, eta))
        
        # Last remaining parameter is the single shared offset
        shared_offset = popt[param_idx]
        
        full_popt = []
        for x0, amp, width, eta in peak_shape_params:
            full_popt.extend([x0, amp, width, eta, shared_offset])
        
        return np.array(full_popt)

    def calculate_peak_metrics(self, popt: np.ndarray, pcov: np.ndarray,
                             fixed_x0: List[bool], fixed_amp: List[bool],
                             fixed_width: List[bool], fixed_eta: List[bool]) -> List[Dict]:
        """Calculate metrics for each fitted peak, using the shared offset's error for all peaks."""
        n_peaks = len(popt) // 5
        errors = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros(len(popt))
        error_idx = 0

        peak_shape_errors = []
        for i in range(n_peaks):
            if fixed_x0[i]:
                x0_err = 0
            else:
                x0_err = errors[error_idx]
                error_idx += 1

            if fixed_amp[i]:
                amp_err = 0
            else:
                amp_err = errors[error_idx]
                error_idx += 1

            if fixed_width[i]:
                width_err = 0
            else:
                width_err = errors[error_idx]
                error_idx += 1

            if fixed_eta[i]:
                eta_err = 0
            else:
                eta_err = errors[error_idx]
                error_idx += 1

            peak_shape_errors.append((x0_err, amp_err, width_err, eta_err))
        
        # Last remaining error entry belongs to the single shared offset
        offset_err = errors[error_idx] if error_idx < len(errors) else 0
        
        peak_results = []
        for i in range(n_peaks):
            x0, amp, width, eta, offset = popt[5*i:5*(i+1)]
            x0_err, amp_err, width_err, eta_err = peak_shape_errors[i]
            
            sigma = width / (2 * np.sqrt(2 * np.log(2)))
            gamma = width / 2
            
            gauss_area = (1 - eta) * amp * sigma * np.sqrt(2 * np.pi)
            lorentz_area = eta * amp * np.pi * gamma
            total_area = gauss_area + lorentz_area
            
            gauss_area_err = np.sqrt(
                ((1 - eta) * sigma * np.sqrt(2 * np.pi) * amp_err) ** 2 +
                (amp * sigma * np.sqrt(2 * np.pi) * eta_err) ** 2 +
                ((1 - eta) * amp * np.sqrt(2 * np.pi) * (width_err / (2 * np.sqrt(2 * np.log(2))))) ** 2
            )
            
            lorentz_area_err = np.sqrt(
                (eta * np.pi * gamma * amp_err) ** 2 +
                (amp * np.pi * gamma * eta_err) ** 2 +
                (eta * amp * np.pi * (width_err / 2)) ** 2
            )
            
            total_area_err = np.sqrt(gauss_area_err ** 2 + lorentz_area_err ** 2)
            
            peak_results.append({
                'x0': (x0, x0_err),
                'amplitude': (amp, amp_err),
                'width': (width, width_err),
                'eta': (eta, eta_err),
                'offset': (offset, offset_err),
                'gaussian_area': (gauss_area, gauss_area_err),
                'lorentzian_area': (lorentz_area, lorentz_area_err),
                'total_area': (total_area, total_area_err)
            })
        
        return peak_results

    def plot_results(self, x_data: np.ndarray, y_data: np.ndarray,
                    fitted_data: np.ndarray,
                    popt: np.ndarray) -> Tuple[plt.Figure, plt.Axes, List[np.ndarray]]:
        """Plot the fitting results with components (all in normalized 0-1 scale)."""
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        ax1.plot(x_data, y_data, 'ok', ms=1, label='Data')
        ax1.plot(x_data, fitted_data, '-r', lw=2, label='Fit')
        residuals = y_data - fitted_data
        ax1.plot(x_data, residuals-0.05, '-g', lw=2, label='Residuals', alpha=0.5)
        
        n_peaks = len(popt) // 5
        components = []
        
        for i in range(n_peaks):
            x0, amp, width, eta, offset = popt[5*i:5*(i+1)]
            component = self.pseudo_voigt(x_data, x0, amp, width, eta) + offset
            components.append(component)
            
            ax1.fill(x_data, component, alpha=0.5, label=f'Component {i+1}')
            peak_height = self.pseudo_voigt(np.array([x0]), x0, amp, width, eta)[0] + offset
            ax1.plot(x0, peak_height, 'ob', markersize=8, label='Peak Position' if i == 0 else '')
        
        ax1.invert_xaxis()
        ax1.legend(ncol=2, fontsize=10)
        ax1.set_title('NMR Peak Fit (Normalized Scale 0-1)')
        ax1.set_xlabel(f'$^{{{self.number}}} \\ {self.nucleus}$ chemical shift (ppm)')
        ax1.set_ylabel('Intensity (normalized 0-1)')
        ax1.hlines(0, x_data[0], x_data[-1], colors='blue', linestyles='dashed', alpha=0.5)
        ax1.set_ylim(-0.15, 1.15)
        
        return fig, ax1, components

    def _print_detailed_results(self, peak_metrics: List[Dict]) -> None:
        """Print detailed fitting results and statistics."""
        print("\nPeak Fitting Results:")
        print("===================")
        
        area_of_peaks = []
        for i, metrics in enumerate(peak_metrics, 1):
            print(f"\nPeak {i} (Position: {metrics['x0'][0]:.2f} ± {metrics['x0'][1]:.2f}):")
            print(f"  Amplitude: {metrics['amplitude'][0]:.3f} ± {metrics['amplitude'][1]:.3f}")
            print(f"  Width (FWHM): {metrics['width'][0]:.2f} ± {metrics['width'][1]:.2f} ppm")
            print(f"  Width (FWHM): {metrics['width'][0]*self.carrier_freq:.2f} ± {metrics['width'][1]*self.carrier_freq:.2f} Hz")
            print(f"  Carrier Frequency: {self.carrier_freq} MHz")
            print(f"  Eta: {metrics['eta'][0]:.2f} ± {metrics['eta'][1]:.2f}")
            print(f"  Offset: {metrics['offset'][0]:.4f} ± {metrics['offset'][1]:.4f}")
            print(f"  Gaussian Area: {metrics['gaussian_area'][0]:.2f} ± {metrics['gaussian_area'][1]:.2f}")
            print(f"  Lorentzian Area: {metrics['lorentzian_area'][0]:.2f} ± {metrics['lorentzian_area'][1]:.2f}")
            print(f"  Total Area: {metrics['total_area'][0]:.2f} ± {metrics['total_area'][1]:.2f}")
            area_of_peaks.append(metrics['total_area'])

        self._calculate_and_print_percentages(area_of_peaks)

    def _calculate_and_print_percentages(self, area_of_peaks: List[Tuple[float, float]]) -> None:
        """Calculate and print percentage contributions of each peak."""
        total_area_sum = sum(area[0] for area in area_of_peaks)
        total_area_sum_err = np.sqrt(sum(area[1]**2 for area in area_of_peaks))
        
        for i, (area, area_err) in enumerate(area_of_peaks, 1):
            percentage = (area / total_area_sum) * 100
            percentage_err = percentage * np.sqrt((area_err / area) ** 2 + 
                                               (total_area_sum_err / total_area_sum) ** 2)
            print(f'Peak {i} Percentage: {percentage:.2f}% ± {percentage_err:.2f}%')

        overall_percentage = sum((area[0] / total_area_sum) * 100 for area in area_of_peaks)
        print(f'Total: {overall_percentage:.2f}%')

    def save_results(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray,
                    fitted_data: np.ndarray, peak_metrics: List[Dict],
                    popt: np.ndarray, components: List[np.ndarray]) -> None:
        """Save all results to files."""
        self._save_peak_data(filepath, x_data, y_data, fitted_data, components)
        self._save_metrics(filepath, peak_metrics)
        self._save_plot(filepath, x_data, y_data, fitted_data, popt)
        self._print_detailed_results(peak_metrics)

    def _save_peak_data(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray, 
                       fitted_data: np.ndarray, components: List[np.ndarray]) -> None:
        """Save peak data to CSV file."""
        df = pd.DataFrame({'x_data': x_data, 'y_data': y_data, 'y_fit': fitted_data})
        for i, component in enumerate(components):
            df[f'component_{i+1}'] = component
        df.to_csv(filepath + 'peak_data.csv', index=False)

    def _save_metrics(self, filepath: str, peak_metrics: List[Dict]) -> None:
        """Save peak metrics to text file."""
        with open(filepath + 'pseudoVoigtPeak_metrics.txt', 'w') as file:
            area_of_peaks = []
            for i, metrics in enumerate(peak_metrics, 1):
                file.write(f"\nPeak {i} (Position: {metrics['x0'][0]:.2f} ± {metrics['x0'][1]:.2f}):\n")
                file.write(f"Amplitude: {metrics['amplitude'][0]:.3f} ± {metrics['amplitude'][1]:.3f}\n")
                file.write(f"Width (FWHM): {metrics['width'][0]:.2f} ± {metrics['width'][1]:.2f} ppm\n")
                file.write(f"Width (FWHM): {metrics['width'][0]*self.carrier_freq:.2f} ± {metrics['width'][1]*self.carrier_freq:.2f} Hz\n")
                file.write(f"Eta: {metrics['eta'][0]:.2f} ± {metrics['eta'][1]:.2f}\n")
                file.write(f"Offset: {metrics['offset'][0]:.4f} ± {metrics['offset'][1]:.4f}\n")
                file.write(f"Total Area: {metrics['total_area'][0]:.2f} ± {metrics['total_area'][1]:.2f}\n\n")
                area_of_peaks.append(metrics['total_area'])
            
            total_area_sum = sum(area[0] for area in area_of_peaks)
            for i, (area, area_err) in enumerate(area_of_peaks, 1):
                percentage = (area / total_area_sum) * 100
                file.write(f'Peak {i} Percentage: {percentage:.2f}%\n')

    def _save_plot(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray,
                   fitted_data: np.ndarray, popt: np.ndarray) -> None:
        """Save the plot to a file."""
        fig, _, _ = self.plot_results(x_data, y_data, fitted_data, popt)
        fig.savefig(filepath + 'pseudoVoigtPeakFit.png', bbox_inches='tight')
        plt.close(fig)