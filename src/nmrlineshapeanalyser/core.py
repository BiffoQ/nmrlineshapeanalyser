import nmrglue as ng
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional, Union
import warnings
import pandas as pd

class NMRProcessor:
    """
        A comprehensive processor for Nuclear Magnetic Resonance (NMR) spectroscopic data.

            This class provides functionality for loading, processing, and analyzing NMR data, specifically:
            - Loading and processing Bruker format NMR data
            - Selecting and normalizing specific spectral regions
            - Fitting multiple peaks using Pseudo-Voigt functions
            - Calculating peak parameters and error estimates
            - Visualizing results with publication-quality plots
            - Exporting analysis results in multiple formats

            Attributes:
                data (np.ndarray): Raw NMR spectral data
                number (str): Nuclear spin quantum number
                nucleus (str): Nuclear species (e.g., 'H', 'C', 'N')
                uc (ng.unit_conversion): Unit conversion object for frequency/ppm
                ppm (np.ndarray): Chemical shift scale in PPM
                ppm_limits (tuple): Range of chemical shifts in dataset
                fixed_params (list): Parameters fixed during fitting
                carrier_freq (float): Carrier frequency in MHz
        """
    
    def __init__(self):
        """
        Initialize the NMR processor with default settings.

        Sets up default plot styling and initializes class attributes to None.
        Plot styling is configured for publication-quality figures.
        
        """
        self.data = None
        self.number = None
        self.nucleus = None
        self.uc = None
        self.ppm = None
        self.ppm_limits = None
        self.fixed_params = None
        self.carrier_freq = None
        self.set_plot_style()

    @staticmethod
    def set_plot_style() -> None:
        """
            Configure matplotlib plotting parameters for consistent, publication-quality figures.

            Sets font family, sizes, tick parameters, and line widths to create
            professional-looking plots suitable for publication.
        """
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
        Load and process Bruker format NMR data.

        Args:
            filepath (str): Path to the Bruker data directory containing processed data

        Raises:
            FileNotFoundError: If the specified filepath does not exist
            ValueError: If the data cannot be properly loaded or processed

        Note:
            This method extracts key spectral parameters including:
            - Nuclear species and spin quantum number
            - Carrier frequency in MHz
            - Chemical shift scale in PPM
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

    def select_region(self, ppm_start: float, ppm_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a specific region of the NMR spectrum for analysis.

        Args:
            ppm_start (float): Starting chemical shift value in PPM
            ppm_end (float): Ending chemical shift value in PPM

        Returns:
            tuple: Containing:
                - np.ndarray: X-axis data (chemical shift in PPM)
                - np.ndarray: Y-axis data (signal intensity)

        Raises:
            ValueError: If no data is loaded or selected region is outside data range

        Note:
            The selected region is inclusive of both start and end points
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if ppm_start > np.max(self.ppm) or ppm_end < np.min(self.ppm):
            raise ValueError(f"Selected region ({ppm_start}, {ppm_end}) is outside "
                        f"data range ({np.min(self.ppm)}, {np.max(self.ppm)})")
            
        region_mask = (self.ppm >= ppm_start) & (self.ppm <= ppm_end)
        x_region = self.ppm[region_mask]
        y_real = self.data.real
        y_region = y_real[region_mask]
        
        return x_region, y_region

    def normalize_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
       Normalize spectral data for processing.

        Performs baseline correction and amplitude normalization:
        1. Subtracts minimum value (baseline correction)
        2. Divides by maximum value (amplitude normalization)

        Args:
            x_data (np.ndarray): X-axis data (chemical shift)
            y_data (np.ndarray): Y-axis data (signal intensity)

        Returns:
            tuple: Containing:
                - np.ndarray: Original x-axis data
                - np.ndarray: Normalized y-axis data

        Note:
            X-axis data is returned unchanged; only y-axis data is normalized
        """
        # Convert to float type to avoid integer division issues
        y_data = y_data.astype(float)
        y_ground = np.min(y_data)
        y_normalized = y_data - y_ground
        y_amp = np.max(y_normalized)
        
        
        # Handle the case where all values are the same (y_amp would be 0)
        if y_amp != 0:
            y_normalized /= y_amp
        
        return x_data, y_normalized

    @staticmethod
    def pseudo_voigt(x: np.ndarray, x0: float, amp: float, width: float, eta: float) -> np.ndarray:
        
        """
        Calculate the Pseudo-Voigt function, a linear combination of Gaussian and Lorentzian profiles.

        The Pseudo-Voigt function is defined as:
        
        η * L(x) + (1-η) * G(x)
        
        where L(x) is the Lorentzian component and G(x) is the Gaussian component

        Args:
            x (np.ndarray): X-axis values
            x0 (float): Peak center position
            amp (float): Peak amplitude
            width (float): Peak width (FWHM)
            eta (float): Mixing parameter (0 for pure Gaussian, 1 for pure Lorentzian)

        Returns:
            np.ndarray: Calculated Pseudo-Voigt values

        Note:
            FWHM (Full Width at Half Maximum) is the same for both Gaussian and 
            Lorentzian components
            
                   sigma = width / (2 * np.sqrt(2 * np.log(2)))
                   gamma = width / 2
                   lorentzian = amp * (gamma**2 / ((x - x0)**2 + gamma**2))
                   gaussian = amp * np.exp(-0.5 * ((x - x0) / sigma)**2)
        """
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        gamma = width / 2
        lorentzian = amp * (gamma**2 / ((x - x0)**2 + gamma**2))
        gaussian = amp * np.exp(-0.5 * ((x - x0) / sigma)**2)
        return eta * lorentzian + (1 - eta) * gaussian

    def pseudo_voigt_multiple(self, x: np.ndarray, *params) -> np.ndarray:
        """
        Calculate multiple Pseudo-Voigt peaks.
        
        Args:
            x (np.ndarray): X-axis values
            *params: Variable number of peak parameters
            
        Returns:
            np.ndarray: Sum of all Pseudo-Voigt peaks
        """
        n_peaks = len(self.fixed_params)
        param_idx = 0
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            if self.fixed_params[i][0] is not None:
                x0 = self.fixed_params[i][0]
                amp, width, eta, offset = params[param_idx:param_idx + 4]
                param_idx += 4
            else:
                x0, amp, width, eta, offset = params[param_idx:param_idx + 5]
                param_idx += 5
            
            y += self.pseudo_voigt(x, x0, amp, width, eta) + offset
        
        return y

    def fit_peaks(self, x_data: np.ndarray, y_data: np.ndarray, 
                 initial_params: List[float], fixed_x0: Optional[List[bool]] = None) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Fit multiple Pseudo-Voigt peaks to the spectral data.

        Performs non-linear least squares fitting using scipy.optimize.curve_fit.
        Supports fixing peak positions and provides error estimates from the
        covariance matrix.

        Args:
            x_data (np.ndarray): X-axis data
            y_data (np.ndarray): Y-axis data
            initial_params (List[float]): Initial parameters for all peaks
            fixed_x0 (Optional[List[bool]]): Which peak positions to fix during fitting

        Returns:
            tuple: Containing:
                - np.ndarray: Optimized parameters
                - List[Dict]: Peak metrics including errors
                - np.ndarray: Fitted data

        Raises:
            ValueError: If number of parameters is incorrect

        Note:
            For each peak, parameters must be provided in order:
            [x0, amplitude, width, eta, offset]
        """
        # Input validation
        if len(initial_params) % 5 != 0:
            raise ValueError("Number of initial parameters must be divisible by 5")
        
        if fixed_x0 is None:
            fixed_x0 = [False] * (len(initial_params) // 5)
            
        # Setup for fitting
        n_peaks = len(initial_params) // 5
        self.fixed_params = []
        fit_params = []
        lower_bounds = []
        upper_bounds = []
        
        # Process each peak's parameters
        for i in range(n_peaks):
            x0, amp, width, eta, offset = initial_params[5*i:5*(i+1)]
            
            if fixed_x0[i]:
                self.fixed_params.append((x0, None, None, None, None))
                fit_params.extend([amp, width, eta, offset])
                lower_bounds.extend([0, 1, 0, -np.inf])
                upper_bounds.extend([np.inf, np.inf, 1, np.inf])
            else:
                self.fixed_params.append((None, None, None, None, None))
                fit_params.extend([x0, amp, width, eta, offset])
                lower_bounds.extend([x0 - width/2, 0, 1, 0, -np.inf])
                upper_bounds.extend([x0 + width/2, np.inf, np.inf, 1, np.inf])
        
        # Perform the fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            popt, pcov = curve_fit(self.pseudo_voigt_multiple, x_data, y_data,
                                 p0=fit_params, bounds=(lower_bounds, upper_bounds),
                                 maxfev=10000, method='trf')
        
        # Process results
        full_popt = self._process_fit_results(popt, initial_params, fixed_x0)
        peak_metrics = self.calculate_peak_metrics(full_popt, pcov, fixed_x0)
        fitted_data = self.pseudo_voigt_multiple(x_data, *popt)
        
        return full_popt, peak_metrics, fitted_data

    def _process_fit_results(self, popt: np.ndarray, initial_params: List[float], 
                           fixed_x0: List[bool]) -> np.ndarray:
        """
        Process and organize fitting results into a complete parameter set.

        Args:
            popt (np.ndarray): Optimized parameters from curve_fit
            initial_params (List[float]): Initial parameters including fixed values
            fixed_x0 (List[bool]): Which peak positions were fixed

        Returns:
            np.ndarray: Complete set of optimized parameters
        """
        full_popt = []
        param_idx = 0
        n_peaks = len(initial_params) // 5
        
        for i in range(n_peaks):
            if fixed_x0[i]:
                x0 = initial_params[5*i]
                amp, width, eta, offset = popt[param_idx:param_idx + 4]
                param_idx += 4
            else:
                x0, amp, width, eta, offset = popt[param_idx:param_idx + 5]
                param_idx += 5
            full_popt.extend([x0, amp, width, eta, offset])
        
        return np.array(full_popt)

    def calculate_peak_metrics(self, popt: np.ndarray, pcov: np.ndarray, 
                             fixed_x0: List[bool]) -> List[Dict]:
        
        """
        
        Calculate comprehensive metrics for each fitted peak.

        Computes peak parameters and their uncertainties, including:
        - Peak position, amplitude, width
        - Gaussian and Lorentzian areas
        - Total peak area
        - Error estimates for all parameters

        Args:
            popt (np.ndarray): Optimized parameters
            pcov (np.ndarray): Covariance matrix from fitting
            fixed_x0 (List[bool]): Which peak positions were fixed

        Returns:
            List[Dict]: Metrics for each peak including error estimates

        Note:
            Error estimates are calculated using error propagation from the
            covariance matrix

        """
        n_peaks = len(popt) // 5
        peak_results = []
        errors = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros_like(popt)
        error_idx = 0
        
        for i in range(n_peaks):
            # Extract parameters for current peak
            x0, amp, width, eta, offset = popt[5*i:5*(i+1)]
            
            # Calculate errors based on whether x0 was fixed
            if fixed_x0[i]:
                x0_err = 0
                amp_err, width_err, eta_err, offset_err = errors[error_idx:error_idx + 4]
                error_idx += 4
            else:
                x0_err, amp_err, width_err, eta_err, offset_err = errors[error_idx:error_idx + 5]
                error_idx += 5
            
            # Calculate areas and their errors
            sigma = width / (2 * np.sqrt(2 * np.log(2)))
            gamma = width / 2
            
            gauss_area = (1 - eta) * amp * sigma * np.sqrt(2 * np.pi)
            lorentz_area = eta * amp * np.pi * gamma
            total_area = gauss_area + lorentz_area
            
            # Calculate error propagation
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
            
            # Store results
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
        """
            Create publication-quality visualization of fitting results.

                    Generates a plot showing:
                    - Original data points
                    - Overall fitted curve
                    - Individual peak components
                    - Residuals between data and fit
                    - Peak positions marked

                    Args:
                        x_data (np.ndarray): X-axis data
                        y_data (np.ndarray): Y-axis data
                        fitted_data (np.ndarray): Fitted curve data
                        popt (np.ndarray): Optimized parameters

                    Returns:
                        tuple: Containing:
                            - plt.Figure: Figure object
                            - plt.Axes: Axes object
                            - List[np.ndarray]: Individual peak components

                    Note:
            Plot styling is controlled by set_plot_style() method
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot normalized data
        ax1.plot(x_data, y_data, 'ok', ms=1, label='Data')
        ax1.plot(x_data, fitted_data, '-r', lw=2, label='Fit')
        residuals = y_data - fitted_data
        ax1.plot(x_data, residuals-0.05, '-g', lw=2, label='Residuals', alpha=0.5)
        
        # Plot components
        n_peaks = len(popt) // 5
        components = []
        
        for i in range(n_peaks):
            x0, amp, width, eta, offset = popt[5*i:5*(i+1)]
            component = self.pseudo_voigt(x_data, x0, amp, width, eta)
            components.append(component)
            
            ax1.fill(x_data, component, alpha=0.5, label=f'Component {i+1}')
            ax1.plot(x0, self.pseudo_voigt(np.array([x0]), x0, amp, width, eta), 
                    'ob', label='Peak Position' if i == 0 else None)
        
        ax1.invert_xaxis()
        ax1.legend(ncol=2, fontsize=10)
        ax1.set_title('Normalized Scale')
        ax1.set_xlabel(f'$^{{{self.number}}} \\ {self.nucleus}$ chemical shift  (ppm)')
        ax1.hlines(0, x_data[0], x_data[-1], colors='blue', linestyles='dashed', alpha=0.5)
        
        plt.tight_layout()
        
        return fig, ax1, components

    def _print_detailed_results(self, peak_metrics: List[Dict]) -> None:
        """
         Print detailed analysis results to console.

        Displays comprehensive information for each peak:
        - Peak position and error
        - Width in both PPM and Hz
        - Gaussian and Lorentzian contributions
        - Total areas and percentages

        Args:
            peak_metrics (List[Dict]): List of dictionaries containing peak metrics
        """
        print("\nPeak Fitting Results:")
        print("===================")
        
        area_of_peaks = []
        for i, metrics in enumerate(peak_metrics, 1):
            print(f"\nPeak {i} (Position: {metrics['x0'][0]:.2f} ± {metrics['x0'][1]:.2f}):")
            print(f"Amplitude: {metrics['amplitude'][0]:.3f} ± {metrics['amplitude'][1]:.3f}")
            print(f"Width: {metrics['width'][0]:.2f} ± {metrics['width'][1]:.2f} in ppm")
            print(f"Width: {metrics['width'][0]*self.carrier_freq:.2f} ± {metrics['width'][1]*self.carrier_freq:.2f} in Hz")
            print(f"Eta: {metrics['eta'][0]:.2f} ± {metrics['eta'][1]:.2f}")
            print(f"Offset: {metrics['offset'][0]:.3f} ± {metrics['offset'][1]:.3f}")
            print(f"Gaussian Area: {metrics['gaussian_area'][0]:.2f} ± {metrics['gaussian_area'][1]:.2f}")
            print(f"Lorentzian Area: {metrics['lorentzian_area'][0]:.2f} ± {metrics['lorentzian_area'][1]:.2f}")
            print(f"Total Area: {metrics['total_area'][0]:.2f} ± {metrics['total_area'][1]:.2f}")
            print("-" * 50)
            area_of_peaks.append(metrics['total_area'])

        self._calculate_and_print_percentages(area_of_peaks)

    def _calculate_and_print_percentages(self, area_of_peaks: List[Tuple[float, float]]) -> None:
        """
         Calculate and display percentage contributions of each peak.

        Computes relative contributions of each peak to total spectral area
        and their associated uncertainties using error propagation.

        Args:
            area_of_peaks (List[Tuple[float, float]]): List of (area, error) pairs
        """
        total_area_sum = sum(area[0] for area in area_of_peaks)
        total_area_sum_err = np.sqrt(sum(area[1]**2 for area in area_of_peaks))
        
        overall_percentage = []
        for i, (area, area_err) in enumerate(area_of_peaks, 1):
            percentage = (area / total_area_sum) * 100
            percentage_err = percentage * np.sqrt((area_err / area) ** 2 + 
                                               (total_area_sum_err / total_area_sum) ** 2)
            print(f'Peak {i} Percentage is {percentage:.2f}% ± {percentage_err:.2f}%')
            overall_percentage.append((percentage, percentage_err))

        overall_percentage_sum = sum(p[0] for p in overall_percentage)
        overall_percentage_sum_err = np.sqrt(sum(p[1]**2 for p in overall_percentage))
        print(f'Overall Percentage is {overall_percentage_sum:.2f}% ± {overall_percentage_sum_err:.2f}%')

    def save_results(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray,
                    fitted_data: np.ndarray, peak_metrics: List[Dict],
                    popt: np.ndarray, components: List[np.ndarray]) -> None:
        """
            Save all analysis results to files.

                    Creates three output files:
                    1. CSV file with peak data and components
                    2. Text file with detailed metrics
                    3. PNG file with visualization plot

                    Args:
                        filepath (str): Base path for saving files
                        x_data (np.ndarray): X-axis data
                        y_data (np.ndarray): Y-axis data
                        fitted_data (np.ndarray): Fitted curve data
                        peak_metrics (List[Dict]): Peak metrics including errors
                        popt (np.ndarray): Optimized parameters
                        components (List[np.ndarray]): Individual peak components

                    Note:
                        Files are named automatically based on the base filepath:
                        - peak_data.csv
                        - pseudoVoigtPeak_metrics.txt
                        - pseudoVoigtPeakFit.png
        """
        self._save_peak_data(filepath, x_data, y_data, fitted_data, components)
        self._save_metrics(filepath, peak_metrics)
        self._save_plot(filepath, x_data, y_data, fitted_data,
                       popt)
        self._print_detailed_results(peak_metrics)

    def _save_peak_data(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray, 
                       fitted_data: np.ndarray, components: List[np.ndarray]) -> None:
        """
         Save peak data to CSV file.

        Creates a CSV file containing:
        - X-axis data
        - Original Y-axis data
        - Fitted curve
        - Individual peak components

        Args:
            filepath (str): Base path for saving file
            x_data (np.ndarray): X-axis data
            y_data (np.ndarray): Y-axis data
            fitted_data (np.ndarray): Fitted curve data
            components (List[np.ndarray]): Individual peak components
            """
        df = pd.DataFrame({'x_data': x_data, 'y_data': y_data, 'y_fit': fitted_data})
        
        for i, component in enumerate(components):
            df[f'component_{i+1}'] = component
        
        df.to_csv(filepath + 'peak_data.csv', index=False)

    def _save_metrics(self, filepath: str, peak_metrics: List[Dict]) -> None:
        """
        Save detailed peak metrics to text file.

        Writes comprehensive analysis results including:
        - Peak positions and widths
        - Areas (Gaussian and Lorentzian contributions)
        - Relative percentages
        - All associated uncertainties

        Args:
            filepath (str): Base path for saving file
            peak_metrics (List[Dict]): Peak metrics including errors
            """
        with open(filepath + 'pseudoVoigtPeak_metrics.txt', 'w') as file:
            area_of_peaks = []
            for i, metrics in enumerate(peak_metrics, 1):
                file.write(f"\nPeak {i} (Position: {metrics['x0'][0]:.2f} ± {metrics['x0'][1]:.2f}):\n")
                file.write(f"Amplitude: {metrics['amplitude'][0]:.3f} ± {metrics['amplitude'][1]:.3f}\n")
                file.write(f"Width: {metrics['width'][0]:.2f} ± {metrics['width'][1]:.2f} in ppm\n")
                file.write(f"Width: {metrics['width'][0]*self.carrier_freq:.2f} ± {metrics['width'][1]*self.carrier_freq:.2f} in Hz\n")
                file.write(f"Eta: {metrics['eta'][0]:.2f} ± {metrics['eta'][1]:.2f}\n")
                file.write(f"Offset: {metrics['offset'][0]:.3f} ± {metrics['offset'][1]:.3f}\n")
                file.write(f"Gaussian Area: {metrics['gaussian_area'][0]:.2f} ± {metrics['gaussian_area'][1]:.2f}\n")
                file.write(f"Lorentzian Area: {metrics['lorentzian_area'][0]:.2f} ± {metrics['lorentzian_area'][1]:.2f}\n")
                file.write(f"Total Area: {metrics['total_area'][0]:.2f} ± {metrics['total_area'][1]:.2f}\n")
                file.write("\n" + "-" * 50 + "\n")
                area_of_peaks.append(metrics['total_area'])
            
            # Write percentages
            total_area_sum = sum(area[0] for area in area_of_peaks)
            total_area_sum_err = np.sqrt(sum(area[1]**2 for area in area_of_peaks))
            
            for i, (area, area_err) in enumerate(area_of_peaks, 1):
                percentage = (area / total_area_sum) * 100
                percentage_err = percentage * np.sqrt((area_err / area) ** 2 + 
                                                   (total_area_sum_err / total_area_sum) ** 2)
                file.write(f'Peak {i} Percentage is {percentage:.2f}% ± {percentage_err:.2f}%\n')
            
            overall_percentage = sum((area[0] / total_area_sum) * 100 for area in area_of_peaks)
            file.write(f'Overall Percentage is {overall_percentage:.2f}%\n')

    def _save_plot(self, filepath: str, x_data: np.ndarray, y_data: np.ndarray,
                   fitted_data: np.ndarray,
                   popt: np.ndarray) -> None:
        
        """        
        Save visualization plot to PNG file.

        Creates a publication-quality figure showing:
        - Original data
        - Fitted curves
        - Individual components
        - Residuals

        Args:
            filepath (str): Base path for saving file
            x_data (np.ndarray): X-axis data (chemical shift values)
            y_data (np.ndarray): Y-axis data (original spectral intensities)
            fitted_data (np.ndarray): The overall fitted curve
            popt (np.ndarray): Optimized parameters for all peaks

        Note:
            - The plot is saved as 'pseudoVoigtPeakFit.png'
            - Plot styling is controlled by set_plot_style() method
            - The figure is automatically closed after saving to free memory
            - Figure is saved with tight layout to prevent label clipping"""
            
        fig, _, _ = self.plot_results(x_data, y_data, fitted_data, 
                                    popt)
        fig.savefig(filepath + 'pseudoVoigtPeakFit.png', bbox_inches='tight')
        plt.close(fig)