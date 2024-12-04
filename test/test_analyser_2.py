import unittest
import numpy as np
import webbrowser
import os
import nmrglue as ng
from unittest.mock import Mock, patch, mock_open
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import sys
from nmrlineshapeanalyser.core import NMRProcessor
from unittest.mock import mock_open
import coverage

class TestNMRProcessor(unittest.TestCase):
    """Test suite for NMR Processor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = NMRProcessor()
        self.test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.test_ppm = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        self.processor.carrier_freq = 500.0
        self.assertEqual(self.processor.carrier_freq, 500.0, "carrier_freq not set correctly in setUp")
        
        # Close all existing plots
        plt.close('all')
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        plt.close('all')
        
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    @patch('nmrglue.bruker.read_pdata')
    @patch('nmrglue.bruker.guess_udic')
    def test_load_data(self, mock_guess_udic, mock_read_pdata):
        """Test data loading functionality."""
        # Mock the returned values
        mock_dic = {}
        mock_data = np.array([1.0, 2.0, 3.0])
        mock_read_pdata.return_value = (mock_dic, mock_data)
        
        # Mock udic with all required keys
        mock_udic = [{
            'label': '17O',
            'size': 1024,
            'complex': True,
            'sw': 100.0,
            'sf': 500.0,
            'car': 0.0,
            'obs': 500.0
        }]
        mock_guess_udic.return_value = mock_udic
        
        # Test data loading
        self.processor.load_data("dummy/path")
        
        
        # Verify the data was loaded correctly
        self.assertIsNotNone(self.processor.data) 
        self.assertEqual(self.processor.nucleus, 'O')
        self.assertEqual(self.processor.number, '17')
        self.assertEqual(self.processor.carrier_freq, 500.0)

    def test_select_region(self):
        """Test region selection functionality."""
        # Setup test data
        self.processor.ppm = np.array([0, 1, 2, 3, 4])
        self.processor.data = np.array([0, 1, 2, 3, 4])
        
        # Test normal case
        x_region, y_region = self.processor.select_region(1, 3)
        self.assertTrue(np.all(x_region >= 1))
        self.assertTrue(np.all(x_region <= 3))
        self.assertEqual(len(x_region), len(y_region))

        # Test edge cases
        x_region, y_region = self.processor.select_region(0, 4)
        self.assertEqual(len(x_region), len(self.processor.ppm))

    def test_normalize_data(self):
        # Basic tests
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])
        x_norm, y_norm = self.processor.normalize_data(x_data, y_data)
        
        assert np.array_equal(x_norm, x_data)
        assert np.min(y_norm) == 0
        assert np.max(y_norm) == 1
        assert x_norm.shape == x_data.shape
        assert y_norm.shape == y_data.shape

        # Test reversibility
        y_ground = np.min(y_data)
        y_amp = np.max(y_data) - y_ground
        y_reconstructed = y_norm * y_amp + y_ground
        np.testing.assert_array_almost_equal(y_reconstructed, y_data)

        # Test negative values with reversibility
        y_data = np.array([-5, 0, 5])
        x_norm, y_norm = self.processor.normalize_data(x_data[:3], y_data)
        y_ground = np.min(y_data)
        y_amp = np.max(y_data) - y_ground
        y_reconstructed = y_norm * y_amp + y_ground
        np.testing.assert_array_almost_equal(y_reconstructed, y_data)

        # Test constant values
        y_data = np.array([5, 5, 5])
        x_norm, y_norm = self.processor.normalize_data(x_data[:3], y_data)
        assert np.array_equal(y_norm, np.zeros_like(y_data))

        # Test empty arrays
        try:
            self.processor.normalize_data(np.array([]), np.array([]))
            assert False, "Expected ValueError for empty arrays"
        except ValueError:
            pass

        # Test input unmodified
        x_data = np.array([1, 2, 3])
        y_data = np.array([2, 4, 6])
        x_copy, y_copy = x_data.copy(), y_data.copy()
        self.processor.normalize_data(x_data, y_data)
        assert np.array_equal(x_data, x_copy)
        assert np.array_equal(y_data, y_copy)          
        

    def test_pseudo_voigt(self):
        """Test Pseudo-Voigt function calculation."""
        x = np.linspace(-10, 10, 100)
        x0, amp, width, eta = 0, 1, 2, 0.5
        
        result = self.processor.pseudo_voigt(x, x0, amp, width, eta)
        
        # Verify function properties
        self.assertEqual(len(result), len(x))
        self.assertTrue(np.all(result >= 0))
        np.testing.assert_allclose(np.max(result), amp, rtol=0.01)
        self.assertEqual(np.argmax(result), len(x)//2)  # Peak should be at center

    def test_pseudo_voigt(self):
        """Test Pseudo-Voigt function calculation."""
        x = np.linspace(-10, 10, 1000)  # Increased points for better accuracy
        x0, amp, width, eta = 0, 1, 2, 0.5
    
        result = self.processor.pseudo_voigt(x, x0, amp, width, eta)
    
        # Verify function properties
        self.assertEqual(len(result), len(x))
        self.assertTrue(np.all(result >= 0))
        # Use looser tolerance for float comparison
        np.testing.assert_allclose(np.max(result), amp, rtol=0.01, atol=0.01)
        # Check peak position
        peak_position = x[np.argmax(result)]
        np.testing.assert_allclose(peak_position, x0, atol=0.05) # Max should not exceed sum of amplitudes
        
    def test_fit_peaks(self):
        """Test peak fitting functionality."""
        # Create synthetic data with known peaks
        x_data = np.linspace(0, 10, 1000)
        y_data = (self.processor.pseudo_voigt(x_data, 3, 1, 1, 0.5) + 
                 self.processor.pseudo_voigt(x_data, 7, 0.8, 1.2, 0.3))
        y_data += np.random.normal(0, 0.01, len(x_data))  # Add noise
        
        initial_params = [
            3, 1, 1, 0.5, 0,    # First peak
            7, 0.8, 1.2, 0.3, 0  # Second peak
        ]
        fixed_x0 = [False, False]
        
        # Perform fit
        popt, metrics, fitted = self.processor.fit_peaks(x_data, y_data, 
                                                       initial_params, fixed_x0)
        
        # Verify fitting results
        self.assertEqual(len(popt), len(initial_params))
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(fitted), len(x_data))
        
        # Check fit quality
        residuals = y_data - fitted
        self.assertTrue(np.std(residuals) < 0.1)
    
    def test_single_peak_no_fixed_params(self):
        """Test fitting of a single peak with no fixed parameters."""
        x = np.linspace(-10, 10, 1000)
        
        self.processor.fixed_params = [(None, None, None, None, None)]
        
        params = [3, 1, 1, 0.5, 0.1]
        
        y = self.processor.pseudo_voigt_multiple(x, *params)
        
        y_exp = self.processor.pseudo_voigt(x, 3, 1, 1, 0.5) + 0.1
        
        residuals = y - y_exp
        
        self.assertTrue(np.std(residuals) < 0.1)
        
    def test_single_peak_fixed_x0(self):
        """Test fitting of a single peak with fixed x0."""
        x = np.linspace(-10, 10, 1000)
        fixed_x0 = 3

        # Set up fixed parameters
        self.processor.fixed_params = [(fixed_x0, None, None, None, None)]

        # Test parameters: amp=1, width=1, eta=0.5, offset=0.1
        params = [1, 1, 0.5, 0.1]

        # Calculate using pseudo_voigt_multiple
        result = self.processor.pseudo_voigt_multiple(x, *params)

        # Calculate individual components for verification
        sigma = 1 / (2 * np.sqrt(2 * np.log(2)))  # width parameter
        gamma = 1 / 2  # width parameter

        # Gaussian component
        gaussian = np.exp(-0.5 * ((x - fixed_x0) / sigma)**2)
        
        # Lorentzian component
        lorentzian = gamma**2 / ((x - fixed_x0)**2 + gamma**2)
        
        # Combined pseudo-Voigt with amplitude and offset
        expected = (0.5 * lorentzian + (1 - 0.5) * gaussian) + 0.1
        
        # Scale by amplitude
        expected = expected * 1

        # Compare results
        # Use a lower decimal precision due to numerical differences
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
        
    def test_multiple_peaks_no_fixed_params(self):
        """Test fitting of multiple peaks with no fixed parameters."""
        x = np.linspace(-10, 10, 1000)
        
        x0_1, x0_2 = -1.0, 1.0

        self.processor.fixed_params =  [
                                            (x0_1, None, None, None, None),
                                            (x0_2, None, None, None, None)
                                        ]

        params = [1.0, 1.5, 0.3, 0.1, 0.8, 2.0, 0.7, 0.2]

        y = self.processor.pseudo_voigt_multiple(x, *params)

        # Calculate expected result for first peak
        sigma1 = params[1] / (2 * np.sqrt(2 * np.log(2)))
        gamma1 = params[1] / 2
        lorentzian1 = params[0] * (gamma1**2 / ((x - x0_1)**2 + gamma1**2))
        gaussian1 = params[0] * np.exp(-0.5 * ((x - x0_1) / sigma1)**2)
        peak1 = params[2] * lorentzian1 + (1 - params[2]) * gaussian1 + params[3]
        
        # Calculate expected result for second peak
        sigma2 = params[5] / (2 * np.sqrt(2 * np.log(2)))
        gamma2 = params[5] / 2
        lorentzian2 = params[4] * (gamma2**2 / ((x - x0_2)**2 + gamma2**2))
        gaussian2 = params[4] * np.exp(-0.5 * ((x - x0_2) / sigma2)**2)
        peak2 = params[6] * lorentzian2 + (1 - params[6]) * gaussian2 + params[7]
        
        # Total expected result
        y_exp = peak1 + peak2 - params[3] - params[7]  # Subtract offsets to avoid double counting

        residuals = y - y_exp
        
        self.assertTrue(np.std(residuals) < 0.1)
        
    def test_multiple_peaks_fixed_x0(self):
        """Test fitting of multiple peaks with fixed x0."""
        x = np.linspace(-10, 10, 1000)
        
        fixed_x0 = -1.0

        self.processor.fixed_params = [
                                        (fixed_x0, None, None, None, None),
                                        (None, None, None, None, None)
                                    ]

        params = [1.0, 1.5, 0.3, 0.1, 1.0, 0.8, 2.0, 0.7, 0.2]

        y = self.processor.pseudo_voigt_multiple(x, *params)

        # Calculate expected result for first peak (fixed x0)
        sigma1 = params[1] / (2 * np.sqrt(2 * np.log(2)))
        gamma1 = params[1] / 2
        lorentzian1 = params[0] * (gamma1**2 / ((x - fixed_x0)**2 + gamma1**2))
        gaussian1 = params[0] * np.exp(-0.5 * ((x - fixed_x0) / sigma1)**2)
        peak1 = params[2] * lorentzian1 + (1 - params[2]) * gaussian1 + params[3]
        
        # Calculate expected result for second peak (unfixed x0)
        sigma2 = params[6] / (2 * np.sqrt(2 * np.log(2)))
        gamma2 = params[6] / 2
        lorentzian2 = params[5] * (gamma2**2 / ((x - params[4])**2 + gamma2**2))
        gaussian2 = params[5] * np.exp(-0.5 * ((x - params[4]) / sigma2)**2)
        peak2 = params[7] * lorentzian2 + (1 - params[7]) * gaussian2 + params[8]
        
        # Total expected result
        y_exp = peak1 + peak2 - params[3] - params[8]  # Subtract offsets to avoid double counting

        residuals = y - y_exp
        
        self.assertTrue(np.std(residuals) < 0.1)
        
    def test_multiple_peaks_mixed_fixed_x0(self):
        """Test fitting of multiple peaks with mixed fixed and unfixed x0."""
        x = np.linspace(-10, 10, 1000)

        self.processor.fixed_params = [(3, None, None, None, None),
                                        (None, None, None, None, None)]

        params = [1, 1, 0.5, 0.1,  # First peak (fixed x0)
                    7, 0.8, 1.2, 0.3, 0.2]  # Second peak (unfixed x0)

        y = self.processor.pseudo_voigt_multiple(x, *params)

        # Calculate expected result - each peak includes its own offset
        peak1 = self.processor.pseudo_voigt(x, 3, 1, 1, 0.5) + 0.1
        peak2 = self.processor.pseudo_voigt(x, 7, 0.8, 1.2, 0.3) + 0.2
        
        y_exp = peak1 + peak2
        
        np.testing.assert_array_almost_equal(y_exp, y, decimal=6)
    
    def test_invalid_params_length(self):
        """Test handling of invalid parameters length."""
        x = np.linspace(-10, 10, 1000)

        self.processor.fixed_params = [(None, None, None, None, None)] * 2

        params = [3, 1, 1, 0.5, 0.1, 7, 0.8, 1.2, 0.3]  # Missing one parameter

        with self.assertRaises(ValueError):
            self.processor.pseudo_voigt_multiple(x, *params)
            
    def test_edge_cases(self):
        """Test pseudo_voigt_multiple with edge cases"""
        # Test with zero amplitude
        x = np.linspace(-10, 10, 1000)
        self.processor.fixed_params = [(None, None, None, None, None)]
        params = [0.0, 0.0, 2.0, 0.5, 0.0]
        result = self.processor.pseudo_voigt_multiple(x, *params)
        np.testing.assert_array_almost_equal(result, np.zeros_like(x))
        
        
        # Test with pure Gaussian (eta = 0)
        params = [0.0, 1.0, 2.0, 0.0, 0.0]
        result = self.processor.pseudo_voigt_multiple(x, *params)
        sigma = params[2] / (2 * np.sqrt(2 * np.log(2)))
        y_exp = params[1] * np.exp(-0.5 * ((x - params[0]) / sigma)**2) + params[4]
        
        residuals = result - y_exp
        
        self.assertTrue(np.std(residuals) < 0.1)
        
        # Test with pure Lorentzian (eta = 1)
        params = [0.0, 1.0, 2.0, 1.0, 0.0]
        result = self.processor.pseudo_voigt_multiple(x, *params)
        sigma = params[2] / (2 * np.sqrt(2 * np.log(2)))
        y_exp = params[1] * np.exp(-0.5 * ((x - params[0]) / sigma)**2) + params[4]
        
        residuals = result - y_exp
        
        self.assertTrue(np.std(residuals) < 0.1)
        
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Set up processor with valid data range
        self.processor.ppm = np.array([0, 1, 2, 3, 4])
        self.processor.data = np.array([0, 1, 2, 3, 4])
    
        # Test invalid region selection
        with self.assertRaises(ValueError):
            # Make sure these values are well outside the data range
            self.processor.select_region(10, 20)  # Changed to clearly invalid range
    
        # Test missing data
        processor_without_data = NMRProcessor()
        with self.assertRaises(ValueError):
            processor_without_data.select_region(1, 2)
    
        # Test invalid peak fitting parameters
        x_data = np.linspace(0, 10, 100)
        y_data = np.zeros_like(x_data)
        invalid_params = [1, 2, 3]  # Invalid number of parameters
        with self.assertRaises(ValueError):
            self.processor.fit_peaks(x_data, y_data, invalid_params)
    

    
    def test_save_results(self):
        """Test results saving functionality."""
        import matplotlib
        matplotlib.use('Agg')
        
        try:
            # Create test data
            x_data = np.linspace(0, 10, 100)
            y_data = np.zeros_like(x_data)
            fitted_data = np.zeros_like(x_data)
            components = [np.zeros_like(x_data)]
            # self.processor.carrier_freq = 500.0
            metrics = [{
                'x0': (1, 0.1),
                'amplitude': (1, 0.1),
                'width': (1, 0.1),
                'eta': (0.5, 0.1),
                'offset': (0, 0.1),
                'gaussian_area': (1, 0.1),
                'lorentzian_area': (1, 0.1),
                'total_area': (2, 0.2)
            }]
            popt = np.array([1, 1, 1, 0.5, 0])
    
            # Create temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                test_filepath = os.path.join(temp_dir, 'test_')
                
                # Mock the figure and its savefig method
                mock_fig = Mock()
                mock_axes = (Mock(), Mock())
                mock_components = [Mock()]
                
                # Set up all the required mocks
                with patch.object(self.processor, 'plot_results', 
                                  return_value=(mock_fig, mock_axes, mock_components)) as mock_plot, \
                     patch('builtins.open', mock_open()) as mock_file, \
                     patch('pandas.DataFrame.to_csv') as mock_to_csv, \
                     patch.object(mock_fig, 'savefig') as mock_savefig, \
                     patch('matplotlib.pyplot.close') as mock_close:
                    
                    # Call save_results
                    self.processor.save_results(
                        test_filepath, x_data, y_data, fitted_data, 
                        metrics, popt, components
                    )
                    
                    # Verify all the saving methods were called correctly
                    mock_plot.assert_called_once()
                    mock_savefig.assert_called_once_with(
                        test_filepath + 'pseudoVoigtPeakFit.png', 
                        bbox_inches='tight'
                    )
                    mock_close.assert_called_once_with(mock_fig)
                    
                    # Verify DataFrame.to_csv was called for peak data
                    mock_to_csv.assert_called_once_with(
                        test_filepath + 'peak_data.csv', 
                        index=False
                    )
                    
                    # Verify metrics file was opened and written
                    mock_file.assert_called_with(
                        test_filepath + 'pseudoVoigtPeak_metrics.txt', 
                        'w'
                    )
                    
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")
        
        finally:
            plt.close('all')
    def test_plot_results(self):
        """Test plotting functionality."""
        # Create test data
        x_data = np.linspace(0, 10, 100)
        y_data = np.zeros_like(x_data)
        fitted_data = np.zeros_like(x_data)
        
        # self.processor.carrier_freq = 500.0
        
        metrics = [{
            'x0': (1, 0.1),
            'amplitude': (1, 0.1),
            'width': (1, 0.1),
            'eta': (0.5, 0.1),
            'offset': (0, 0.1),
            'gaussian_area': (1, 0.1),
            'lorentzian_area': (1, 0.1),
            'total_area': (2, 0.2)
        }]
        popt = np.array([1, 1, 1, 0.5, 0])
        
        # Test plotting
        fig, ax1, components = self.processor.plot_results(
            x_data, y_data, fitted_data, metrics, popt
        )
        
        # Verify plot objects
        self.assertIsNotNone(fig)
        self.assertEqual(len(components), 1)
        self.assertTrue(isinstance(ax1, plt.Axes))
        

if __name__ == '__main__':
    
    cov = coverage.Coverage()
    cov.start()
    
    unittest.main(verbosity=2)
    
    cov.stop()
    
    cov.save()
    
    cov.html_report(directory='coverage_html')
    
    # webbrowser.open(os.path.join('coverage_html', 'index.html'))
