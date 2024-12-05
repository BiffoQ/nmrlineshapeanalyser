# nmrlineshapeanalyser: ss-NMR peak shape deconvolution and line shape analysis

nmrlineshapeanalyser is an open-source Python package designed to make peak deconvolution or line shape analysis easier.

This package is for now only compatible with Bruker's NMR data.

# Key Features

 - Load and process Bruker NMR data
 - Select and analyse specific spectral regions
 - Perform peak fitting using Pseudo-Voigt profiles
 - Calculate detailed peak metrics and statistics
 - Generate publication-quality visuals
 - Export results in various formats:
       - txt: calculated peak metrics and statistics
       - png: visualisation of the fitted spectral regions
       - csv: save plots to a file
  
# Install

```bash
pip install nmrlineshapeanalyser
```

# Dependencies

The following packages are required:

```bash
nmrglue 
numpy >= 1.26.0
scipy 
matplotlib >= 3.9.0
pandas >= 2.2.0
```

You can install these dependencies using pip:

```bash
pip install nmrglue numpy>=1.26.0 scipy matplotlib>=3.9.0 pandas>=2.2.0
```

# A Single Peak Fitting Example
```python
from nmrlineshapeanalyser.core import NMRProcessor

#create NMRProcessor object

processor = NMRProcessor()

#Load filepath: always include the '\\'
filepath = r"..\data\single_peak\10\pdata\1\\"

# Load the data

processor.load_data(filepath)

#Select the region of interest

x_data, y_data = processor.select_region(512, 650)

#Normalize the data and return normalised y_axis and the corresponding x_axis

x_data, y_normalized = processor.normalize_data(x_data, y_data)

#define initial parameters for the fitting
#this example is for a single peak
#format of the parameters is [x0, amplitude, width, eta, offset]
# x0 (position), amplitude, width, eta (mixing parameter), offset
#x0 has to be close to the peak position
initial_params = [
581, 0.12, 40.51, 0.89, -143.115, 
  ]

# fixed_x0 controls whether peak positions should be fixed during fitting
# False means position can vary, True means position is fixed
fixed_x0 = [False]

#FIt the data
popt, metrics, fitted = processor.fit_peaks(x_data, y_normalized, initial_params, fixed_x0)

#popt is the optimized parameters
#metrics is the metrics of the fitting
#fitted is the fitted curve data

#Plot and examine the results of the fitting
fig, axes, components = processor.plot_results(x_data, y_normalized, fitted, metrics, popt)

#Save the figure as an svg file and the results as a csv file
processor.save_results(filepath, x_data, y_normalized, fitted, metrics, popt, components)
```
This should generate the image below.

![Single Peak Fit](data/single_peak/10/pdata/1/pseudoVoigtPeakFit.png){ width=600 height=400 }


# Contact

For questions and support, please open an issue in the GitHub repository.
