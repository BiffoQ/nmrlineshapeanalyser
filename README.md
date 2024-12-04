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
nmrglue >= 0.10
numpy >= 1.26.4
scipy >= 1.13.1
matplotlib >= 3.9.0
pandas >= 2.2.2
```

You can install these dependencies using pip:

```bash
pip install nmrglue>=0.1.0 numpy>=1.26.4 scipy>=1.13.1 matplotlib>=3.9.0 pandas>=2.2.2
```

# Contact

For questions and support, please open an issue in the GitHub repository.
