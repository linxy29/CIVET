# CIVET: Clonal Information based mitochondrial Variation idEnTification

[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CIVET is an R package for analyzing clonal evolution patterns in single-cell mitochondrial genomic data. It provides statistical methods to detect clone-associated variants using generalized linear models, specifically designed for single-cell sequencing data with mitochondrial information.

## 🚀 Features

- **Statistical Analysis**: Beta-binomial regression models for variant-clone associations
- **Multiple Testing**: FDR correction for genome-wide significance
- **Flexible Models**: Support for both fixed and random effects models
- **Quality Control**: Built-in depth filtering and data validation
- **Robust Error Handling**: Graceful handling of model convergence issues
- **Multiple Input Formats**: Support for various cellSNP output formats

## 📋 Prerequisites

### Required R Packages
```r
install.packages(c("tidyverse", Seurat", "aod"))
```

### Required System Dependencies
- R (version 4.0 or higher)
- Unix-like system with shell commands (for VCF processing)

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/CIVET.git
cd CIVET
```
2. Load the CIVET functions in R:
```r
source("civet_function.R")
```

## 📁 Project Structure

```
CIVET/
├── README.md                    # This file
├── civet_tutorial.md           # Comprehensive tutorial
├── civet_function.R            # Core CIVET functions
├── civet_function_allgroup.R   # Multi-group analysis functions
├── civet_function_test.R       # Testing functions
├── civet_function_twogroup.R   # Two-group analysis functions
├── civet_function_v1.R         # Version 1 functions
└── example_data/               # Example datasets
    ├── cellSNP/               # cellSNP output files
    │   ├── cellSNP.tag.AD.mtx
    │   ├── cellSNP.tag.DP.mtx
    │   ├── cellSNP.tag.barcodes.txt
    │   └── cellSNP.tag.mutations.txt
    └── metadata/              # Cell metadata
        └── simulation_cell_metadata.csv
```

## 🔧 Usage

### Basic Workflow

1. Required Libraries:
```r
library(tidyverse)
library(Seurat)
library(aod)
source("civet_function.R")
```

2. Load Input Data:
```r
# Load matrices from cellSNP output
matrices <- load_matrices(base_path = "path/to/cellSNP/output")

# Or load directly with custom paths
matrices <- load_matrices(
  ad_mtx_path = "path/to/cellSNP.tag.AD.mtx",
  dp_mtx_path = "path/to/cellSNP.tag.DP.mtx",
  features_path = "path/to/cellSNP.variants.tsv",
  cells_path = "path/to/cellSNP.samples.tsv"
)
```

3. Prepare Clone Assignments:
```r
# Read metadata and prepare clone matrix
metadata <- read_csv("metadata.csv")
clone_mat <- metadata %>%
  rename(cell_id = 1) %>%
  column_to_rownames("cell_id") %>%
  dplyr::select(generation)  # Select clone-related columns
```

4. Run CIVET Analysis:
```r
results <- civet(
  AD_mat = matrices$AD,
  DP_mat = matrices$DP,
  clone_mat = clone_mat,
  minDP = 20,  # Minimum depth threshold
  use_random_effect = FALSE  # Use fixed effects model
)
```

5. Process Results:
```r
# Convert to data frame format
results_df <- purrr::imap_dfr(
  results,
  ~ as.data.frame(.x) %>%
    tibble::rownames_to_column("variant") %>%
    mutate(test_type = .y)
)

# Save results
write.csv(results_df, "civet_results.csv", row.names = FALSE)
```

### Complete Example

See `civet_tutorial.md` for a complete step-by-step tutorial with example data.

## 📊 Input Data Requirements

### Required Files
- **AD Matrix**: Alternative allele depth matrix (`cellSNP.tag.AD.mtx`)
- **DP Matrix**: Total depth matrix (`cellSNP.tag.DP.mtx`)
- **Variants File**: Variant information (`cellSNP.variants.tsv`)
- **Cells File**: Cell barcode information (`cellSNP.samples.tsv`)
- **Clone Assignments**: Cell-to-clone mapping matrix

### Data Format
- Matrices should have variants as rows and cells as columns
- Clone matrix should have cells as rows and clone factors as columns
- All matrices must have matching cell/variant names

## 📈 Output

The analysis generates several result matrices:
- `LR_vals`: Likelihood ratio values
- `LRT_pvals`: Likelihood ratio test p-values
- `Wald_pvals`: Wald test p-values (fixed effects only)
- `ANOVA_pvals`: ANOVA p-values (random effects only)
- `LRT_fdr`: FDR-corrected p-values

## 🔬 Statistical Methods

### Models
- **Fixed Effects**: Beta-binomial regression with clone assignments as fixed effects
- **Random Effects**: Beta-binomial regression with clone assignments as random effects

### Quality Control
- Minimum depth filtering (`minDP` parameter)
- Data validation and consistency checks
- Graceful handling of model convergence failures

### Multiple Testing
- FDR correction using Benjamini-Hochberg method
- Genome-wide significance assessment

## 🎯 Key Functions

### `load_matrices()`
Loads and processes input data matrices from cellSNP output.

**Parameters:**
- `base_path`: Path to cellSNP output directory
- `ad_mtx_path`: Path to AD matrix file
- `dp_mtx_path`: Path to DP matrix file
- `features_path`: Path to variants file
- `cells_path`: Path to cells file

### `civet()`
Main analysis function for detecting clone-associated variants.

**Parameters:**
- `AD_mat`: Alternative allele depth matrix
- `DP_mat`: Total depth matrix
- `clone_mat`: Clone assignment matrix
- `minDP`: Minimum depth threshold (default: 20`use_random_effect`: Use random effects model (default: FALSE)

### `CheckDf()`
Data validation function for quality control.

## 🐛 Troubleshooting

### Common Issues

1. **Matrix Dimension Mismatch**
   - Ensure all matrices have matching cell/variant names
   - Check that clone matrix has same cells as expression matrices
2. **Low Coverage**
   - Increase `minDP` parameter for noisy data
   - Consider filtering low-coverage variants

3. **Model Convergence**
   - Some variants may fail to converge (handled gracefully)
   - Check data quality and clone assignments

4. **Missing Files**
   - Verify all required files exist
   - Check file paths and permissions

### Error Messages

- `"Cells in AD, DP, and clone matrices are not identical"`: Check cell barcode matching
- `"AD and DP matrices have different dimensions"`: Verify matrix consistency
- `"Matrix file does not exist"`: Check file paths

## 📚 Documentation

- **Tutorial**: See `civet_tutorial.md` for comprehensive usage guide
- **Function Documentation**: Inline documentation in R function files
- **Example Data**: Use `example_data/` for testing and learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the `aod` package for beta-binomial regression
- Designed for single-cell genomic data analysis
- Inspired by clonal evolution research needs

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the tutorial documentation
- Review the example data and workflows

---

**CIVET**: Empowering single-cell clonal evolution analysis through robust statistical methods. 