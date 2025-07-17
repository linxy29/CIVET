---
title: "CIVET Tutorial: Single-Cell Clonal Evolution Analysis"
author: "Your Name"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    toc_float: true
    code_folding: show
    theme: flatly
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
```

# Overview

CIVET (Clonal Inference from Variant Expression in Transcriptomes) is an R package for analyzing clonal evolution patterns in single-cell genomic data. This tutorial will guide you through the main functions and workflow for detecting clone-associated variants using generalized linear models.

## Installation and Setup

First, load the required libraries:

```{r libraries, eval=FALSE}
library(tidyverse)
library(Seurat)
library(aod)  # For beta-binomial regression
```

Then, load the CIVET functions:

```{r load_civet_functions, eval=FALSE}
# Load CIVET functions from the civet_function.R file
source("civet_function.R")
```

**Note:** Make sure the `civet_function.R` file is in your working directory or provide the full path to the file. The file contains the core CIVET functions:
- `load_matrices()`: Loads and processes input data matrices
- `civet()`: Performs statistical testing for clone-associated variants
- `read_mtx_safe()`: Helper function for safe matrix reading
- `CheckDf()`: Data validation function
- `new.wald.test()`: Custom Wald test implementation

## Core Functions

### 1. `load_matrices()` - Loading and Processing Input Data

This function loads allelic depth (AD) and total depth (DP) matrices from cellSNP output files.

```{r load_matrices_example, eval=FALSE}
# Basic usage
matrices <- load_matrices(base_path = "/path/to/cellSNP/output")

# With custom file paths
matrices <- load_matrices(
  base_path = NULL,
  ad_mtx_path = "path/to/cellSNP.tag.AD.mtx",
  dp_mtx_path = "path/to/cellSNP.tag.DP.mtx",
  features_path = "path/to/cellSNP.variants.tsv",
  cells_path = "path/to/cellSNP.samples.tsv"
)
```

**Input Files Required:**
- `cellSNP.tag.AD.mtx`: Alternative allele depth matrix
- `cellSNP.tag.DP.mtx`: Total depth matrix
- `cellSNP.variants.tsv`: Variant information (auto-generated if missing)
- `cellSNP.samples.tsv`: Cell barcode information

**Output:**
Returns a list containing three matrices:
- `AD`: Alternative allele depth matrix
- `DP`: Total depth matrix  
- `AF`: Allele frequency matrix (AD/DP)

### 2. `civet()` - Statistical Testing for Clone-Associated Variants

This is the main analysis function that performs statistical tests to identify variants associated with clonal populations.

```{r civet_example, eval=FALSE}
# Basic usage
results <- civet(
  AD_mat = matrices$AD,
  DP_mat = matrices$DP,
  clone_mat = clone_assignments,
  minDP =20 use_random_effect = FALSE
)
```

**Parameters:**
- `AD_mat`: Alternative allele depth matrix (variants × cells)
- `DP_mat`: Total depth matrix (variants × cells)
- `clone_mat`: Clone assignment matrix (cells × clones)
- `minDP`: Minimum depth threshold for including a cell (default: 20)
- `use_random_effect`: Whether to use random effects model (default: FALSE)

**Output:**
Returns a list of matrices containing:
- `LR_vals`: Likelihood ratio values
- `LRT_pvals`: Likelihood ratio test p-values
- `Wald_pvals`: Wald test p-values (fixed effects only)
- `ANOVA_pvals`: ANOVA p-values (random effects only)
- `LRT_fdr`: FDR-corrected p-values

## Complete Workflow Example

Here's a complete example using the provided wrapper function:

### Step 1: Prepare Your Data Structure

Your data should be organized as follows:
```
example_data/
├── metadata/
│   └── simulation_cell_metadata.csv
└── cellSNP/
    ├── cellSNP.tag.AD.mtx
    ├── cellSNP.tag.DP.mtx
    ├── cellSNP.tag.barcodes.txt
    └── cellSNP.tag.mutations.txt
```

### Step 2: Run the Analysis Step by Step

#### 2.1 Define File Paths

```{r file_paths, eval=FALSE}
# Set your base directory
subrun_dir <-example_data"

# Define file paths
metadata_csv <- file.path(subrun_dir, "metadata",simulation_cell_metadata.csv)
ad_mtx_path  <- file.path(subrun_dir, cellSNP,cellSNP.tag.AD.mtx)
dp_mtx_path  <- file.path(subrun_dir, cellSNP,cellSNP.tag.DP.mtx)
barcodes_txt <- file.path(subrun_dir, cellSNP",cellSNP.tag.barcodes.txt")
mutations_txt <- file.path(subrun_dir, cellSNP", "cellSNP.tag.mutations.txt")
```

#### 2.2 Check Required Files

```{r check_files, eval=FALSE}
# Check for required files
required_files <- c(metadata_csv, ad_mtx_path, dp_mtx_path, barcodes_txt, mutations_txt)
missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop("Missing files: ", paste(missing_files, collapse = ", "))
} else {
  cat("All required files found!\n")
}
```

#### 2.3 Load and Prepare Metadata

```{r load_metadata, eval=FALSE}
# Read metadata
metadata <- read_csv(metadata_csv)
head(metadata)

# Prepare clone matrix
# Adjust column selection based on your metadata structure
clone_mat <- metadata %>%
  rename(cell_id = 1) %>%  # Rename first column to cell_id
  column_to_rownames("cell_id") %>%
  dplyr::select(generation)  # Select clone-related columns

# Check clone matrix structure
dim(clone_mat)
head(clone_mat)
```

#### 2.4 Load Matrices Using CIVET Functions

```{r load_matrices, eval=FALSE}
# Option 1: Use load_matrices function (if you have cellSNP base path)
# matrices <- load_matrices(base_path = file.path(subrun_dir, "cellSNP"))

# Option 2: Load matrices directly using read_mtx_safe
AD_mtx <- read_mtx_safe(ad_mtx_path, mutations_txt, barcodes_txt)
DP_mtx <- read_mtx_safe(dp_mtx_path, mutations_txt, barcodes_txt)

# Check matrix dimensions
cat("AD matrix dimensions:", dim(AD_mtx), "\n")
cat("DP matrix dimensions:", dim(DP_mtx), "\n")
cat("Number of variants:", nrow(AD_mtx), "\n")
cat("Number of cells:", ncol(AD_mtx), "\n")
```

#### 2.5 Data Quality Check and Subset

```{r data_subset, eval=FALSE}
# Find common barcodes between clone matrix and expression matrices
common_barcodes <- intersect(rownames(clone_mat), colnames(AD_mtx))
cat("Common barcodes:", length(common_barcodes), "\n")
cat("Clone matrix cells:", nrow(clone_mat), "\n")
cat("Expression matrix cells:", ncol(AD_mtx), "\n")

# Subset matrices to common barcodes
subset_AD <- AD_mtx[, common_barcodes, drop = FALSE]
subset_DP <- DP_mtx[, common_barcodes, drop = FALSE]
subset_clones <- clone_mat[common_barcodes, , drop = FALSE]

# Verify dimensions match
cat("Final dimensions:\n")
cat("AD matrix:", dim(subset_AD), "\n")
cat("DP matrix:", dim(subset_DP), "\n")
cat("Clone matrix:", dim(subset_clones), "\n")
```

#### 2.6 Run Statistical Analysis

```{r run_analysis, eval=FALSE}
# Run CIVET analysis
cat("Running CIVET analysis...\n")
results <- civet(
  AD_mat = subset_AD,
  DP_mat = subset_DP,
  clone_mat = subset_clones,
  minDP = 5,  # Minimum depth threshold
  use_random_effect = FALSE  # Use fixed effects model
)

# Check results structure
str(results)
```

#### 2.7 Process and Save Results

```{r save_results, eval=FALSE}
# Convert results to data frame format
results_df <- purrr::imap_dfr(
  results,
  ~ as.data.frame(.x) %>%
    tibble::rownames_to_column("variant") %>%
    mutate(test_type = .y)
)

# Create output directory
outdir <- file.path(subrun_dir, "civet_res")
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}

# Save results in multiple formats
saveRDS(results, file = file.path(outdir, "civet_results.rds"))
write.csv(results_df, file = file.path(outdir, "civet_results.csv"), row.names = FALSE)

# Save summary statistics
summary_stats <- data.frame(
  n_variants = nrow(subset_AD),
  n_cells = ncol(subset_AD),
  n_clones = ncol(subset_clones),
  min_depth_threshold = 5,
  analysis_date = Sys.Date()
)

write.csv(summary_stats, file = file.path(outdir, "analysis_summary.csv"), row.names = FALSE)

cat("Analysis completed! Results saved to:", outdir, "\n")
```

### Step 3: Interpret Results

```{r results_interpretation, eval=FALSE}
# Load saved results
results <- readRDS("example_data/civet_res/civet_results.rds")

# View significant variants (FDR <00.05)
significant_variants <- which(results$LRT_fdr < 0000.5, arr.ind = TRUE)
print(significant_variants)

# Extract p-values for specific clone
clone1_pvals <- results$LRT_pvals[,1]
significant_clone1 <- names(clone1_pvals)clone1vals <000.5 & !is.na(clone1_pvals)]
```

## Key Features

### Statistical Models
- **Fixed Effects Model**: Tests association between variants and clone assignments using beta-binomial regression
- **Random Effects Model**: Incorporates random effects for more complex experimental designs
- **Multiple Testing Correction**: FDR adjustment for genome-wide significance

### Quality Control
- **Minimum Depth Filtering**: Excludes low-coverage cells to improve reliability
- **Data Validation**: Checks matrix dimensions and data consistency
- **Error Handling**: Robust error handling for failed model fits

### Flexible Input
- **Multiple Clone Types**: Supports continuous (generation) or categorical (cell type) clone assignments
- **Custom Thresholds**: Adjustable minimum depth and significance thresholds
- **Batch Processing**: Designed for analyzing multiple datasets or conditions

## Tips for Best Results

1. **Data Quality**: Ensure good coverage (>10-20 reads per variant per cell)
2. **Clone Assignment**: Use high-confidence clone assignments from phylogenetic analysis
3. **Filtering**: Apply appropriate quality filters before analysis
4. **Multiple Testing**: Always use FDR correction for multiple comparisons
5. **Validation**: Validate significant variants using independent methods

## Troubleshooting

**Common Issues:**
- **Matrix Dimension Mismatch**: Ensure all matrices have matching cell/variant names
- **Low Coverage**: Increase `minDP` parameter for noisy data
- **Model Convergence**: Some variants may fail to converge; this is normal and handled gracefully
- **Missing Files**: Check file paths and ensure all required files exist

## Output Files

The analysis generates:
- `civet_results.rds`: Complete results object for further analysis
- `civet_results.csv`: Tabular format for easy viewing and downstream analysis

## Session Information

```{r session_info, eval=FALSE}
sessionInfo()
```

This tutorial provides a comprehensive guide to using CIVET for single-cell clonal evolution analysis. The package is 
designed to be robust and user-friendly while providing powerful statistical methods for detecting clone-associated genomic 
variants.