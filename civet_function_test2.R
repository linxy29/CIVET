# Test code for civet function
# Load required library
library(aod)
source('civet_function_v2.R', chdir = TRUE)

# ============================================================================
# Test 1: Basic functionality with small dataset (base_model = NULL)
# ============================================================================
cat("Test 1: Basic functionality with base_model = NULL\n")
cat("===================================================\n")

set.seed(123)
n_variants <- 10
n_cells <- 50
n_clones <- 3

# Create test matrices
AD_mat_test1 <- matrix(rbinom(n_variants * n_cells, size = 100, prob = 0.3), 
                       nrow = n_variants, ncol = n_cells)
DP_mat_test1 <- matrix(sample(50:150, n_variants * n_cells, replace = TRUE), 
                       nrow = n_variants, ncol = n_cells)
clone_mat_test1 <- matrix(rnorm(n_cells * n_clones), 
                          nrow = n_cells, ncol = n_clones)

# Set row and column names
rownames(AD_mat_test1) <- paste0("var_", 1:n_variants)
colnames(AD_mat_test1) <- paste0("cell_", 1:n_cells)
rownames(DP_mat_test1) <- paste0("var_", 1:n_variants)
colnames(DP_mat_test1) <- paste0("cell_", 1:n_cells)
rownames(clone_mat_test1) <- paste0("cell_", 1:n_cells)
colnames(clone_mat_test1) <- paste0("clone_", 1:n_clones)

# Run test
result_test1 <- civet(AD_mat = AD_mat_test1, 
                      DP_mat = DP_mat_test1, 
                      clone_mat = clone_mat_test1, 
                      minDP = 20, 
                      use_random_effect = FALSE,
                      base_model = NULL)

cat("Result dimensions:\n")
print(sapply(result_test1, dim))
cat("\nSample LR values (first 5 variants, all clones):\n")
print(result_test1$LR_vals[1:5, ])
cat("\nSample LRT p-values (first 5 variants, all clones):\n")
print(result_test1$LRT_pvals[1:5, ])

# ============================================================================
# Test 2: base_model = "full"
# ============================================================================
cat("\n\nTest 2: base_model = 'full'\n")
cat("============================\n")

result_test2 <- civet(AD_mat = AD_mat_test1, 
                      DP_mat = DP_mat_test1, 
                      clone_mat = clone_mat_test1, 
                      minDP = 20, 
                      use_random_effect = FALSE,
                      base_model = "full")

cat("Result dimensions:\n")
print(sapply(result_test2, dim))
cat("\nSample LR values (first 5 variants, all clones):\n")
print(result_test2$LR_vals[1:5, ])
cat("\nComparison: base_model=NULL vs base_model='full'\n")
cat("Clone 1 LR values differ: ", 
    !identical(result_test1$LR_vals[,1], result_test2$LR_vals[,1]), "\n")

# ============================================================================
# Test 3: Single clone (edge case for base_model = "full")
# ============================================================================
cat("\n\nTest 3: Single clone matrix (edge case)\n")
cat("=========================================\n")

clone_mat_test3 <- matrix(rnorm(n_cells), nrow = n_cells, ncol = 1)
rownames(clone_mat_test3) <- paste0("cell_", 1:n_cells)
colnames(clone_mat_test3) <- "clone_1"

result_test3 <- civet(AD_mat = AD_mat_test1, 
                      DP_mat = DP_mat_test1, 
                      clone_mat = clone_mat_test3, 
                      minDP = 20, 
                      use_random_effect = FALSE,
                      base_model = "full")

cat("Single clone test completed successfully\n")
cat("Result dimensions:\n")
print(sapply(result_test3, dim))

# ============================================================================
# Test 4: With random effects
# ============================================================================
cat("\n\nTest 4: With random effects (base_model = NULL)\n")
cat("================================================\n")

result_test4 <- civet(AD_mat = AD_mat_test1, 
                      DP_mat = DP_mat_test1, 
                      clone_mat = clone_mat_test1, 
                      minDP = 20, 
                      use_random_effect = TRUE,
                      base_model = NULL)

cat("ANOVA p-values available: ", !all(is.na(result_test4$ANOVA_pvals)), "\n")
cat("\nSample ANOVA p-values (first 5 variants, all clones):\n")
print(result_test4$ANOVA_pvals[1:5, ])

# ============================================================================
# Test 5: Low coverage variants (many should be NA)
# ============================================================================
cat("\n\nTest 5: Low coverage variants\n")
cat("==============================\n")

DP_mat_test5 <- matrix(sample(1:30, n_variants * n_cells, replace = TRUE), 
                       nrow = n_variants, ncol = n_cells)
rownames(DP_mat_test5) <- paste0("var_", 1:n_variants)
colnames(DP_mat_test5) <- paste0("cell_", 1:n_cells)

result_test5 <- civet(AD_mat = AD_mat_test1, 
                      DP_mat = DP_mat_test5, 
                      clone_mat = clone_mat_test1, 
                      minDP = 25,  # High threshold
                      use_random_effect = FALSE,
                      base_model = NULL)

cat("Proportion of NA values in LR_vals: ", 
    mean(is.na(result_test5$LR_vals)), "\n")

# ============================================================================
# Test 6: Error handling - inconsistent matrices
# ============================================================================
cat("\n\nTest 6: Error handling - inconsistent matrices\n")
cat("===============================================\n")

AD_mat_wrong <- AD_mat_test1[, 1:40]  # Different number of cells

result_test6 <- tryCatch({
  civet(AD_mat = AD_mat_wrong, 
        DP_mat = DP_mat_test1, 
        clone_mat = clone_mat_test1, 
        minDP = 20)
}, error = function(e) {
  cat("Caught expected error: ", e$message, "\n")
  return(NULL)
})

# ============================================================================
# Test 7: Error handling - invalid base_model
# ============================================================================
cat("\n\nTest 7: Error handling - invalid base_model\n")
cat("============================================\n")

result_test7 <- tryCatch({
  civet(AD_mat = AD_mat_test1, 
        DP_mat = DP_mat_test1, 
        clone_mat = clone_mat_test1, 
        minDP = 20,
        base_model = "invalid")
}, error = function(e) {
  cat("Caught expected error: ", e$message, "\n")
  return(NULL)
})

# ============================================================================
# Test 8: Matrices without names (should auto-generate)
# ============================================================================
cat("\n\nTest 8: Matrices without names\n")
cat("===============================\n")

AD_mat_nonames <- matrix(rbinom(50, size = 100, prob = 0.3), nrow = 5, ncol = 10)
DP_mat_nonames <- matrix(sample(50:150, 50, replace = TRUE), nrow = 5, ncol = 10)
clone_mat_nonames <- matrix(rnorm(20), nrow = 10, ncol = 2)

result_test8 <- civet(AD_mat = AD_mat_nonames, 
                      DP_mat = DP_mat_nonames, 
                      clone_mat = clone_mat_nonames, 
                      minDP = 20,
                      base_model = NULL)

cat("Auto-generated variant names: ", rownames(result_test8$LR_vals)[1:3], "\n")
cat("Auto-generated clone names: ", colnames(result_test8$LR_vals), "\n")

# ============================================================================
# Test 9: Compare null vs full base models
# ============================================================================
cat("\n\nTest 9: Detailed comparison of null vs full base models\n")
cat("========================================================\n")

# Create a larger dataset for better comparison
set.seed(456)
n_variants_large <- 20
n_cells_large <- 100
n_clones_large <- 4

AD_mat_large <- matrix(rbinom(n_variants_large * n_cells_large, size = 100, prob = 0.3), 
                       nrow = n_variants_large, ncol = n_cells_large)
DP_mat_large <- matrix(sample(80:200, n_variants_large * n_cells_large, replace = TRUE), 
                       nrow = n_variants_large, ncol = n_cells_large)
clone_mat_large <- matrix(rnorm(n_cells_large * n_clones_large), 
                          nrow = n_cells_large, ncol = n_clones_large)

rownames(AD_mat_large) <- paste0("var_", 1:n_variants_large)
colnames(AD_mat_large) <- paste0("cell_", 1:n_cells_large)
rownames(DP_mat_large) <- paste0("var_", 1:n_variants_large)
colnames(DP_mat_large) <- paste0("cell_", 1:n_cells_large)
rownames(clone_mat_large) <- paste0("cell_", 1:n_cells_large)
colnames(clone_mat_large) <- paste0("clone_", 1:n_clones_large)

result_null <- civet(AD_mat = AD_mat_large, 
                     DP_mat = DP_mat_large, 
                     clone_mat = clone_mat_large, 
                     minDP = 20,
                     base_model = NULL)

result_full <- civet(AD_mat = AD_mat_large, 
                     DP_mat = DP_mat_large, 
                     clone_mat = clone_mat_large, 
                     minDP = 20,
                     base_model = "full")

cat("\nSummary statistics for base_model = NULL:\n")
cat("Mean LR value: ", mean(result_null$LR_vals, na.rm = TRUE), "\n")
cat("Median LRT p-value: ", median(result_null$LRT_pvals, na.rm = TRUE), "\n")

cat("\nSummary statistics for base_model = 'full':\n")
cat("Mean LR value: ", mean(result_full$LR_vals, na.rm = TRUE), "\n")
cat("Median LRT p-value: ", median(result_full$LRT_pvals, na.rm = TRUE), "\n")

cat("\nCorrelation between LR values (null vs full) for clone 1:\n")
valid_idx <- !is.na(result_null$LR_vals[,1]) & !is.na(result_full$LR_vals[,1])
if (sum(valid_idx) > 1) {
  cat("Correlation: ", 
      cor(result_null$LR_vals[valid_idx,1], result_full$LR_vals[valid_idx,1]), "\n")
}

cat("\n==================\n")
cat("All tests completed!\n")
cat("==================\n")