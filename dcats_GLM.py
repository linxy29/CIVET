import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.formula.api import ols

def multinom_EM(X, simMM, min_iter=10, max_iter=1000, logLik_threshold=1e-2):
    # Ensure the shape of simMM is correct; row sums should be 1
    K = simMM.shape[1]

    # Initialization
    mu = np.random.dirichlet(np.ones(K))
    Z = np.empty((K, K))
    logLik_old = logLik_new = np.sum(np.log(np.dot(mu, simMM)) * X)

    for it in range(max_iter):
        # E step: expectation of count from each component
        for i in range(K):
            for j in range(K):
                Z[i, j] = simMM[i, j] * mu[i] / np.sum(mu * simMM[:, j])
        
        # M step: maximizing likelihood
        mu = np.dot(Z, X)
        mu = mu / np.sum(mu)

        # Check convergence
        logLik_new = np.sum(np.log(np.dot(mu, simMM)) * X)
        if it > min_iter and logLik_new - logLik_old < logLik_threshold:
            break
        else:
            logLik_old = logLik_new
    
    # Return values
    return {
        "mu": mu,
        "logLik": logLik_new,
        "simMM": simMM,
        "X": X,
        "X_prop": X / np.sum(X),
        "predict_X_prop": np.dot(mu, simMM)
    }

# Example usage
X = np.array([100, 300, 1500, 500, 1000])
# Assuming create_simMat is a function that generates a similarity matrix
# simMM = create_simMat(5, confuse_rate=0.2)
# result = multinom_EM(X, simMM)
# print(result)


def dcats_GLM(count_mat, design_mat, similarity_mat=None, pseudo_count=None, 
              base_model="NULL", fix_phi=None, reference=None):
    # Initialize matrices to store results
    K = count_mat.shape[1]
    num_factors = design_mat.shape[1]
    coeffs = np.full((K, num_factors), np.nan)
    coeffs_err = np.full((K, num_factors), np.nan)
    LR_vals = np.full((K, num_factors), np.nan)
    LRT_pvals = np.full((K, num_factors), np.nan)
    pvals = np.full((K, num_factors), np.nan)
    LRT_fdr = np.full((K, num_factors), np.nan)
    
    count_use = count_mat.copy()
    
    # Apply similarity matrix adjustments if provided
    if similarity_mat is not None:
        for i in range(count_mat.shape[0]):
            count_use[i, :] = np.sum(count_mat[i, :]) * multinom_EM(count_mat[i, :], similarity_mat)
    
    if pseudo_count is None:
        if np.any(np.mean(count_mat, axis=0) == 0):
            print("Empty cell type exists in at least one condition; adding replicate & condition specific pseudo count:")
            count_use += 1
    else:
        count_use += pseudo_count
    
    count_use = np.round(count_use).astype(int)
    
    n_samples = 1
    
    for k in range(num_factors):
        sub_LR_val = np.full((n_samples, K), np.nan)
        sub_coeffs_val = np.full((n_samples, K), np.nan)
        sub_coeffs_err = np.full((n_samples, K), np.nan)
        
        for ir in range(n_samples):
            idx = np.arange(0, count_use.shape[0], n_samples) + ir
            
            for m in range(K):
                if reference is None:
                    df_use = pd.DataFrame({'n1': count_use[:, m], 'total': np.sum(count_use, axis=1)})[idx]
                    df_use['ref_count'] = df_use['total'] - df_use['n1']
                else:
                    if len(reference) == 1:
                        df_use = pd.DataFrame({'n1': count_use[:, m], 'ref_count': count_use[:, reference]})[idx]
                    else:
                        df_use = pd.DataFrame({'n1': count_use[:, m], 'ref_count': np.sum(count_use[:, reference], axis=1)})[idx]
                
                df_use = pd.concat([df_use, design_mat], axis=1)
                df_tmp = df_use[~design_mat.iloc[:, k].isna()]
                
                if base_model == "NULL" or design_mat.shape[1] == 1:
                    formula_fm0 = 'n1 ~ 1'
                    formula_fm1 = f'n1 ~ 1 + {design_mat.columns[k]}'
                elif base_model == "FULL":
                    fm0_right = ' + '.join(design_mat.columns.drop(design_mat.columns[k]))
                    fm1_right = ' + '.join(design_mat.columns)
                    formula_fm0 = f'n1 ~ 1 + {fm0_right}'
                    formula_fm1 = f'n1 ~ 1 + {fm1_right}'
                
                fm0 = ols(formula_fm0, data=df_tmp).fit()
                fm1 = ols(formula_fm1, data=df_tmp).fit()
                
                if fix_phi is not None:
                    fm0 = ols(formula_fm0, data=df_tmp).fit(cov_type='HC0')
                    fm1 = ols(formula_fm1, data=df_tmp).fit(cov_type='HC0')
                
                if len(fm1.params) < 2 or np.isnan(fm1.bse[1]):
                    continue
                
                sub_LR_val[ir, m] = fm0.llf - fm1.llf
                parID = [i for i, name in enumerate(fm1.params.index) if name == design_mat.columns[k]]
                
                if len(parID) > 1:
                    raise ValueError("Please check the design matrix, make sure all factors are continuous or categorical with only two levels.")
                
                sub_coeffs_val[ir, m] = fm1.params[parID[0]]
                if fix_phi is None:
                    sub_coeffs_err[ir, m] = fm1.bse[parID[0]]
        
        coeff_val_mean = np.nanmean(sub_coeffs_val, axis=0)
        
        if fix_phi is None:
            if n_samples == 1 or similarity_mat is None:
                sub_coeff_err_pool = np.nanmean(sub_coeffs_err**2, axis=0)
            else:
                sub_coeff_err_pool = (np.nanmean(sub_coeffs_err**2, axis=0) + 
                                      np.nanstd(sub_coeffs_val, axis=0)**2 + 
                                      np.nanstd(sub_coeffs_val, axis=0)**2 / n_samples)
            
            pvals[:, k] = norm.sf(np.abs(coeff_val_mean) / np.sqrt(sub_coeff_err_pool)) * 2
            coeffs_err[:, k] = np.sqrt(sub_coeff_err_pool)
        
        LR_median = np.nanmedian(sub_LR_val, axis=0)
        LR_vals[:, k] = LR_median
        LRT_pvals[:, k] = chi2.sf(LR_median, df=1)
        coeffs[:, k] = coeff_val_mean
    
    LRT_fdr = np.apply_along_axis(lambda p: np.array([p_value * K / (i+1) for i, p_value in enumerate(np.sort(p))]), 0, LRT_pvals)
    
    res = {
        'coeffs': coeffs,
        'coeffs_err': coeffs_err,
        'LR_vals': LR_vals,
        'LRT_pvals': LRT_pvals,
        'fdr': LRT_fdr
    }
    
    return res

# You need to implement `multinom_EM` function and add necessary import statements for any additional functions or modules.
