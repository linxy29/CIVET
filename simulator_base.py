## This file is still under testing!

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit
from scipy.special import logit as scipy_logit

def perform_multi_level_statistical_tests(df_use, use_random_effect):
    # Convert cell_label to categorical (factor in R)
    df_use['cell_label'] = pd.Categorical(df_use['cell_label'])

    # Define the formula for the null model
    formula_fm0 = "n1 / (n1 + ref_count) ~ 1"
    
    try:
        # Fit the null model using GLM (Generalized Linear Model) with a binomial family
        fm0 = sm.GLM.from_formula(formula_fm0, data=df_use, family=sm.families.Binomial()).fit()
    except Exception as e:
        print(f"Error fitting the null model: {e}")
        return None
    
    if not use_random_effect:
        # Define the formula for the alternative model
        formula_fm1 = "n1 / (n1 + ref_count) ~ 1 + cell_label"
        try:
            # Attempt to fit the alternative model
            fm1 = sm.GLM.from_formula(formula_fm1, data=df_use, family=sm.families.Binomial()).fit()
        except Exception as e:
            print(f"Error fitting the alternative model: {e}")
            fm1 = None
    else:
        # Random effects require mixed models, which is a more complex implementation
        # Since statsmodels does not support random effects in GLM directly, we would need
        # to use a package like `statsmodels.mixedlm` or `pymer4`, but for simplicity:
        fm1 = None
        try:
            # Example of mixed model fitting, though not identical to R's betabin function
            formula_random = "n1 / (n1 + ref_count) ~ 1"
            # A real implementation would require a package supporting beta-binomial with random effects
            # fm1 = MixedLM.from_formula(formula_random, data=df_use, groups=df_use['cell_label']).fit()
        except Exception as e:
            print(f"Error fitting the random effects model: {e}")
            fm1 = None

    # Extract p-values
    if fm1 is None:
        pvals = {label: np.nan for label in df_use['cell_label'].cat.categories}
    else:
        try:
            # Extract p-values from the model
            pvals = fm1.pvalues[1:]  # Skip the intercept
            pvals = pvals.to_dict()
        except Exception as e:
            print(f"Error extracting p-values: {e}")
            pvals = {label: np.nan for label in df_use['cell_label'].cat.categories}
    
    return pvals

# Example usage
df_use = pd.DataFrame({
    'n1': np.random.randint(0, 100, size=50),
    'ref_count': np.random.randint(0, 100, size=50),
    'cell_label': np.random.choice(['Label1', 'Label2', 'Label3'], size=50)
})

use_random_effect = False
pvals = perform_multi_level_statistical_tests(df_use, use_random_effect)
print(pvals)
