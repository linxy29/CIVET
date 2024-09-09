## This file is still under testing!

import numpy as np
import pandas as pd
from scipy.special import betainc, gammaln
from scipy.optimize import minimize
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit, cloglog

def invlink(eta, link_type):
    if link_type == "logit":
        return 1 / (1 + np.exp(-eta))
    elif link_type == "cloglog":
        return 1 - np.exp(-np.exp(eta))
    else:
        raise ValueError(f"Unknown link function: {link_type}")

def minuslogL(param, modmatrix_b, modmatrix_phi, y, n, link, fixpar):
    if fixpar:
        param[fixpar[0]] = fixpar[1]
    
    b = param[:modmatrix_b.shape[1]]
    eta = np.dot(modmatrix_b, b)
    p = invlink(eta, link)
    
    phi = np.dot(modmatrix_phi, param[modmatrix_b.shape[1]:])
    cnd = phi == 0
    
    f1 = np.sum(np.log(np.power(p[cnd], y[cnd]) * np.power(1 - p[cnd], n[cnd] - y[cnd])))
    
    n2 = n[~cnd]
    y2 = y[~cnd]
    p2 = p[~cnd]
    phi2 = phi[~cnd]
    
    lbeta_term1 = np.exp(gammaln(p2 * (1 - phi2) / phi2 + y2) - gammaln(p2 * (1 - phi2) / phi2))
    lbeta_term2 = np.exp(gammaln((1 - p2) * (1 - phi2) / phi2 + n2 - y2) - gammaln((1 - p2) * (1 - phi2) / phi2))
    
    f2 = np.sum(gammaln(n2 + 1) - gammaln(y2 + 1) - gammaln(n2 - y2 + 1) + np.log(lbeta_term1 * lbeta_term2))
    
    fn = f1 + f2
    if not np.isfinite(fn):
        fn = -1e20
    return -fn

def glimML(formula, random, data=None, link="logit", phi_ini=None, warnings=False, na_action='omit', fixpar=None, hessian=True, control={'maxit': 2000}, **kwargs):
    if data is None:
        raise ValueError("Data must be provided.")
    
    if link == "logit":
        link_func = logit()
    elif link == "cloglog":
        link_func = cloglog()
    else:
        raise ValueError("Link function must be either 'logit' or 'cloglog'")
    
    if not isinstance(formula[0], tuple) or formula[0][0] != "cbind":
        raise ValueError("The formula is not valid. The response must be a matrix of the form cbind(success, failure).")
    
    if len(random) == 3:
        print(f"Warning: The formula for phi ({random}) contains a response which is ignored.")
        random = random[1:]
    
    explain = random[1:] if len(random) > 1 else None
    if len(explain) > 1:
        print(f"Warning: The formula for phi contains several explanatory variables ({explain}). Only the first one ({explain[0]}) was considered.")
        explain = explain[0]
    
    gf3 = f"{formula[2]} + {explain}" if explain else formula[2]
    gf = f"{formula[1]} ~ {gf3}"
    
    y = data[list(formula[0][1:3])].values
    data['response'] = y[:, 0] / y.sum(axis=1)
    
    model = glm(formula=gf, family=Binomial(link=link_func), data=data)
    fit = model.fit()
    
    b = fit.params.values
    if any(np.isnan(b)):
        raise ValueError(f"Initial values for the fixed effects contain at least one missing value: {b[np.isnan(b)]}")
    
    modmatrix_b = fit.model.exog
    modmatrix_phi = data[explain].values.reshape(-1, 1) if explain else np.ones((data.shape[0], 1))
    
    if phi_ini is None:
        phi_ini = np.full(modmatrix_phi.shape[1], 0.1)
    
    param_ini = np.concatenate([b, phi_ini])
    if fixpar:
        param_ini[fixpar[0]] = fixpar[1]
    
    result = minimize(minuslogL, param_ini, args=(modmatrix_b, modmatrix_phi, y[:, 0], y.sum(axis=1), link, fixpar), 
                      method='BFGS', options=control, **kwargs)
    
    if warnings and not result.success:
        print(f"Warning: {result.message}")
    
    return {
        "params": result.x,
        "logL": -result.fun,
        "iterations": result.nit,
        "convergence": result.success,
        "message": result.message
    }

# Test the function with a simple example
data = pd.DataFrame({
    'y1': [1, 0, 1, 0, 1],
    'y2': [0, 1, 0, 1, 0],
    'x': [1, 2, 3, 4, 5]
})

result = glimML(formula=(("cbind", "y1", "y2"), "~", "x"), random=("~", "x"), data=data)
print(result)
