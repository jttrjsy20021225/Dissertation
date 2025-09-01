import pandas as pd
import statsmodels.api as sm

# This code is used to define the forward and backward regression by adjust R^2 and AIC.

# Forward adjust R^2
def forward_selection_adjR_2(X, y, predictors, tol=1e-6, max_steps=None):
    included = []

    def fit(vars_):
        if vars_:
            X_ = sm.add_constant(X[vars_], has_constant='add')
        else:
            X_ = sm.add_constant(pd.DataFrame(index=X.index), has_constant='add')
        return sm.OLS(y, X_).fit()

    best_model = fit([])
    best_score = best_model.rsquared_adj

    while True:
        remaining = [p for p in predictors if p not in included]
        if not remaining:
            break

        cand = []
        for p in remaining:
            m = fit(included + [p])
            cand.append((m.rsquared_adj, p, m))

        cand.sort(key=lambda t: t[0], reverse=True)
        new_score, best_p, new_model = cand[0]

        if new_score > best_score + tol:
            included.append(best_p)
            best_score = new_score
            best_model = new_model
        else:
            break

        if max_steps is not None and len(included) >= max_steps:
            break

    return included, best_model


# Backward adjust R^2
def backward_selection_adjR_2(X, y, predictors, sl_remove=0.10):
    included = predictors.copy()

    def fit(vars_):
        return sm.OLS(y, sm.add_constant(X[vars_], has_constant='add')).fit()

    best_model  = fit(included)
    best_vars   = included.copy()
    best_score  = best_model.rsquared_adj

    while True:
        model = fit(included)

        if model.rsquared_adj > best_score + 1e-12:
            best_score = model.rsquared_adj
            best_vars  = included.copy()
            best_model = model

        pvalues = model.pvalues.drop('const', errors='ignore')
        if pvalues.empty:
            break
        worst_pval = pvalues.max()
        if worst_pval > sl_remove:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if not included:
                break
        else:
            break

    return best_vars, best_model


# Farward_AIC
def forward_selection_aic(X, y, predictors, tol=1e-6, max_steps=None):
    included = []

    def fit(vars_):
        if vars_:
            X_ = sm.add_constant(X[vars_], has_constant='add')
        else:
  
            X_ = sm.add_constant(pd.DataFrame(index=X.index), has_constant='add')
        return sm.OLS(y, X_).fit()

    best_model = fit([])
    best_aic   = best_model.aic

    while True:
        remaining = [p for p in predictors if p not in included]
        if not remaining:
            break

        candidates = []
        for p in remaining:
            try:
                m = fit(included + [p])
                candidates.append((m.aic, p, m))
            except Exception:
                continue

        if not candidates:
            break

        new_aic, best_p, new_model = min(candidates, key=lambda t: t[0])

        if new_aic + tol < best_aic:
            included.append(best_p)
            best_aic   = new_aic
            best_model = new_model
        else:
            break
        if max_steps is not None and len(included) >= max_steps:
            break

    return included, best_model


# backward_aic
def backward_selection_aic(X, y, predictors, tol=1e-6):
    included = list(predictors)

    def fit(vars_):
        if vars_: 
            X_ = sm.add_constant(X[vars_], has_constant='add')
        else: 
            X_ = sm.add_constant(pd.DataFrame(index=X.index), has_constant='add')
        return sm.OLS(y, X_).fit()

    best_model = fit(included)
    best_aic   = best_model.aic

    while True:
        candidates = []
        for v in list(included):
            trial = [t for t in included if t != v]
            try:
                m = fit(trial)   
                candidates.append((m.aic, v, m))
            except Exception:
                continue

        if not candidates:
            break

        new_aic, drop_v, new_model = min(candidates, key=lambda t: t[0])

        if new_aic + tol < best_aic:   
            included.remove(drop_v)
            best_aic   = new_aic
            best_model = new_model
        else:
            break

    return included, best_model

