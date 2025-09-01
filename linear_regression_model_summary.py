import pandas as pd
import numpy as np
import statsmodels.api as sm
from Variable_selections import forward_selection_adjR_2, backward_selection_adjR_2, forward_selection_aic, backward_selection_aic

# This code is used to build model 1, model 2 and model 3 before adding the interactions.

# Read the dataset and remove the outliers
df = pd.read_csv("dog_trust_data_cleaned.csv")
max_val = df['days_to_available'].max()
df_clean = df[df['days_to_available'] < max_val].reset_index(drop=True)

df_clean = df_clean.rename(columns={
    'Total dog facing non managers': 'Total_dog_facing_non_managers',
    'RC Kennel No.':                'RC_Kennel_No', 
    'Total managers': 'Total_managers'
})

df_clean['staff_kennel_ratio'] = (
    df_clean['Total_dog_facing_non_managers'] /
    df_clean['RC_Kennel_No']
)

# define the response varible
y1 = df_clean['days_to_available'].astype(float)
assert (y1 >= 0).all()
y2 = np.log1p(y1) 
y3 = df_clean['days_for_available'].astype(float)

# Select the categorical variables
categorical = [
    'newer_centre',
    'IWB',
    'Dog_age_bracket',
    'sex',
    'size',
    'kc_group',
    'rural_urban_level'
]
X_cat_1 = pd.get_dummies(df_clean[categorical], drop_first=True)

# Select the continuous variables
continuous1 = [
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio',
    'Total_managers'
]
X_cont_1 = df_clean[continuous1].astype(float)

continuous3 = [
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio',
    'Total_managers',
    'days_for_handover'
]
X_cont_3 = df_clean[continuous3].astype(float)


X1 = pd.concat([X_cat_1, X_cont_1], axis=1)
X1 = sm.add_constant(X1, has_constant='add')
X1 = X1.astype(float)

X2 = X1

X3 = pd.concat([X_cat_1, X_cont_3], axis=1)
X3 = sm.add_constant(X3, has_constant='add')
X3 = X3.astype(float)

# Fit the model
model1 = sm.OLS(y1, X1).fit()
print('The summary of model 1')
print(model1.summary())

model2 = sm.OLS(y2, X2).fit()
print('The summary of model 2')
print(model2.summary())

model3 = sm.OLS(y3, X3).fit()
print('The summary of model 3')
print(model3.summary())

###################################################Variable selections###########################################################################
# Remove constant intercption
all_predictors_1 = list(X1.columns)
all_predictors_1.remove('const')

all_predictors_2 = list(X2.columns)
all_predictors_2.remove('const')

all_predictors_3 = list(X3.columns)
all_predictors_3.remove('const')
# Stepwise
# Model 1
fwd_vars_adj1, fwd_model_adj1 = forward_selection_adjR_2(X1, y1, all_predictors_1)
print("Forward selection variables with adjust R^2 model 1:", fwd_vars_adj1)
print(fwd_model_adj1.summary())

bwd_vars_adj1, bwd_model_adj1 = backward_selection_adjR_2(X1, y1, all_predictors_1)
print("Backward selection variables with adjust R^2 model 1:", bwd_vars_adj1)
print(bwd_model_adj1.summary())

fwd_vars_aic1, fwd_model_aic1 = forward_selection_aic(X1, y1, all_predictors_1)
print("Forward selection variables with AIC model 1:", fwd_vars_aic1)
print(fwd_model_aic1.summary())

bwd_vars_aic1, bwd_model_aic1 = backward_selection_aic(X1, y1, all_predictors_1)
print("Backward selection variables with AIC model 1:", bwd_vars_aic1)
print(bwd_model_aic1.summary())

# Model 2
fwd_vars_adj2, fwd_model_adj2 = forward_selection_adjR_2(X2, y2, all_predictors_2)
print("Forward selection variables with adjust R^2 model 2:", fwd_vars_adj2)
print(fwd_model_adj2.summary())

bwd_vars_adj2, bwd_model_adj2 = backward_selection_adjR_2(X2, y2, all_predictors_2)
print("Backward selection variables with adjust R^2 model 2:", bwd_vars_adj2)
print(bwd_model_adj2.summary())

fwd_vars_aic2, fwd_model_aic2 = forward_selection_aic(X2, y2, all_predictors_2)
print("Forward selection variables with AIC model 2:", fwd_vars_aic2)
print(fwd_model_aic2.summary())

bwd_vars_aic2, bwd_model_aic2 = backward_selection_aic(X2, y2, all_predictors_2)
print("Backward selection variables with AIC model 2:", bwd_vars_aic2)
print(bwd_model_aic2.summary())

# Model 3
fwd_vars_adj3, fwd_model_adj3 = forward_selection_adjR_2(X3, y3, all_predictors_3)
print("Forward selection variables with adjust R^2 model 3:", fwd_vars_adj3)
print(fwd_model_adj3.summary())

bwd_vars_adj3, bwd_model_adj3 = backward_selection_adjR_2(X3, y3, all_predictors_2)
print("Backward selection variables with adjust R^2 model 3:", bwd_vars_adj3)
print(bwd_model_adj3.summary())

fwd_vars_aic3, fwd_model_aic3 = forward_selection_aic(X3, y3, all_predictors_3)
print("Forward selection variables with AIC model 3:", fwd_vars_aic3)
print(fwd_model_aic3.summary())

bwd_vars_aic3, bwd_model_aic3 = backward_selection_aic(X3, y3, all_predictors_3)
print("Backward selection variables with AIC model 3:", bwd_vars_aic3)
print(bwd_model_aic3.summary())

