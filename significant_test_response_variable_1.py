import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

# This code tests the results under different stepwise regression and includes some specific variables.

# Read the dataset and remove the outliers
df = pd.read_csv("dog_trust_data_cleaned.csv")
max_val = df['days_to_available'].max()
df_clean = df[df['days_to_available'] < max_val].reset_index(drop=True)

# define the response varible
y = df_clean['days_to_available'].astype(float)


df_clean = df_clean.rename(columns={
    'Total dog facing non managers': 'Total_dog_facing_non_managers',
    'RC Kennel No.':                'RC_Kennel_No', 
    'Total managers': 'Total_managers'
})

df_clean['staff_kennel_ratio'] = (
    df_clean['Total_dog_facing_non_managers'] /
    df_clean['RC_Kennel_No']
)


X_cat = pd.get_dummies(
    df_clean[['newer_centre',
              'IWB',
              'Dog_age_bracket',
              'sex',
              'size',
              'kc_group',
              'rural_urban_level']],
    drop_first=True
)

X_cont = df_clean[['vet_times',
                   'on_hold_vet_days',
                   'behaviour_times',
                   'on_hold_behaviour_days',
                   'staff_kennel_ratio',
                   'Total_managers']].astype(float)

X_base = pd.concat([X_cat, X_cont], axis=1)

picked = [
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Utility',
    'kc_group_Working',
    'on_hold_vet_days',
    'on_hold_behaviour_days',
]

picked_AIC = [
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Utility',
    'kc_group_Working',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
]

picked_rural = [
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Utility',
    'kc_group_Working',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'rural_urban_level_rural'
]

picked_staff_kennel_ratio = [
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Utility',
    'kc_group_Working',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio'
]

picked_both = [
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Utility',
    'kc_group_Working',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio',
    'rural_urban_level_rural'
]

X = sm.add_constant(X_base[picked], has_constant='add').astype(float)
X_AIC = sm.add_constant(X_base[picked_AIC], has_constant='add').astype(float)
X_rural = sm.add_constant(X_base[picked_rural], has_constant='add').astype(float)
X_staff_kennel_ratio = sm.add_constant(X_base[picked_staff_kennel_ratio], has_constant='add').astype(float)
X_both = sm.add_constant(X_base[picked_both], has_constant='add').astype(float)

model = sm.OLS(y, X).fit()
model_AIC = sm.OLS(y, X_AIC).fit()
model_rural = sm.OLS(y, X_rural).fit()
model_staff_kennel_ratio = sm.OLS(y, X_staff_kennel_ratio).fit()
model_both = sm.OLS(y, X_both).fit()

print(model.summary())
print(model_AIC.summary())
print(model_rural.summary())
print(model_staff_kennel_ratio.summary())
print(model_both.summary())

F, pval, df = model_AIC.compare_f_test(model)  
print("Partial F test(both): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_rural.compare_f_test(model_AIC)  
print("Partial F test(rural urban level): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_staff_kennel_ratio.compare_f_test(model_AIC)  
print("Partial F test(staff_kennel_ratio): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_both.compare_f_test(model_AIC)  
print("Partial F test(both): F=%.3f, p=%.4g, df=%s" % (F, pval, df))

# Calculate the MSE and RMSE
resid = model_AIC.resid        
n = len(resid)
mse_in_sample = np.mean(resid**2)
rmse_in_sample = np.sqrt(mse_in_sample)
print("In-sample MSE :", mse_in_sample)
print("In-sample RMSE:", rmse_in_sample)

# residual plot
fitted = model_AIC.fittedvalues
resid  = model_AIC.resid

plt.figure(figsize=(6, 4))
plt.scatter(fitted, resid,color='tab:green', alpha=0.2)
mask = np.isfinite(fitted) & np.isfinite(resid)
smooth = lowess(resid[mask], fitted[mask], frac=0.25, it=1, return_sorted=True)
plt.plot(smooth[:,0], smooth[:,1], linewidth=2, label='Lowess smooth')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.legend()
plt.tight_layout()
plt.savefig("residual_plot_model_2.png", dpi=300, bbox_inches="tight")

# QQ-plot
fig = qqplot(model_AIC.resid, line='45', fit=True)
plt.title("QQ plot of residuals (Normal)")
plt.tight_layout()
fig.savefig("qqplot_residuals_2.png", dpi=300, bbox_inches="tight")

# Calculate the MSE and RMSE
resid = model_AIC.resid        
n = len(resid)
mse_in_sample = np.mean(resid**2)
rmse_in_sample = np.sqrt(mse_in_sample)
print("In-sample MSE :", mse_in_sample)
print("In-sample RMSE:", rmse_in_sample)