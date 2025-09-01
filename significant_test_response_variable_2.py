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

df_clean = df_clean.rename(columns={
    'Total dog facing non managers': 'Total_dog_facing_non_managers',
    'RC Kennel No.':                'RC_Kennel_No', 
    'Total managers': 'Total_managers',
    'days_for_available': 'days_for_available_with_baseline'
})

# define the response varible
y = df_clean['days_for_available_with_baseline'].astype(float)

df_clean['staff_kennel_ratio'] = (
    df_clean['Total_dog_facing_non_managers'] /
    df_clean['RC_Kennel_No']
)

df_clean['breed_type'] = np.where(df_clean['kc_group'] == 'Crossbreed', 'Crossbreed', 'Purebreed').astype(str)

df_clean['on_hold_days'] = (
    df_clean['on_hold_behaviour_days'] +
    df_clean['on_hold_vet_days']
)

X_cat = pd.get_dummies(
    df_clean[['newer_centre',
              'IWB',
              'Dog_age_bracket',
              'sex',
              'size',
              'kc_group',
              'rural_urban_level',
              'breed_type']],
    drop_first=True
)

# Select the continuous variables
continuous = [
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio',
    'Total_managers',
    'age_years',
    'on_hold_days'
]
X_cont = df_clean[continuous].astype(float)
X_base = pd.concat([X_cat, X_cont], axis=1)

picked = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'on_hold_vet_days',
    # 'vet_times',
    # 'behaviour_times',
    'on_hold_behaviour_days'
]

picked_age_years = [
    'newer_centre_Y',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'on_hold_vet_days',
    'on_hold_behaviour_days',
    'age_years'
]

picked_breed = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'breed_type_Purebreed',
    'rural_urban_level_rural',
    'days_for_handover',
    'on_hold_vet_days',
    'on_hold_behaviour_days'
]

picked_on_hold = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'on_hold_days'
]

picked_AIC = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days'
]

picked_urban = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'rural_urban_level_urban',
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days'
]

picked_IWB= [
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
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days'
]

picked_manager = [
    'newer_centre_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Medium',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

picked_adj = [
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
    'kc_group_Pastoral',
    'kc_group_Working',
    'rural_urban_level_rural',
    'rural_urban_level_urban',
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

X = sm.add_constant(X_base[picked], has_constant='add').astype(float)
X_age_years = sm.add_constant(X_base[picked_age_years], has_constant='add').astype(float)
X_breed = sm.add_constant(X_base[picked_breed], has_constant='add').astype(float)
X_on_hold = sm.add_constant(X_base[picked_on_hold], has_constant='add').astype(float)
X_AIC = sm.add_constant(X_base[picked_AIC], has_constant='add').astype(float)
X_urban = sm.add_constant(X_base[picked_urban], has_constant='add').astype(float)
X_IWB = sm.add_constant(X_base[picked_IWB], has_constant='add').astype(float)
X_manager = sm.add_constant(X_base[picked_manager], has_constant='add').astype(float)
X_adj = sm.add_constant(X_base[picked_adj], has_constant='add').astype(float)

model = sm.OLS(y, X).fit()
model_age_years = sm.OLS(y, X_age_years).fit()
model_breed = sm.OLS(y, X_breed).fit()
model_on_hold = sm.OLS(y, X_on_hold).fit()
model_AIC = sm.OLS(y, X_AIC).fit()
model_urban = sm.OLS(y, X_urban).fit()
model_IWB = sm.OLS(y, X_IWB).fit()
model_manager = sm.OLS(y, X_manager).fit()
model_adj = sm.OLS(y, X_adj).fit()

print(model.summary())
print(model_age_years.summary())
print(model_breed.summary())
print(model_on_hold.summary())
print(model_AIC.summary())
print(model_urban.summary())
print(model_IWB.summary())
print(model_manager.summary())
print(model_adj.summary())

F, pval, df = model_AIC.compare_f_test(model)  
print("Partial F test(times): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_urban.compare_f_test(model_AIC)  
print("Partial F test(rural urban level): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_IWB.compare_f_test(model_AIC)  
print("Partial F test(IWB): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_manager.compare_f_test(model_AIC)  
print("Partial F test(manager): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_adj.compare_f_test(model_AIC)  
print("Partial F test(Adjusted R^2): F=%.3f, p=%.4g, df=%s" % (F, pval, df))

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
