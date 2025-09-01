import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Each interaction is test one by one to find the effect on the model 3. This code is used to find the interaction.

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
                   'Total_managers',
                   'days_for_handover']].astype(float)

X_base = pd.concat([X_cat, X_cont], axis=1)

# Cross table -- Check the stability
print(pd.crosstab(df_clean['size'], df_clean['rural_urban_level'], margins=True))
print(pd.crosstab(df_clean['kc_group'], df_clean['rural_urban_level'], margins=True))
print(pd.crosstab(df_clean['on_hold_behaviour_days'], df_clean['kc_group'], margins=True))

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
    'on_hold_behaviour_days'
]

X = sm.add_constant(X_base[picked], has_constant='add').astype(float)

model_base = sm.OLS(y, X).fit()
print("\n=== Base model summary ===")
print(model_base.summary())

# rural * size
# X_rural_size = X.copy()
# X_rural_size['size_Medium_rural'] = X_rural_size['size_Medium'] * X_rural_size['rural_urban_level_rural']
# X_rural_size['size_Small_rural']  = X_rural_size['size_Small']  * X_rural_size['rural_urban_level_rural'] # useful, p-value = 0.05

# rural * kc_group
# X_rural_breed = X.copy()
# X_rural_breed['Gundog_rural'] = X_rural_breed['kc_group_Gundog'] * X_rural_breed['rural_urban_level_rural']
# X_rural_breed['Hound_rural']  = X_rural_breed['kc_group_Hound']  * X_rural_breed['rural_urban_level_rural'] # p-value = 0.08
# X_rural_breed['Terrier_rural'] = X_rural_breed['kc_group_Terrier'] * X_rural_breed['rural_urban_level_rural']
# X_rural_breed['Pastoral_rural']  = X_rural_breed['kc_group_Pastoral']  * X_rural_breed['rural_urban_level_rural']
# X_rural_breed['Working_rural']  = X_rural_breed['kc_group_Working']  * X_rural_breed['rural_urban_level_rural']

# rural * newer_centre
# X_rural_newer = X.copy()
# X_rural_newer['newer_rural'] = X_rural_newer['newer_centre_Y'] * X_rural_newer['rural_urban_level_rural']

# size * kc_group
# X_size_breed = X.copy()
# X_size_breed['Medium_Gundog'] = X_size_breed['kc_group_Gundog'] * X_size_breed['size_Medium']
# X_size_breed['Medium_Hound']  = X_size_breed['kc_group_Hound']  * X_size_breed['size_Medium'] 
# X_size_breed['Medium_Terrier'] = X_size_breed['kc_group_Terrier'] * X_size_breed['size_Medium']
# X_size_breed['Medium_Pastoral']  = X_size_breed['kc_group_Pastoral']  * X_size_breed['size_Medium']
# X_size_breed['Medium_Working']  = X_size_breed['kc_group_Working']  * X_size_breed['size_Medium']
# X_size_breed['Small_Gundog'] = X_size_breed['kc_group_Gundog'] * X_size_breed['size_Small']
# X_size_breed['Small_Hound']  = X_size_breed['kc_group_Hound']  * X_size_breed['size_Small'] # p-value = 0.006
# X_size_breed['Small_Terrier'] = X_size_breed['kc_group_Terrier'] * X_size_breed['size_Small']
# X_size_breed['Small_Pastoral']  = X_size_breed['kc_group_Pastoral']  * X_size_breed['size_Small']
# X_size_breed['Small_Working']  = X_size_breed['kc_group_Working']  * X_size_breed['size_Small']

# age * size
# X_age_size = X.copy()
# X_age_size['size_Medium_senior'] = X_age_size['size_Medium'] * X_age_size['Dog_age_bracket_senior/geriatric (>7 years)'] # p-value = 0.06
# X_age_size['size_Small_senior']  = X_age_size['size_Small']  * X_age_size['Dog_age_bracket_senior/geriatric (>7 years)']
# X_age_size['size_Medium_puppy'] = X_age_size['size_Medium'] * X_age_size['Dog_age_bracket_puppy (0-6 months)']
# X_age_size['size_Small_puppy']  = X_age_size['size_Small']  * X_age_size['Dog_age_bracket_puppy (0-6 months)']
# X_age_size['size_Medium_adult'] = X_age_size['size_Medium'] * X_age_size['Dog_age_bracket_mature adult (2-7 years)']
# X_age_size['size_Small_adult']  = X_age_size['size_Small']  * X_age_size['Dog_age_bracket_mature adult (2-7 years)']


# Final interaction
X_final = X.copy()
X_final['size_Small_rural']  = X_final['size_Small']  * X_final['rural_urban_level_rural']
X_final['Hound_rural']  = X_final['kc_group_Hound']  * X_final['rural_urban_level_rural'] 
X_final['Small_Hound']  = X_final['kc_group_Hound']  * X_final['size_Small']
X_final['size_Medium_senior'] = X_final['size_Medium'] * X_final['Dog_age_bracket_senior/geriatric (>7 years)']


# Build the model
# model_rural_urban_level= sm.OLS(y, X_rural_size).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_rural_urban_level.summary())

# model_rural_breed= sm.OLS(y, X_rural_breed).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_rural_breed.summary())

# model_rural_newer= sm.OLS(y, X_rural_newer).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_rural_newer.summary())

# model_size_breed= sm.OLS(y, X_size_breed).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_size_breed.summary())

# model_age_size= sm.OLS(y, X_age_size).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_age_size.summary())

# model_hold_breed= sm.OLS(y, X_hold_breed).fit()
# print("\n=== Full model (with interactions) summary ===")
# print(model_hold_breed.summary())

model_final= sm.OLS(y, X_final).fit()
print("\n=== Full model (with interactions) summary ===")
print(model_final.summary())

# F-test
# F, pval, df = model_rural_urban_level.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_rural_urban_level.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# F, pval, df = model_rural_breed.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_rural_breed.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# F, pval, df = model_rural_newer.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_rural_newer.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# F, pval, df = model_size_breed.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_size_breed.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# F, pval, df = model_age_size.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_age_size.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# F, pval, df = model_hold_breed.compare_f_test(model_base)  
# print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
# lr_stat, lr_pval, df_diff = model_hold_breed.compare_lr_test(model_base)
# print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

F, pval, df = model_final.compare_f_test(model_base)  
print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
lr_stat, lr_pval, df_diff = model_final.compare_lr_test(model_base)
print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# Calculate the MSE and RMSE
resid = model_final.resid        
n = len(resid)
mse_in_sample = np.mean(resid**2)
rmse_in_sample = np.sqrt(mse_in_sample)
print("In-sample MSE :", mse_in_sample)
print("In-sample RMSE:", rmse_in_sample)

# other check -- remove the interaction between hound and rural
X_final_select = X.copy()
X_final_select['size_Small_rural']  = X_final_select['size_Small']  * X_final_select['rural_urban_level_rural']
X_final_select['Small_Hound']  = X_final_select['kc_group_Hound']  * X_final_select['size_Small']
X_final_select['size_Medium_senior'] = X_final_select['size_Medium'] * X_final_select['Dog_age_bracket_senior/geriatric (>7 years)']

model_final_select= sm.OLS(y, X_final_select).fit()
print("\n=== Full model (with interactions) summary ===")
print(model_final_select.summary())

F, pval, df = model_final.compare_f_test(model_final_select)  
print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
lr_stat, lr_pval, df_diff = model_final.compare_lr_test(model_final_select)
print(f"\nLikelihood ratio test (full vs base): lr_stat={lr_stat:.3f}, p-value={lr_pval:.4f}, df_diff={df_diff}")

# Calculate the MSE and RMSE
resid = model_final_select.resid        
n = len(resid)
mse_in_sample = np.mean(resid**2)
rmse_in_sample = np.sqrt(mse_in_sample)
print("In-sample MSE :", mse_in_sample)
print("In-sample RMSE:", rmse_in_sample)

# check the VIF
X_final_select = sm.add_constant(X_final, has_constant='add')
vif_data = []
for i, col in enumerate(X_final_select.columns):
    if col == 'const':
        continue
    vif_val = variance_inflation_factor(X_final_select.values, i)
    vif_data.append({'variable': col, 'VIF': vif_val})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

vif_df['flag'] = np.where(vif_df['VIF'] > 10, 'HIGH (>10)',
                   np.where(vif_df['VIF'] > 5, 'MODERATE (5-10)', 'OK (<5)'))

print("\nVIF sorted by value:")
print(vif_df.to_string(index=False))
