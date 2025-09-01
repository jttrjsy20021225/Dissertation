import pandas as pd
import numpy as np
import statsmodels.api as sm

# This code is used to do sensitivity test

# Read the dataset and remove the outliers
df = pd.read_csv("dog_trust_data_cleaned.csv")
max_val = df['days_to_available'].max()
df_clean = df[df['days_to_available'] < max_val].reset_index(drop=True)

# define the response varible
y_raw = df_clean['days_for_available'].astype(float)
assert (y_raw >= 0).all()
y = np.log1p(y_raw) 


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

X_cont = df_clean[['days_for_handover',
                   'vet_times',
                   'on_hold_vet_days',
                   'behaviour_times',
                   'on_hold_behaviour_days',
                   'staff_kennel_ratio',
                   'Total_managers']].astype(float)

X_base = pd.concat([X_cat, X_cont], axis=1)

picked= [
    'days_for_handover',
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_puppy (0-6 months)',
    'sex_Male',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Working',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

picked_age= [
    'days_for_handover',
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Working',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

picked_breed= [
    'days_for_handover',
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_puppy (0-6 months)',
    'sex_Male',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Working',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

picked_AIC = [
    'days_for_handover',
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Working',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]


picked_adj = [
    'days_for_handover',
    'newer_centre_Y',
    'IWB_Y',
    'Dog_age_bracket_mature adult (2-7 years)',
    'Dog_age_bracket_puppy (0-6 months)',
    'Dog_age_bracket_senior/geriatric (>7 years)',
    'sex_Male',
    'size_Small',
    'kc_group_Gundog',
    'kc_group_Hound',
    'kc_group_Terrier',
    'kc_group_Working',
    'kc_group_Pastoral',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'Total_managers'
]

X = sm.add_constant(X_base[picked], has_constant='add').astype(float)
X_age = sm.add_constant(X_base[picked_age], has_constant='add').astype(float)
X_breed = sm.add_constant(X_base[picked_breed], has_constant='add').astype(float)
X_AIC = sm.add_constant(X_base[picked_AIC], has_constant='add').astype(float)
X_adj = sm.add_constant(X_base[picked_adj], has_constant='add').astype(float)

model = sm.OLS(y, X).fit()
model_age = sm.OLS(y, X_age).fit()
model_breed = sm.OLS(y, X_breed).fit()
model_AIC = sm.OLS(y, X_AIC).fit()
model_adj = sm.OLS(y, X_adj).fit()

print(model.summary())
print(model_age.summary())
print(model_breed.summary())
print(model_AIC.summary())
print(model_adj.summary())

F, pval, df = model_age.compare_f_test(model)  
print("Partial F test(age): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_breed.compare_f_test(model)  
print("Partial F test(breed): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_AIC.compare_f_test(model)  
print("Partial F test(forward or backward): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
F, pval, df = model_adj.compare_f_test(model_AIC)  
print("Partial F test(AIC or adjusted R^2): F=%.3f, p=%.4g, df=%s" % (F, pval, df))
