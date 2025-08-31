import pandas as pd


# This code combine all dataset
# Read datasets
df1 = pd.read_csv('Dog_trust_data_updated_breed.csv')
df2 = pd.read_csv('on_hold_vet.csv')
df3 = pd.read_csv('on_hold_behaviour.csv')
df4 = pd.read_csv('unclear_breeds.csv')
df5 = pd.read_csv('staff_and_kennels_average.csv')

# Combine vet on hold informatiom
df1 = df1.merge(df2[['registration_id', 'vet_times']], 
                on='registration_id', 
                how='left')
df1 = df1.merge(df2[['registration_id', 'on_hold_vet_days']], 
                on='registration_id', 
                how='left')
df1['vet_times'] = df1['vet_times'].fillna(0)
df1['on_hold_vet_days'] = df1['on_hold_vet_days'].fillna(0)

# Combine behaviour on hold informatiom
df1 = df1.merge(df3[['registration_id', 'behaviour_times']], 
                on='registration_id', 
                how='left')
df1 = df1.merge(df3[['registration_id', 'on_hold_behaviour_days']], 
                on='registration_id', 
                how='left')
df1['behaviour_times'] = df1['behaviour_times'].fillna(0)
df1['on_hold_behaviour_days'] = df1['on_hold_behaviour_days'].fillna(0)

# Combine unclear breeds informatiom
df1 = df1.merge(df4[['registration_id', 'kc_group']], 
                on='registration_id', 
                how='left', 
                suffixes=('', '_df4'))
mask = df1['kc_group_df4'].notna()
df1.loc[mask, 'kc_group'] = df1.loc[mask, 'kc_group_df4']
df1.drop(columns='kc_group_df4', inplace=True)


# Build the comparsions
centre_map = {
    'Ballymena Rehoming Centre':   'Ballymena',
    'Basildon Rehoming Centre':    'Basildon',
    'Bridgend Rehoming Centre':    'Bridgend',
    'Canterbury Rehoming Centre':  'Cantebury',
    'Cardiff Rehoming Centre':     'Cardiff',
    'Darlington Rehoming Centre':  'Darlington',
    'Evesham Rehoming Centre':     'Evesham',
    'Glasgow Rehoming Centre':     'Glasgow',
    'Harefield Rehoming Centre':   'Harefield',
    'Ilfracombe Rehoming Centre':  'Ilfracombe',
    'Kenilworth Rehoming Centre':  'Kenilworth',
    'Leeds Rehoming Centre':       'Leeds',
    'Loughborough Rehoming Centre':'Loughborough',
    'Manchester Rehoming Centre':  'Manchester',
    'Merseyside Rehoming Centre':  'Merseyside',
    'Newbury Rehoming Centre':     'Newbury',
    'Salisbury Rehoming Centre':   'Salisbury',
    'Shoreham Rehoming Centre':    'Shoreham',
    'Shrewsbury Rehoming Centre':  'Shrewsbury',
    'Snetterton Rehoming Centre':  'Snetterton',
    'West Calder Rehoming Centre': 'West Calder'
}


df1['Centre'] = df1['rehoming_centre'].map(centre_map)

rural_urban_map = {
    'Ballymena Rehoming Centre':   'rural',
    'Basildon Rehoming Centre':    'urban',
    'Bridgend Rehoming Centre':    'intermediate',
    'Canterbury Rehoming Centre':  'intermediate',
    'Cardiff Rehoming Centre':     'urban',
    'Darlington Rehoming Centre':  'urban',
    'Evesham Rehoming Centre':     'intermediate',
    'Glasgow Rehoming Centre':     'urban',
    'Harefield Rehoming Centre':   'urban',
    'Ilfracombe Rehoming Centre':  'rural',
    'Kenilworth Rehoming Centre':  'urban',
    'Leeds Rehoming Centre':       'urban',
    'Loughborough Rehoming Centre':'intermediate',
    'Manchester Rehoming Centre':  'urban',
    'Merseyside Rehoming Centre':  'urban',
    'Newbury Rehoming Centre':     'intermediate',
    'Salisbury Rehoming Centre':   'rural',
    'Shoreham Rehoming Centre':    'urban',
    'Shrewsbury Rehoming Centre':  'intermediate',
    'Snetterton Rehoming Centre':  'rural',
    'West Calder Rehoming Centre': 'intermediate'
}

df1['rural_urban_level'] = df1['rehoming_centre'].map(rural_urban_map).fillna('unknown')

cols_to_pull = [
    'Centre',
    'Total dog facing non managers',
    'Total number of dog staff',
    'Total managers',
    'RC Kennel No.'
]
df1 = df1.merge(
    df5[cols_to_pull],
    on='Centre',
    how='left'
)
df1.drop(columns='Centre', inplace=True)

# Fill the NA values in 'newer_centre'
df1['newer_centre'] = df1['newer_centre'].fillna('N')

# adjust size
df1['size'] = df1['size'].replace({'Giant': 'Large'})

# Save the file
df1 = df1.drop_duplicates()
print(len(df1))
df1.to_csv('dog_trust_data_combination.csv', index=False)


# check and delect NA values
na_counts = df1.isna().sum()
print(na_counts)
df_cleaned = df1.dropna(how='any')
df_cleaned = df_cleaned.drop(columns=[
    'dog_date_of_birth', 'breed', 'breed_cleaned',
    'handover_date', 'available_date',
    'vet_hold_period', 'behaviour_hold_period',
    'baseline_data'
])
print(len(df_cleaned))
# Save the file
df_cleaned.to_csv('dog_trust_data_cleaned.csv', index=False)
