import pandas as pd
# This code calculates the frequency and duration of behavioural and veterinary on-hold periods.\

df1 = pd.read_csv('Dog_trust_data.csv')
df2 = pd.read_csv('on_hold_data.csv')

df1['available_date'] = pd.to_datetime(
    df1['available_date'], format='%m/%d/%Y', errors='coerce'
)
df2['from'] = pd.to_datetime(df2['from'], format='%m/%d/%Y', errors='coerce')
df2['to']   = pd.to_datetime(df2['to'],   format='%m/%d/%Y', errors='coerce')
df2['Days_on_hold'] = (df2['to'] - df2['from']).dt.days

df2 = df2.merge(
    df1[['registration_id', 'available_date']],
    on='registration_id',
    how='left'
)

df_vet       = df2[df2['status'] == 'On Hold: Vet'].copy()
df_behaviour = df2[df2['status'] == 'On Hold: Behaviour'].copy()

drop_cols = ['from', 'to', 'available_date']
df_vet.drop(columns=drop_cols, inplace=True)
df_behaviour.drop(columns=drop_cols, inplace=True)

df_vet['vet_times'] = df_vet.groupby('registration_id')['registration_id'].transform('count')
df_behaviour['behaviour_times'] = df_behaviour.groupby('registration_id')['registration_id'].transform('count')

df_vet_agg = df_vet.groupby(['registration_id', 'status'], as_index=False).agg({
    'Days_on_hold': 'sum',
    'vet_times': 'max'
}).rename(columns={'Days_on_hold': 'on_hold_vet_days'})

df_behaviour_agg = df_behaviour.groupby(['registration_id', 'status'], as_index=False).agg({
    'Days_on_hold': 'sum',
    'behaviour_times': 'max'
}).rename(columns={'Days_on_hold': 'on_hold_behaviour_days'})

df_vet_agg.to_csv('on_hold_vet.csv', index=False)
df_behaviour_agg.to_csv('on_hold_behaviour.csv', index=False)
