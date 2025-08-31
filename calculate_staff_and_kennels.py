import pandas as pd

# Calculate the average value of kennels, staff and managers.
df = pd.read_excel('staff_and_kennels.xlsx')
df.columns = df.columns.str.strip()

cols_to_average = [
    'Total dog facing non managers',
    'Total number of dog staff',
    'Total managers',
    'RC Kennel No.'
]

df_avg = df.groupby('Centre', as_index=False)[cols_to_average].mean()

df_avg.to_csv('staff_and_kennels_average.csv', index=False)
