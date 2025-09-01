import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

# Mann–Whitney U test for IWB

# Read the dataset and remove the outliers
df = pd.read_csv("dog_trust_data_cleaned.csv")
targets = [
    "Snetterton Rehoming Centre",
    "Cardiff Rehoming Centre",
    "Manchester Rehoming Centre",
]
df_sel = df[df["rehoming_centre"].isin(targets)].copy()

# The time when they introduced IWB
cond = (
    ((df_sel['rehoming_centre'] == 'Snetterton Rehoming Centre') & (df_sel['days_for_handover'] > 1729)) |
    ((df_sel['rehoming_centre'] == 'Manchester Rehoming Centre') & (df_sel['days_for_handover'] > 1973)) |
    ((df_sel['rehoming_centre'] == 'Cardiff Rehoming Centre') & (df_sel['days_for_handover'] > 1882))
)

df_sel['ad_IWB'] = np.where(cond, 'Y', 'N')

print(df_sel.groupby(['rehoming_centre', 'ad_IWB']).size())


df_Y = df_sel[df_sel['ad_IWB'] == 'Y'].copy()
df_N = df_sel[df_sel['ad_IWB'] == 'N'].copy()
Y_Y = df_Y["days_to_available"].astype(float)
Y_N = df_N["days_to_available"].astype(float)

def normality_tests(arr, label):

    # Shapiro–Wilk
    sw_stat, sw_p = stats.shapiro(arr)
    # Lilliefors
    lf_stat, lf_p = lilliefors(arr, dist='norm')

    print(f"  Shapiro–Wilk: W = {sw_stat:.4f}, p = {sw_p:.4g}")
    print(f"  Kolmogorov–Smirnov (Lilliefors): D = {lf_stat:.4f}, p = {lf_p:.4g}")
    return (sw_p >= 0.05) and (lf_p >= 0.05)

is_normal_Y = normality_tests(Y_Y, "IWB = Y") #  Shapiro–Wilk: W = 0.6975, p = 4.515e-26   Kolmogorov–Smirnov (Lilliefors): D = 0.1874, p = 0.001
is_normal_N = normality_tests(Y_N, "IWB = N")  #  Shapiro–Wilk: W = 0.5222, p = 1.575e-35  Kolmogorov–Smirnov (Lilliefors): D = 0.2717, p = 0.001

u_stat, u_p = stats.mannwhitneyu(Y_Y, Y_N, alternative='two-sided')
print("\n[Mann–Whitney U test]")
print(f"  U = {u_stat:.0f}, p = {u_p:.4g}")
print(f"  medium: IWB=Y -> {np.median(Y_Y):.2f},  IWB=N -> {np.median(Y_N):.2f}")
# medium: IWB=Y -> 14.50,  IWB=N -> 25.00
# Significant different 

