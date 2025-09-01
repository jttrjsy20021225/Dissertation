import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib.colors import Normalize

# This code is used to do some data exploration.

# Read dataset
df = pd.read_csv("dog_trust_data_cleaned.csv")

df = df.rename(columns={
    'Total dog facing non managers': 'Total_dog_facing_non_managers',
    'RC Kennel No.':                'RC_Kennel_No', 
    'Total managers': 'Total_managers'
})

df['staff_kennel_ratio'] = (
    df['Total_dog_facing_non_managers'] /
    df['RC_Kennel_No']
)

na_counts = df.isna().sum()
print(na_counts)

# Correlation between categorical variables

cats = [
    'newer_centre',
    'IWB',
    'Dog_age_bracket',
    'sex',
    'size',
    'kc_group',
    'rural_urban_level'
]

pval_mat = pd.DataFrame(index=cats, columns=cats, dtype=float)
for v1 in cats:
    for v2 in cats:
        if v1 == v2:
            pval_mat.loc[v1, v2] = 1.0
        else:
            sub = df[[v1, v2]].dropna()
            ct = pd.crosstab(sub[v1], sub[v2])
            _, p, _, _ = chi2_contingency(ct, correction=False)
            pval_mat.loc[v1, v2] = p

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cramer_mat = pd.DataFrame(index=cats, columns=cats, dtype=float)
for v1 in cats:
    for v2 in cats:
        if v1 == v2:
            cramer_mat.loc[v1, v2] = 1.0
        else:
            sub = df[[v1, v2]].dropna()
            cramer_mat.loc[v1, v2] = cramers_v(sub[v1], sub[v2])

plt.figure(figsize=(6, 5))
im1 = plt.imshow(
    pval_mat.astype(float),
    cmap='YlGn_r',          
    interpolation='nearest',
    aspect='auto',
    vmin=0, vmax=1         
)
plt.colorbar(im1, label='p-value')

plt.xticks(range(len(cats)), cats, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(cats)), cats, fontsize=8)

for i in range(len(cats)):
    for j in range(len(cats)):
        v = float(pval_mat.iloc[i, j])
   
        txt = f"{v:.3f}"   
        plt.text(j, i, txt, ha='center', va='center', fontsize=6, color='black')

plt.title("Heatmap of Pairwise Chi-square p-values", fontsize=12)
plt.tight_layout()
plt.savefig("p-value_between_categorical_variables.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(6, 5))
im2 = plt.imshow(cramer_mat.astype(float), cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im2, label="Pearson value")
plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
plt.yticks(range(len(cats)), cats)
for i in range(len(cats)):
    for j in range(len(cats)):
        plt.text(j, i, f"{cramer_mat.iloc[i, j]:.2f}", ha='center', va='center',
                 color='white' if cramer_mat.iloc[i, j] > 0.5 else 'black')
plt.title("Heatmap of Categorical variables")
plt.tight_layout()
plt.savefig("Correlation_between_categorical_variables.png", dpi=300, bbox_inches="tight")

##########################################################################################################

# Correlation between Continuous variables
cols = [
    'days_for_handover',
    'vet_times',
    'on_hold_vet_days',
    'behaviour_times',
    'on_hold_behaviour_days',
    'staff_kennel_ratio',
    'Total_managers'
]


corr_matrix = df[cols].corr(method='pearson')

# Heatmap for continuous variables — match style with the categorical one
plt.figure(figsize=(6, 5))
im3 = plt.imshow(
    corr_matrix.astype(float),
    cmap='coolwarm',
    interpolation='nearest',
    aspect='auto'       # Pearson r ∈ [-1, 1]
)
plt.colorbar(im3, label="Pearson r")  

plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.yticks(range(len(cols)), cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        val = float(corr_matrix.iloc[i, j])
        plt.text(
            j, i, f"{val:.2f}",
            ha='center', va='center',
            color='white' if abs(val) > 0.5 else 'black'
        )

plt.title("Heatmap of Continuous variables")
plt.tight_layout()
plt.savefig("Correlation_between_continuous_variables.png", dpi=300, bbox_inches="tight")

# Boxplot for response variable
fig, ax = plt.subplots(figsize=(12, 4))
df.boxplot(column=['days_to_available'], vert=False, ax=ax)
ax.set_title("Boxplot for days_to_available", fontsize=25, pad=10)
ax.set_xlabel("Days", fontsize=20)
ax.tick_params(axis='x', labelsize=20) 
ax.tick_params(axis='y', labelsize=20) 


s = df['days_to_available'].dropna()
Q1, Q3 = s.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
out = s[(s < lower) | (s > upper)]
print(Q1, Q3)

if not out.empty:
    med = s.median()
    far_idx = (out - med).abs().idxmax()
    x_far = s.loc[far_idx]
    ax.scatter([x_far], [1], color='red', zorder=3)
    ax.annotate(f'{x_far:.0f}', xy=(x_far, 1),
                xytext=(0, 12), textcoords='offset points',
                ha='center', va='bottom', color='red',
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='red'))
else:
    print("No outliers under 1.5×IQR rule.")

plt.tight_layout()
plt.savefig("Boxplot_for_Response_Variables_horizontal.png", dpi=300, bbox_inches="tight")
