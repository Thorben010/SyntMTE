import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Patch

# Set global font
plt.rcParams['font.family'] = ['DejaVu Sans']

# Load predictions from five runs
pred_df1 = pd.read_csv("/home/thor/code/synth_con_pred/logs/20250514-191037/predictions.csv")
pred_df2 = pd.read_csv("/home/thor/code/synth_con_pred/logs/20250514-191040/predictions.csv")
pred_df3 = pd.read_csv("/home/thor/code/synth_con_pred/logs/20250514-191044/predictions.csv")
pred_df4 = pd.read_csv("/home/thor/code/synth_con_pred/logs/20250514-191047/predictions.csv")
pred_df5 = pd.read_csv("/home/thor/code/synth_con_pred/logs/20250514-191050/predictions.csv")
prediction_dfs = [pred_df1, pred_df2, pred_df3, pred_df4, pred_df5]

# Load true values
true_df = pd.read_csv("/home/thor/code/synth_con_pred/data/conditions/inference/test_new.csv")

# --- Helper Functions and Site Map ---
site_map = {
    # predominantly Li-site substitutions
    'Al': 'Li-site',  # well established:contentReference[oaicite:0]{index=0}
    'Ga': 'Li-site',  # ditto:contentReference[oaicite:1]{index=1}
    'Fe': 'Li-site',  # Mössbauer shows Fe³⁺ on 24d Li tetrahedra:contentReference[oaicite:2]{index=2}

    # mostly La-site (large divalents / isovalent REE)
    'Sr': 'La-site',  # XRD & EXAFS put Sr²⁺ on La³⁺ sites:contentReference[oaicite:3]{index=3}
    'Ba': 'La-site',  # Ba²⁺ clearly replaces La³⁺ in co-doped work:contentReference[oaicite:4]{index=4}
    'Nd': 'La-site',  # Li₇Nd₃Zr₂O₁₂ prototypes La→Nd swapping:contentReference[oaicite:5]{index=5}

    # almost exclusively Zr-site supervalent dopants
    'Bi': 'Zr-site',  # Bi³⁺ (and Bi⁵⁺ in air) occupies 16c Zr sites:contentReference[oaicite:6]{index=6}
    'Ti': 'Zr-site',  # Ti⁴⁺, Ge⁴⁺, Sn⁴⁺ … all replace Zr⁴⁺:contentReference[oaicite:7]{index=7}
    'Ta': 'Zr-site',  # Ta⁵⁺ → Zr⁴⁺ gives Li vacancies:contentReference[oaicite:8]{index=8}
    'Nb': 'Zr-site',  # Nb⁵⁺ behaves like Ta⁵⁺:contentReference[oaicite:9]{index=9}
    'W':  'Zr-site',  # W⁶⁺ likewise substitutes Zr:contentReference[oaicite:10]{index=10}
    'Gd': 'Zr-site',  # Gd³⁺ intentionally doped on Zr⁴⁺:contentReference[oaicite:11]{index=11}
    'Y':  'Zr-site'   # Y³⁺–Nb⁵⁺ co-doping literature puts Y on Zr:contentReference[oaicite:12]{index=12}
}


def get_dopant(comp):
    elems = re.findall(r'([A-Z][a-z]?)(?:\d*\.?\d*)', comp)
    dop = [e for e in elems if e not in ('Li','La','Zr','O')]
    return ','.join(sorted(set(dop))) if dop else 'none'

def get_site_type(dopstr):
    if dopstr == 'none': return 'none'
    dopants = dopstr.split(',')
    if len(dopants) > 1: return 'multiple'
    return site_map.get(dopants[0], 'unknown')

def standardize_composition_column(df):
    df['composition'] = df['composition'].astype(str).apply(lambda x: x.strip("[]").replace("'", "").replace('"', ''))
    df['composition'] = df['composition'].str.strip()
    return df

# --- Process Ground Truth Data ---
true_df = standardize_composition_column(true_df.rename(columns={'target_formula': 'composition'}))
#true_df = true_df.drop_duplicates(subset="composition", keep="first").reset_index(drop=True)
true_df['dopant'] = true_df['composition'].apply(get_dopant)
true_df['site_type'] = true_df['dopant'].apply(get_site_type)

# Filter out co-doped systems (those with multiple dopants)
true_df = true_df[~true_df['dopant'].str.contains(',')]

# Manual site type corrections from original script
true_df.loc[true_df['dopant'] == 'Gd', 'site_type'] = 'La-site'
true_df.loc[true_df['dopant'] == 'Bi', 'site_type'] = 'Zr-site'
true_df.loc[true_df['dopant'] == 'Fe', 'site_type'] = 'La-site'
true_df.loc[true_df['dopant'].isin(['Al', 'Ga', 'Mg']), 'site_type'] = 'Li-site'

true_df_filtered = true_df[~true_df['site_type'].isin(['None', 'none', 'unknown'])].copy()

true_stats_df = (
    true_df_filtered.groupby(['site_type', 'dopant'])
    .agg(
        true_mean = pd.NamedAgg(column='Sintering Temperature', aggfunc='mean'),
        true_std = pd.NamedAgg(column='Sintering Temperature', aggfunc='std')
    )
    .reset_index()
)
true_stats_df['true_std'] = true_stats_df['true_std'].fillna(0)


# --- Process Prediction Runs ---
# Use processed true_df for consistent dopant/site info and to filter predictions
valid_compositions_info = true_df_filtered[['composition', 'dopant', 'site_type']].drop_duplicates()

all_run_dopant_means = []
for run_df in prediction_dfs:
    run_df = standardize_composition_column(run_df.copy())
    # Merge with valid_compositions_info to get consistent dopant/site_type and filter
    merged_run_df = pd.merge(run_df, valid_compositions_info, on='composition', how='inner')
    
    run_dopant_mean_df = (
        merged_run_df.groupby(['site_type', 'dopant'])
        .agg(run_pred_mean = pd.NamedAgg(column='pred_sint_temp', aggfunc='mean'))
        .reset_index()
    )
    all_run_dopant_means.append(run_dopant_mean_df)

combined_run_means_df = pd.concat(all_run_dopant_means)

pred_stats_df = (
    combined_run_means_df.groupby(['site_type', 'dopant'])
    .agg(
        pred_mean = pd.NamedAgg(column='run_pred_mean', aggfunc='mean'), # Mean of the 5 run_pred_means
        pred_std = pd.NamedAgg(column='run_pred_mean', aggfunc='std')   # Std of the 5 run_pred_means
    )
    .reset_index()
)
pred_stats_df['pred_std'] = pred_stats_df['pred_std'].fillna(0)

# --- Final Merge for Plotting ---
grp = pd.merge(true_stats_df, pred_stats_df, on=['site_type', 'dopant'], how='inner') # Inner join to keep only groups present in both

# Define the order of site types and sort
site_order = ['Li-site', 'La-site', 'Zr-site'] # Reordered to put A-site first, then B, then C
grp['site_type'] = pd.Categorical(grp['site_type'], categories=site_order, ordered=True)
grp = grp.dropna(subset=['site_type']) # Drop rows where site_type became NaN due to not being in the refined site_order
grp = grp.sort_values(['site_type', 'true_mean'], ascending=[True, False]).reset_index(drop=True)


# Prepare shading colors for each site block
shade_colors = {
    'Li-site':    '#e0ecf4',  # A-site
    'La-site':    '#e5f5e0',  # B-site
    'Zr-site':    '#fde0dd',  # C-site
    # 'unknown':    '#f6eff7'  # No longer present after filtering
}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8)) 
y_pos = np.arange(len(grp))

# Shade each site_type block
for site_val in site_order: 
    sub = grp[grp.site_type == site_val]
    if sub.empty: 
        continue
    i0, i1 = sub.index.min(), sub.index.max()
    ax.axhspan(i0 - 0.5, i1 + 0.5, color=shade_colors.get(site_val, '#f0f0f0'), alpha=0.4, zorder=0)

# Draw lines and error bars for each dopant group
for i, row in grp.iterrows():
    # connecting line - REMOVED
    # ax.hlines(y=i,
    #           xmin=row['true_mean'],
    #           xmax=row['pred_mean'],
    #           color='gray', alpha=0.6, linewidth=1, zorder=1)
    # True
    ax.errorbar(
        row['true_mean'], i,
        xerr=row['true_std'] * 2 if pd.notna(row['true_std']) else 0,
        fmt='o', color='C0', capsize=4,
        label='True (Dopant Avg. ± Std)' if i == 0 else "", zorder=2
    )
    
    # Predicted (symmetric error bar using std of run means)
    ax.errorbar(
        row['pred_mean'], i,
        xerr=row['pred_std'] * 2 if pd.notna(row['pred_std']) else 0, # Symmetric error
        fmt='s', color='C1', capsize=4,
        label='Predicted (Avg. of Run Means ± Std of Run Means)' if i == 0 else "", zorder=2
    )

# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(grp['dopant'])
ax.set_ylabel("Dopant", fontsize=17)
ax.set_xlabel("Sintering Temperature (°C)", fontsize=17)

# Add legends
ax.legend(loc='lower right')

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=14)

# Add site-type legend patches
patches = []
# unique_sites_in_plot = sorted(grp['site_type'].unique(), key=lambda x: site_order.index(x) if x in site_order else -1)
# Use site_order directly as we've filtered grp by it
for s in site_order: 
    if s in grp['site_type'].values: # Check if site type is actually in the data after all filtering
        label = str(s) 
        if s == 'Li-site': label = 'Li-site (A site)'
        elif s == 'Zr-site': label = 'Zr-site (C site)'
        elif s == 'La-site': label = 'La-site (B site)'
        elif s == 'multiple': label = 'Multiple Dopants' # Added label for multiple
        patches.append(Patch(facecolor=shade_colors[s], edgecolor='none', alpha=0.5, label=label))

if patches: # Only add legend if there are patches to show
    ax.add_artist(
        ax.legend(handles=patches, title="Site Type",
                  bbox_to_anchor=(1.02, 1), loc='upper left') 
    )

plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.savefig("figures/case_study_plot_dopant_agg_run_means_std.png", dpi=500)
plt.show()
print(f"Saved figure to figures/case_study_plot_dopant_agg_run_means_std.png")

# Removed old processing logic:
# true_df['composition'] = true_df['target_formula'].apply(lambda x: x.strip("[]").replace("'", "").replace('"', ''))
# true_df['composition'] = true_df['composition'].str.strip()
# for df in [pred_df1, pred_df2, pred_df3, pred_df4, pred_df5]:
#     df['composition'] = df['composition'].str.strip()
# merged_data = pd.merge(true_df, all_individual_preds, on="composition", how="inner")
# merged_data['dopant'] = merged_data['composition'].apply(get_dopant)
# merged_data['site_type'] = merged_data['dopant'].apply(get_site_type)
# merged_data = merged_data[~merged_data['site_type'].isin(['None', 'none'])]
# merged_data.loc[merged_data['dopant'] == 'Gd', 'site_type'] = 'La-site'
# merged_data.loc[merged_data['dopant'] == 'Bi', 'site_type'] = 'La-site'
# merged_data.loc[merged_data['dopant'] == 'Fe', 'site_type'] = 'La-site'
# merged_data.loc[merged_data['dopant'].isin(['Al', 'Ga', 'Mg']), 'site_type'] = 'Li-site'
# stoichiometry_summary_df = merged_data.groupby('composition').agg(
#     true_sint_temp=('Sintering Temperature', 'first'),
#     avg_pred_sint_temp=('pred_sint_temp', 'mean'),
#     std_pred_sint_temp_across_runs=('pred_sint_temp', 'std'),
#     dopant=('dopant', 'first'),
#     site_type=('site_type', 'first')
# ).reset_index()
# stoichiometry_summary_df['std_pred_sint_temp_across_runs'] = stoichiometry_summary_df['std_pred_sint_temp_across_runs'].fillna(0)
# stoichiometry_summary_df['pred_lower_bound'] = stoichiometry_summary_df['avg_pred_sint_temp'] - stoichiometry_summary_df['std_pred_sint_temp_across_runs']
# stoichiometry_summary_df['pred_upper_bound'] = stoichiometry_summary_df['avg_pred_sint_temp'] + stoichiometry_summary_df['std_pred_sint_temp_across_runs']
# print(stoichiometry_summary_df.dopant)
# grp = (
#     stoichiometry_summary_df
#     .groupby(['site_type', 'dopant'])
#     .agg(
#         true_mean = pd.NamedAgg(column='true_sint_temp', aggfunc='mean'),
#         true_std  = pd.NamedAgg(column='true_sint_temp', aggfunc='std'),
#         pred_mean = pd.NamedAgg(column='avg_pred_sint_temp', aggfunc='mean'),
#         pred_min_overall = pd.NamedAgg(column='pred_lower_bound', aggfunc='min'),
#         pred_max_overall = pd.NamedAgg(column='pred_upper_bound', aggfunc='max')
#     )
#     .reset_index()
# )
# grp['true_std'] = grp['true_std'].fillna(0)
