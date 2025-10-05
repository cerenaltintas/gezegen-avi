import pandas as pd

df_raw = pd.read_csv('cumulative_2025.10.04_09.55.40.csv', comment='#')
df = df_raw[df_raw['koi_disposition'] == 'CONFIRMED'].copy()

minimal_required = ['koi_period', 'koi_prad']
optional_cols = ['koi_teq', 'koi_srad', 'koi_steff']

# Numerical conversion
for col in minimal_required + optional_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop only rows with NaN in required columns
df = df.dropna(subset=minimal_required)

# Apply defaults for optional columns
if 'koi_teq' not in df.columns or df['koi_teq'].isna().all():
    df['koi_teq'] = 300
else:
    df['koi_teq'] = df['koi_teq'].fillna(df['koi_teq'].median())

if 'koi_srad' not in df.columns or df['koi_srad'].isna().all():
    df['koi_srad'] = 1.0
else:
    df['koi_srad'] = df['koi_srad'].fillna(df['koi_srad'].median())

if 'koi_steff' not in df.columns or df['koi_steff'].isna().all():
    df['koi_steff'] = 5778
else:
    df['koi_steff'] = df['koi_steff'].fillna(df['koi_steff'].median())

# Positive value check
df = df[(df['koi_period'] > 0) & (df['koi_prad'] > 0)]

print(f'Valid planets for visualization: {len(df)}')
print(f'\nSample data:')
print(df[['koi_period', 'koi_prad', 'koi_teq', 'koi_srad', 'koi_steff']].head(5))
print(f'\nData types:')
print(df[['koi_period', 'koi_prad', 'koi_teq']].dtypes)
