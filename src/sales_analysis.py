# Change to your own file path
df = pd.read_csv(r"C:\Users\PotatoOvo\Desktop\INFS2822\store.csv")
train = pd.read_csv(r"C:\Users\PotatoOvo\Desktop\INFS2822\train.csv", low_memory=False)
# Counting null values in each column
null_counts = df.isnull().sum()
print(null_counts)

# Count the number of zeros in the Promo2 column
zero_count = (df['Promo2'] == 0).sum()
print(f"The number of zeros in the Promo2 column is: {zero_count}")

# Key findings: Null values are the same in promo2sinceweek, year, and interval columns, and 0 value in Promo2 column.
# So it's fair that there is no promo for those columns. Also for competitionopensincemonth and sinceyear.

# Fill missing values in 'Promo2SinceWeek' and 'Promo2SinceYear' columns with 0, PromoInterval with 'None'
df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
df['PromoInterval'] = df['PromoInterval'].fillna('None')

# 'PromoInterval' column includes strings, so we can't fill it with 0 like above.

# Drop rows with null values in 'CompetitionDistance' column because only 3 rows have null values. Every store should have a competitor.
df = df.dropna(subset=['CompetitionDistance'])

# fill missing 'CompetitionOpenSinceYear' with group median by 'StoreType'
df['CompetitionOpenSinceYear'] = (
    df.groupby('StoreType')['CompetitionOpenSinceYear']
    .transform(lambda x: x.fillna(x.median()))
)

# fill missing 'CompetitionOpenSinceMonth' with group median by 'StoreType'
df['CompetitionOpenSinceMonth'] = (
    df.groupby('StoreType')['CompetitionOpenSinceMonth']
    .transform(lambda x: x.fillna(x.median()))
)

# Feature Engineering 1: Duration since a competitor’s store opened (in months)
def competition_duration(row):
    return max((2025 - row['CompetitionOpenSinceYear']) * 12 + (4 - row['CompetitionOpenSinceMonth']), 0)

df['CompetitionOpenDurationMonths'] = df.apply(competition_duration, axis=1)

# Feature Engineering 2: How long a Promo2 campaign has been running (in weeks)
# Only apply if Promo2 is 1 and year/week are not 0
def promo_duration(row):
    if row['Promo2'] == 0 or row['Promo2SinceYear'] == 0 or row['Promo2SinceWeek'] == 0:
        return 0
    # Approximate week number for April 2025
    current_year = 2025
    current_week = 14
    return max((current_year - row['Promo2SinceYear']) * 52 + (current_week - row['Promo2SinceWeek']), 0)

df['PromoActiveDurationWeeks'] = df.apply(promo_duration, axis=1)

print(df[['Store', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'PromoActiveDurationWeeks',
          'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'CompetitionOpenDurationMonths']].head())

# Descriptive Data:
print(df.describe())

# Merge datasets on Store
df = pd.merge(train, df, on='Store')

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Drop closed stores to focus on actual sales
df = df[df['Open'] == 1]

from matplotlib.ticker import FuncFormatter

# ==============================
# 🟢 Q1: Total Monthly Sales Over Time (with y-axis in Millions)
# ==============================
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)

plt.figure(figsize=(14,6))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')

# Format y-axis to show millions
formatter = FuncFormatter(lambda x, _: f'{x/1e6:.1f}M')
plt.gca().yaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)
plt.title('Q1: Total Monthly Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales (Millions)')
plt.grid(True)
plt.tight_layout()
plt.show()
# ==============================
# 🟢 Q2: Avg Sales by StoreType
# ==============================
avg_sales_storetype = df.groupby('StoreType')['Sales'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=avg_sales_storetype, x='StoreType', y='Sales', palette='Blues_d')
for i, row in avg_sales_storetype.iterrows():
    plt.text(i, row['Sales'] + 200, f"{row['Sales']:.0f}", ha='center')
plt.title('Q2: Average Daily Sales by Store Type')
plt.ylabel('Average Sales')
plt.xlabel('Store Type')
plt.tight_layout()
plt.show()

# ==============================
# 🟢 Q3: Promo vs Non-Promo Sales by Store Type (with exact numbers)
# ==============================
promo_stats = df.groupby(['StoreType', 'Promo'])['Sales'].mean().reset_index()
pivot = promo_stats.pivot(index='StoreType', columns='Promo', values='Sales').reset_index()
pivot.columns = ['StoreType', 'No Promo', 'Promo']

# Plotting
ax = pivot.plot(x='StoreType', kind='bar', figsize=(10,6), rot=0)
plt.title('Q3: Promo vs Non-Promo Daily Sales by Store Type')
plt.ylabel('Average Sales')
plt.xlabel('Store Type')
plt.legend(['No Promo', 'Promo'])
plt.tight_layout()

# Add text labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

plt.show()

# ==============================
# 🟢 Q4: Effect of Assortment on Sales
# ==============================
avg_sales_assortment = df.groupby('Assortment')['Sales'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=avg_sales_assortment, x='Assortment', y='Sales', palette='Set2')
for i, row in avg_sales_assortment.iterrows():
    plt.text(i, row['Sales'] + 200, f"{row['Sales']:.0f}", ha='center')
plt.title('Q4: Average Sales by Assortment Type')
plt.ylabel('Average Sales')
plt.xlabel('Assortment Type')
plt.tight_layout()
plt.show()

# ==============================
# 🟢 Q5: Correlation Matrix
# ==============================
# Include only numerical columns and engineered durations if they exist
possible_cols = ['Sales', 'Customers', 'Promo', 'Promo2',
                 'CompetitionDistance', 'PromoActiveDurationMonths', 'CompetitionOpenDurationMonths']

corr_cols = [col for col in possible_cols if col in df.columns]

corr = df[corr_cols].corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Q5: Correlation Matrix of Sales-related Features')
plt.tight_layout()
plt.show()
