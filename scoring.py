import pandas as pd
import numpy as np
from datetime import datetime

# Load the CSV data
df = pd.read_csv('presales_data_sample.csv')

print(f"Total rows in dataset: {len(df)}")
print(f"Expected companies: {len(df) // 5}")
print(f"Unique input_row_key values: {df['input_row_key'].nunique()}")

# Define the scoring function
def calculate_accuracy_score(row):
    score = 0

    # 1. COMPLETENESS SCORE (40 points)
    completeness_score = 0

    # Contact info (10 points)
    contact_points = 0
    if pd.notna(row.get('primary_phone')) and row['primary_phone'] not in ['', ' ']:
        contact_points += 3.33
    if pd.notna(row.get('primary_email')) and row['primary_email'] not in ['', ' ']:
        contact_points += 3.33
    if pd.notna(row.get('website_url')) and row['website_url'] not in ['', ' ']:
        contact_points += 3.34
    completeness_score += min(contact_points, 10)

    # Revenue data (10 points)
    if pd.notna(row.get('revenue')) and row['revenue'] not in ['', ' ']:
        if row.get('revenue_type') == 'extracted':
            completeness_score += 10
        elif row.get('revenue_type') == 'modelled':
            completeness_score += 5

    # Employee count (10 points)
    if pd.notna(row.get('employee_count')) and row['employee_count'] not in ['', ' ']:
        if row.get('employee_count_type') == 'extracted':
            completeness_score += 10
        elif row.get('employee_count_type') == 'modelled':
            completeness_score += 5

    # Year founded (5 points)
    if pd.notna(row.get('year_founded')) and row['year_founded'] not in ['', ' ']:
        completeness_score += 5

    # Legal names (5 points)
    if pd.notna(row.get('company_legal_names')) and row['company_legal_names'] not in ['', ' ']:
        completeness_score += 5

    score += completeness_score

    # 2. BUSINESS SUBSTANCE SCORE (40 points)
    substance_score = 0

    # Description detail (15 points)
    if pd.notna(row.get('generated_description')):
        desc_length = len(str(row['generated_description']))
        if desc_length > 200:
            substance_score += 15
        elif desc_length > 100:
            substance_score += 10
        elif desc_length > 50:
            substance_score += 5

    # Industry classification (10 points)
    industry_points = 0
    if pd.notna(row.get('business_tags')) and row['business_tags'] not in ['', ' ']:
        industry_points += 3.33
    if pd.notna(row.get('naics_2022_primary_code')) and row['naics_2022_primary_code'] not in ['', ' ']:
        industry_points += 3.33
    if pd.notna(row.get('main_industry')) and row['main_industry'] not in ['', ' ']:
        industry_points += 3.34
    substance_score += min(industry_points, 10)

    # Multiple locations (10 points)
    if pd.notna(row.get('num_locations')):
        try:
            num_locs = float(row['num_locations'])
            if num_locs > 1:
                substance_score += 10
            elif num_locs == 1:
                substance_score += 5
        except (ValueError, TypeError):
            pass

    # Technology stack (5 points)
    if pd.notna(row.get('technologies')) and row['technologies'] not in ['', ' ']:
        substance_score += 5

    score += substance_score

    # 3. DATA QUALITY SCORE (20 points)
    quality_score = 0

    # Data freshness (10 points)
    if pd.notna(row.get('last_updated_at')):
        try:
            last_updated = pd.to_datetime(row['last_updated_at'])
            days_since_update = (datetime.now() - last_updated).days
            if days_since_update <= 365:  # Within 1 year
                quality_score += 10
            elif days_since_update <= 730:  # Within 2 years
                quality_score += 5
        except (ValueError, TypeError):
            pass

    # Data type quality (10 points)
    data_type_points = 0
    if row.get('revenue_type') == 'extracted':
        data_type_points += 5
    elif row.get('revenue_type') == 'modelled':
        data_type_points += 2.5

    if row.get('employee_count_type') == 'extracted':
        data_type_points += 5
    elif row.get('employee_count_type') == 'modelled':
        data_type_points += 2.5

    quality_score += min(data_type_points, 10)

    score += quality_score

    return {
        'total_score': round(score, 2),
        'completeness_score': round(completeness_score, 2),
        'substance_score': round(substance_score, 2),
        'quality_score': round(quality_score, 2)
    }

# Apply scoring to all rows
print("Calculating accuracy scores for all rows...")
scores = []
for idx, row in df.iterrows():
    score_data = calculate_accuracy_score(row)
    scores.append(score_data)

# Add scores to dataframe
scores_df = pd.DataFrame(scores)
df_scored = pd.concat([df, scores_df], axis=1)

# Group by input_row_key and select best record from each group
print("Selecting best records from each company group...")
selected_companies = []

for group_key in sorted(df_scored['input_row_key'].unique()):
    group_data = df_scored[df_scored['input_row_key'] == group_key]

    if len(group_data) > 0:
        # Sort by total_score, then by last_updated_at, then by description length
        group_data = group_data.copy()
        group_data['desc_length'] = group_data['generated_description'].fillna('').str.len()

        # Sort and select best
        best_row = group_data.sort_values(
            ['total_score', 'last_updated_at', 'desc_length'],
            ascending=[False, False, False]
        ).iloc[0]

        selected_companies.append(best_row)

# Create final results dataframe
results_df = pd.DataFrame(selected_companies)

print(f"\nSelected {len(results_df)} companies out of {len(df_scored['input_row_key'].unique())} groups")

# Display score distribution
print("\nScore Distribution:")
print(f"Average Total Score: {results_df['total_score'].mean():.2f}")
print(f"Max Total Score: {results_df['total_score'].max():.2f}")
print(f"Min Total Score: {results_df['total_score'].min():.2f}")

# Show some examples of high-scoring companies
print("\nTop 10 Highest Scoring Companies:")
top_10 = results_df.nlargest(10, 'total_score')[['company_name', 'total_score', 'completeness_score', 'substance_score', 'quality_score']]
print(top_10.to_string(index=False))

# Save results to CSV
output_filename = 'selected_companies_with_scores.csv'
results_df.to_csv(output_filename, index=False)
print(f"\nResults saved to: {output_filename}")

# Show detailed breakdown for a sample company
print("\nDetailed breakdown for a sample company:")
sample_company = results_df.iloc[0]
print(f"Company: {sample_company['company_name']}")
print(f"Total Score: {sample_company['total_score']}")
print(f" - Completeness: {sample_company['completeness_score']}")
print(f" - Substance: {sample_company['substance_score']}")
print(f" - Quality: {sample_company['quality_score']}")
print(f"Revenue Type: {sample_company.get('revenue_type', 'N/A')}")
print(f"Employee Count Type: {sample_company.get('employee_count_type', 'N/A')}")
print(f"Last Updated: {sample_company.get('last_updated_at', 'N/A')}")
