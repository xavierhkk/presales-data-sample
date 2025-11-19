import pandas as pd
import numpy as np
import re

def calculate_real_presence_score(row):
    """Calculate score for real business presence"""
    score = 0

    # Company type
    if pd.notna(row.get('company_type')) and row['company_type'] in ['Private', 'Public']:
        score += 3

    # Employee count
    if pd.notna(row.get('employee_count')) and row['employee_count'] > 0:
        if row['employee_count'] > 5:
            score += 2
        else:
            score += 1

    # Contact information
    if pd.notna(row.get('website_url')) and row['website_url'] != '':
        score += 2

    if pd.notna(row.get('primary_phone')) and row['primary_phone'] != '':
        score += 1

    # Physical address
    if pd.notna(row.get('main_street')) and row['main_street'] != '':
        score += 1
    if pd.notna(row.get('main_city')) and row['main_city'] != '':
        score += 1

    # Financial indicators
    if pd.notna(row.get('revenue')) and row['revenue'] > 1000000:
        score += 1

    # Year founded (reasonable range)
    if pd.notna(row.get('year_founded')) and 1850 <= row['year_founded'] <= 2024:
        score += 1

    return score

def calculate_manufacturing_relevance_score(row):
    """Calculate score for manufacturing relevance"""
    score = 0

    # NAICS codes for manufacturing (31-33)
    naics_codes = []
    if pd.notna(row.get('naics_2022_primary_code')):
        naics_codes.append(str(row['naics_2022_primary_code']))
    if pd.notna(row.get('naics_2022_secondary_codes')):
        naics_codes.extend(str(row['naics_2022_secondary_codes']).split(','))

    for code in naics_codes:
        code = str(code).strip()
        if code.startswith(('31', '32', '33')):
            score += 5
            break

    # SIC codes for manufacturing (2000-3999)
    sic_codes = []
    if pd.notna(row.get('sic_codes')):
        sic_codes = str(row['sic_codes']).split('|')

    for code in sic_codes:
        code = str(code).strip()
        if code.isdigit() and 2000 <= int(code) <= 3999:
            score += 3
            break

    # Industry keywords
    industry_fields = [
        str(row.get('main_industry', '')),
        str(row.get('main_sector', '')),
        str(row.get('main_business_category', '')),
        str(row.get('generated_description', '')),
        str(row.get('short_description', '')),
        str(row.get('long_description', ''))
    ]

    manufacturing_keywords = [
        'manufactur', 'production', 'industrial', 'factory', 'plant',
        'machinery', 'equipment', 'fabrication', 'assembly', 'processing',
        'engineering', 'supply chain', 'logistics', 'warehousing',
        'automation', 'robotics', 'machining', 'tooling', 'metal',
        'plastic', 'chemical', 'electrical', 'mechanical'
    ]

    for field in industry_fields:
        field_lower = field.lower()
        for keyword in manufacturing_keywords:
            if keyword in field_lower:
                score += 2
                break

    # Business tags
    if pd.notna(row.get('business_tags')):
        tags = str(row['business_tags']).lower()
        for keyword in manufacturing_keywords:
            if keyword in tags:
                score += 1
                break

    return score

def calculate_total_score(row):
    """Calculate total score combining presence and manufacturing relevance"""
    presence_score = calculate_real_presence_score(row)
    manufacturing_score = calculate_manufacturing_relevance_score(row)

    # Weight manufacturing relevance higher for final selection
    total_score = presence_score + (manufacturing_score * 1.5)

    return total_score, presence_score, manufacturing_score

def select_companies_from_groups(df):
    """Select the best company from each group of 5 rows starting from row 2"""
    selected_companies = []
    group_analysis = []

    # Calculate total number of groups (starting from row 2)
    total_rows = len(df)
    num_groups = (total_rows - 1) // 5  # -1 because we start from row 2 (index 1)

    print(f"Total rows: {total_rows}")
    print(f"Processing {num_groups} groups starting from row 2...")
    print("Group structure: 2-6, 7-11, 12-16, etc.")

    for group_num in range(num_groups):
        # Calculate indices: group 0 = rows 2-6 (indices 1-5), group 1 = rows 7-11 (indices 6-10), etc.
        start_idx = 1 + (group_num * 5)  # Start from index 1 (row 2)
        end_idx = start_idx + 5

        # Ensure we don't go beyond dataframe bounds
        if end_idx > total_rows:
            end_idx = total_rows

        group_df = df.iloc[start_idx:end_idx].copy()

        # Skip if group has less than 1 company (shouldn't happen with proper grouping)
        if len(group_df) == 0:
            continue

        group_results = []

        # Calculate scores for each company in the group
        for idx, row in group_df.iterrows():
            total_score, presence_score, manufacturing_score = calculate_total_score(row)

            group_results.append({
                'original_index': idx,
                'company_name': row.get('company_name', ''),
                'total_score': total_score,
                'presence_score': presence_score,
                'manufacturing_score': manufacturing_score,
                'employee_count': row.get('employee_count', 0),
                'revenue': row.get('revenue', 0),
                'website': row.get('website_url', ''),
                'naics_primary': row.get('naics_2022_primary_code', ''),
                'main_industry': row.get('main_industry', '')
            })

        # Sort by total score and select the best
        group_results.sort(key=lambda x: x['total_score'], reverse=True)
        best_company = group_results[0]

        selected_companies.append(df.iloc[best_company['original_index']])

        # Store group analysis for reporting
        group_analysis.append({
            'group': group_num + 1,
            'rows_covered': f"{start_idx+1}-{end_idx}" if end_idx <= total_rows else f"{start_idx+1}-{total_rows}",
            'selected_company': best_company['company_name'],
            'selected_score': best_company['total_score'],
            'group_size': len(group_results),
            'score_range': f"{min(r['total_score'] for r in group_results):.1f}-{max(r['total_score'] for r in group_results):.1f}",
            'top_3_scores': [r['total_score'] for r in group_results[:3]]
        })

    selected_df = pd.DataFrame(selected_companies)

    print(f"\nSelected {len(selected_df)} companies from {num_groups} groups")

    return selected_df, group_analysis

def filter_manufacturing_companies(selected_df):
    """Filter companies for manufacturing relevance"""
    manufacturing_companies = []

    manufacturing_keywords = [
        'manufactur', 'production', 'industrial', 'factory', 'plant',
        'machinery', 'equipment', 'fabrication', 'assembly', 'processing',
        'engineering', 'supply chain', 'logistics', 'warehousing'
    ]

    for idx, row in selected_df.iterrows():
        # Check NAICS codes (31-33 are manufacturing)
        naics_primary = str(row.get('naics_2022_primary_code', ''))
        naics_secondary = str(row.get('naics_2022_secondary_codes', ''))

        is_manufacturing_naics = (
            naics_primary.startswith(('31', '32', '33')) or
            any(code.strip().startswith(('31', '32', '33')) for code in naics_secondary.split(',') if code.strip())
        )

        # Check industry descriptions
        industry_fields = [
            str(row.get('main_industry', '')),
            str(row.get('main_sector', '')),
            str(row.get('main_business_category', '')),
            str(row.get('generated_description', ''))
        ]

        is_manufacturing_desc = any(
            any(keyword in field.lower() for keyword in manufacturing_keywords)
            for field in industry_fields
        )

        # Check SIC codes (2000-3999 are manufacturing)
        sic_codes = str(row.get('sic_codes', ''))
        is_manufacturing_sic = any(
            code.strip().isdigit() and 2000 <= int(code.strip()) <= 3999
            for code in sic_codes.split('|')
            if code.strip()
        )

        if is_manufacturing_naics or is_manufacturing_desc or is_manufacturing_sic:
            manufacturing_companies.append(row)

    manufacturing_df = pd.DataFrame(manufacturing_companies)

    print(f"Found {len(manufacturing_df)} manufacturing-relevant companies")

    return manufacturing_df

def generate_analysis_report(final_companies, group_analysis):
    """Generate detailed analysis report"""

    print("\n=== DETAILED ANALYSIS ===")

    # Score distribution
    if len(final_companies) > 0:
        score_ranges = {
            'Excellent (15+)': len(final_companies[final_companies['total_score'] >= 15]),
            'Good (10-14)': len(final_companies[(final_companies['total_score'] >= 10) & (final_companies['total_score'] < 15)]),
            'Fair (5-9)': len(final_companies[(final_companies['total_score'] >= 5) & (final_companies['total_score'] < 10)]),
            'Poor (<5)': len(final_companies[final_companies['total_score'] < 5])
        }

        print("\nScore Distribution:")
        for range_name, count in score_ranges.items():
            percentage = (count / len(final_companies)) * 100
            print(f"  {range_name}: {count} companies ({percentage:.1f}%)")

        # Company size analysis
        if 'employee_count' in final_companies.columns:
            employee_ranges = {
                'Large (500+)': len(final_companies[final_companies['employee_count'] >= 500]),
                'Medium (50-499)': len(final_companies[(final_companies['employee_count'] >= 50) & (final_companies['employee_count'] < 500)]),
                'Small (1-49)': len(final_companies[(final_companies['employee_count'] >= 1) & (final_companies['employee_count'] < 50)]),
                'Unknown': len(final_companies[final_companies['employee_count'].isna() | (final_companies['employee_count'] == 0)])
            }

            print("\nCompany Size Distribution:")
            for size, count in employee_ranges.items():
                percentage = (count / len(final_companies)) * 100
                print(f"  {size}: {count} companies ({percentage:.1f}%)")

        # Industry breakdown
        if 'main_industry' in final_companies.columns:
            print("\nTop Industries:")
            industry_counts = final_companies['main_industry'].value_counts().head(10)
            for industry, count in industry_counts.items():
                print(f"  {industry}: {count} companies")

    # Group analysis summary
    if group_analysis:
        avg_group_score = np.mean([group['selected_score'] for group in group_analysis])
        print(f"\nAverage selected company score: {avg_group_score:.2f}")

        # Show first few group selections as example
        print("\nSample Group Selections:")
        for group in group_analysis[:5]:
            print(f"  Group {group['group']} (rows {group['rows_covered']}): {group['selected_company']} (score: {group['selected_score']:.1f})")

    return

def export_contact_list(final_companies):
    """Export clean contact information for the manufacturing companies"""

    contact_data = []

    for idx, row in final_companies.iterrows():
        contact_info = {
            'company_name': row.get('company_name', ''),
            'website': row.get('website_url', ''),
            'primary_phone': row.get('primary_phone', ''),
            'primary_email': row.get('primary_email', ''),
            'main_country': row.get('main_country', ''),
            'main_city': row.get('main_city', ''),
            'employee_count': row.get('employee_count', ''),
            'revenue': row.get('revenue', ''),
            'main_industry': row.get('main_industry', ''),
            'total_score': row.get('total_score', 0)
        }

        # Clean phone numbers
        if pd.notna(contact_info['primary_phone']):
            contact_info['primary_phone'] = str(contact_info['primary_phone']).split(',')[0].strip()

        contact_data.append(contact_info)

    contact_df = pd.DataFrame(contact_data)
    contact_df = contact_df.sort_values('total_score', ascending=False)

    # Save contact list
    contact_df.to_csv('manufacturing_contacts_prioritized.csv', index=False)
    print(f"\nExported contact list for {len(contact_df)} manufacturing companies")

    return contact_df

def main():
    print("=== Manufacturing Company Selection Pipeline ===")
    print("Loading and processing data...")

    try:
        # Load the data
        df = pd.read_csv('presales_data_sample.csv')
        print(f"Original dataset shape: {df.shape}")
        print(f"Total records: {len(df)}")

        # Step 1: Select best companies from each group (starting from row 2)
        selected_companies, group_analysis = select_companies_from_groups(df)

        # Step 2: Filter for manufacturing relevance
        manufacturing_companies = filter_manufacturing_companies(selected_companies)

        # Step 3: Add scores to manufacturing companies
        manufacturing_scored = []
        for idx, row in manufacturing_companies.iterrows():
            total_score, presence_score, manufacturing_score = calculate_total_score(row)

            scored_company = row.to_dict()
            scored_company.update({
                'total_score': total_score,
                'presence_score': presence_score,
                'manufacturing_score': manufacturing_score
            })
            manufacturing_scored.append(scored_company)

        manufacturing_df_final = pd.DataFrame(manufacturing_scored)

        # Sort by total score
        manufacturing_df_final = manufacturing_df_final.sort_values('total_score', ascending=False)

        # Step 4: Generate reports
        print("\n=== RESULTS SUMMARY ===")
        print(f"Original dataset: {len(df)} companies")
        print(f"After group selection: {len(selected_companies)} companies")
        print(f"Manufacturing-relevant companies: {len(manufacturing_df_final)} companies")

        # Save results
        selected_companies.to_csv('selected_companies_590.csv', index=False)
        manufacturing_df_final.to_csv('manufacturing_companies_filtered.csv', index=False)

        print("\n=== TOP 10 MANUFACTURING COMPANIES ===")
        if len(manufacturing_df_final) > 0:
            display_columns = ['company_name', 'main_industry', 'employee_count',
                              'total_score', 'presence_score', 'manufacturing_score']

            # Only include columns that exist in the dataframe
            available_columns = [col for col in display_columns if col in manufacturing_df_final.columns]

            top_10 = manufacturing_df_final.head(10)[available_columns]
            print(top_10.to_string(index=False))
        else:
            print("No manufacturing companies found.")

        # Generate detailed analysis
        generate_analysis_report(manufacturing_df_final, group_analysis)

        # Export contacts
        if len(manufacturing_df_final) > 0:
            contact_list = export_contact_list(manufacturing_df_final)

        return manufacturing_df_final, group_analysis

    except FileNotFoundError:
        print("Error: File 'presales_data_sample.csv' not found.")
        return None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

# Run the pipeline
if __name__ == "__main__":
    final_companies, analysis = main()
