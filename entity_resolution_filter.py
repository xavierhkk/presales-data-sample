# entity_resolution_filter.py
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from fuzzywuzzy import fuzz
from urllib.parse import urlparse

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("==============================================")
print("ENTITY RESOLUTION INCONSISTENCY CHECK PROCESS")
print("==============================================")
print(f"Starting process at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# STEP 1: LOAD THE DATA
print("STEP 1: LOADING DATA")
print("-------------------")

# Check if the CSV file exists in the current directory
csv_file = 'presales_data_sample.csv'
if not os.path.exists(csv_file):
    print(f"ERROR: File '{csv_file}' not found in current directory.")
    print("Please ensure the file is in the same directory as this script.")
    print("Current directory:", os.getcwd())
    exit(1)

# Load the data
try:
    df = pd.read_csv(csv_file)
    print(f"✓ Successfully loaded {csv_file}")
    print(f"  - Total records: {len(df)}")
    print(f"  - Columns detected: {len(df.columns)}")
    print(f"  - First 3 column names: {df.columns[:3].tolist()}\n")
except Exception as e:
    print(f"ERROR: Failed to load CSV file. Error: {str(e)}")
    exit(1)

# STEP 2: PREPARE DATA FOR ANALYSIS
print("STEP 2: PREPARING DATA FOR ANALYSIS")
print("---------------------------------")

# Identify key columns that might exist in the dataset
# Try to map to standard column names based on content
def identify_columns(dataframe):
    """Identify relevant columns in the dataset based on content patterns"""
    column_mapping = {}

    # Look for original entity name
    possible_orig_names = ['original_entity_name', 'original_name', 'input_name', 'company_name', 'supplier_name', 'input_company_name']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_orig_names):
            column_mapping['original_entity_name'] = col
            break

    # Look for best match name
    possible_match_names = ['best_match_name', 'matched_name', 'resolved_name', 'canonical_name']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_match_names):
            column_mapping['best_match_name'] = col
            break

    # Look for website URL
    possible_url_names = ['website_url', 'url', 'site', 'domain', 'homepage']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_url_names):
            column_mapping['website_url'] = col
            break

    # Look for match source URLs
    possible_source_names = ['match_source_urls', 'source_urls', 'sources', 'references']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_source_names):
            column_mapping['match_source_urls'] = col
            break

    # Look for industry/classification codes
    possible_industry_names = ['industry_codes', 'industry', 'sector', 'classification', 'main_industry']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_industry_names):
            column_mapping['industry_codes'] = col
            break

    # Look for NAICS codes
    possible_naics_names = ['naics_code', 'naics', 'industry_code', 'naics_2022_primary_code']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_naics_names):
            column_mapping['naics_code'] = col
            break

    # Look for descriptions
    possible_desc_names = ['description', 'desc', 'company_description', 'business_description', 'generated_description']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_desc_names):
            column_mapping['description'] = col
            break

    # Look for contact information
    possible_phone_names = ['phone_numbers', 'phone', 'contact_phone', 'telephone', 'primary_phone']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_phone_names):
            column_mapping['phone_numbers'] = col
            break

    possible_email_names = ['email_addresses', 'email', 'contact_email', 'emails', 'primary_email']
    for col in dataframe.columns:
        if any(name in col.lower() for name in possible_email_names):
            column_mapping['email_addresses'] = col
            break

    return column_mapping

# Identify and map columns
column_mapping = identify_columns(df)
print("Column mapping identified:")
for key, value in column_mapping.items():
    print(f"  - {key} -> {value}")

# Create a working copy with standardized column names
resolution_df = df.copy()

# Rename columns to standard names if mapping was found
for standard_name, actual_name in column_mapping.items():
    if actual_name in resolution_df.columns and standard_name != actual_name:
        resolution_df.rename(columns={actual_name: standard_name}, inplace=True)
        print(f"✓ Mapped '{actual_name}' to '{standard_name}'")

# Add missing standard columns if they don't exist
standard_columns = ['original_entity_name', 'best_match_name', 'website_url', 'match_source_urls',
                    'industry_codes', 'naics_code', 'description', 'phone_numbers', 'email_addresses']

for col in standard_columns:
    if col not in resolution_df.columns:
        resolution_df[col] = None
        print(f"✓ Created empty column for '{col}' (not found in source data)")

print(f"\nWorking dataset shape after preparation: {resolution_df.shape}\n")

# STEP 3: DEFINE INCONSISTENCY DETECTION FUNCTIONS
print("STEP 3: DEFINING INCONSISTENCY DETECTION FUNCTIONS")
print("------------------------------------------------")

def convert_to_scalar(value):
    """Convert pandas Series to scalar value if needed"""
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return value.item()
        else:
            return None
    return value

def calculate_match_confidence(row):
    """Calculate confidence score for entity resolution match (0-1 scale)"""
    # Convert row to dictionary for safer access
    row_dict = row.to_dict()

    confidence_components = []

    # 1. Name similarity score (40% weight)
    name_similarity = 0.5  # Default score
    orig_name = convert_to_scalar(row_dict.get('original_entity_name'))
    best_match_name = convert_to_scalar(row_dict.get('best_match_name'))

    if pd.notna(orig_name):
        orig_name_str = str(orig_name).lower().strip()
        match_name_str = str(best_match_name).lower().strip() if pd.notna(best_match_name) else ''
        name_similarity = fuzz.token_sort_ratio(orig_name_str, match_name_str) / 100.0
    confidence_components.append(('name_similarity', min(1.0, name_similarity) * 0.4))

    # 2. Domain verification (30% weight)
    domain_verified = 0.3  # Default partial score
    website_url = convert_to_scalar(row_dict.get('website_url'))
    match_source_urls = convert_to_scalar(row_dict.get('match_source_urls'))

    if pd.notna(website_url) and isinstance(website_url, str) and str(website_url).startswith('http'):
        domain = urlparse(str(website_url)).netloc.lower()
        if pd.notna(match_source_urls):
            sources = str(match_source_urls).lower()
            if domain in sources:
                domain_verified = 1.0
    confidence_components.append(('domain_verification', domain_verified * 0.3))

    # 3. Industry classification alignment (15% weight)
    industry_alignment = 0.6  # Default moderate score
    industry_codes = convert_to_scalar(row_dict.get('industry_codes'))
    naics_code = convert_to_scalar(row_dict.get('naics_code'))

    if pd.notna(industry_codes) and pd.notna(naics_code):
        industry_text = str(industry_codes).lower()
        naics_text = str(naics_code).lower()
        # Check for obvious mismatches
        if ('software' in industry_text or 'it' in industry_text) and 'manufactur' in naics_text:
            industry_alignment = 0.2
        elif ('manufactur' in industry_text) and ('software' in naics_text or 'it' in naics_text):
            industry_alignment = 0.2
        else:
            industry_alignment = 0.8
    confidence_components.append(('industry_alignment', industry_alignment * 0.15))

    # 4. Contact information availability (15% weight) - FIXED SECTION
    contact_score = 0.4  # Default partial score

    # Get scalar values first to avoid pandas Series ambiguity error
    phone_value = convert_to_scalar(row_dict.get('phone_numbers'))
    email_value = convert_to_scalar(row_dict.get('email_addresses'))

    has_phone = False
    if pd.notna(phone_value):
        phone_str = str(phone_value).strip()
        has_phone = phone_str != '' and phone_str.lower() != 'nan' and phone_str != 'none'

    has_email = False
    if pd.notna(email_value):
        email_str = str(email_value).strip()
        has_email = email_str != '' and email_str.lower() != 'nan' and email_str != 'none' and '@' in email_str

    if has_phone and has_email:
        contact_score = 1.0
    elif has_phone or has_email:
        contact_score = 0.7

    confidence_components.append(('contact_availability', contact_score * 0.15))

    # Calculate final score
    total_score = sum(score for _, score in confidence_components)
    return min(1.0, max(0.0, total_score))

def detect_classification_inconsistency(row):
    """Check for inconsistent industry classifications"""
    # Convert row to dictionary for safer access
    row_dict = row.to_dict()

    issues = []

    industry_codes = convert_to_scalar(row_dict.get('industry_codes'))
    naics_code = convert_to_scalar(row_dict.get('naics_code'))

    if pd.notna(industry_codes) and pd.notna(naics_code):
        industry_codes = str(industry_codes).lower()
        naics_code = str(naics_code).lower()

        # Check for obvious classification mismatches
        if ('retail' in industry_codes or 'store' in industry_codes) and 'manufactur' in naics_code:
            issues.append('Retail/Manufacturing classification mismatch')
        elif 'service' in industry_codes and 'manufactur' in naics_code:
            issues.append('Service/Manufacturing classification mismatch')
        elif ('software' in industry_codes or 'it' in industry_codes) and 'manufactur' in naics_code:
            issues.append('Software/Manufacturing classification mismatch')
        elif 'manufactur' in industry_codes and ('software' in naics_code or 'it' in naics_code):
            issues.append('Manufacturing/Software classification mismatch')

    return '; '.join(issues) if issues else None

def detect_contact_inconsistency(row):
    """Check for inconsistent or outdated contact information"""
    # Convert row to dictionary for safer access
    row_dict = row.to_dict()

    issues = []

    # Check phone numbers if available
    phone_numbers = convert_to_scalar(row_dict.get('phone_numbers'))
    if pd.notna(phone_numbers):
        phones = str(phone_numbers).split('|')
        for phone in phones:
            phone = phone.strip()
            if phone and len(phone) > 3:  # Skip very short strings
                # Basic phone format validation
                digits = re.sub(r'[^\d]', '', phone)
                if len(digits) < 7 or len(digits) > 15:  # Reasonable phone number length
                    issues.append(f'Invalid phone format: {phone}')

    # Check email addresses if available
    email_addresses = convert_to_scalar(row_dict.get('email_addresses'))
    if pd.notna(email_addresses):
        emails = str(email_addresses).split('|')
        for email in emails:
            email = email.strip().lower()
            if email and len(email) > 5:  # Skip very short strings
                # Basic email format validation
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    issues.append(f'Invalid email format: {email}')

    return '; '.join(issues) if issues else None

def detect_description_inconsistency(row):
    """Check if description aligns with industry classification"""
    # Convert row to dictionary for safer access
    row_dict = row.to_dict()

    issues = []

    description = convert_to_scalar(row_dict.get('description'))
    industry_codes = convert_to_scalar(row_dict.get('industry_codes'))

    if pd.notna(description) and pd.notna(industry_codes):
        desc = str(description).lower()
        industry = str(industry_codes).lower()

        # Define keywords for different industries
        tech_keywords = ['software', 'digital', 'technology', 'data', 'cloud', 'it', 'system', 'application', 'platform']
        manufacturing_keywords = ['manufactur', 'factory', 'production', 'plant', 'equipment', 'hardware', 'industrial']
        service_keywords = ['service', 'consulting', 'support', 'solution', 'professional']

        # Check for misalignments
        if any(word in industry for word in ['software', 'technology', 'data', 'it', 'cloud', 'digital']) and not any(word in desc for word in tech_keywords):
            issues.append('Description lacks technology keywords despite technical classification')
        elif 'manufactur' in industry and not any(word in desc for word in manufacturing_keywords):
            issues.append('Description lacks manufacturing keywords despite manufacturing classification')
        elif 'service' in industry and not any(word in desc for word in service_keywords):
            issues.append('Description lacks service keywords despite service classification')
        elif any(word in industry for word in ['retail', 'store', 'shop']) and not any(word in desc for word in ['retail', 'store', 'shop', 'sell', 'customer', 'point of sale']):
            issues.append('Description lacks retail keywords despite retail classification')

    return '; '.join(issues) if issues else None

def detect_website_inconsistency(row):
    """Check for website-related inconsistencies"""
    # Convert row to dictionary for safer access
    row_dict = row.to_dict()

    issues = []

    website_url = convert_to_scalar(row_dict.get('website_url'))
    best_match_name = convert_to_scalar(row_dict.get('best_match_name'))

    if pd.notna(website_url):
        url = str(website_url).strip().lower()

        # Check if URL has proper format
        if not url.startswith(('http://', 'https://')):
            issues.append('Missing protocol in website URL (should start with http:// or https://)')

        # Check for suspicious domains
        if any(suspicious in url for suspicious in ['temp', 'test', 'demo', 'example.com', 'localhost', 'example.org']):
            issues.append('Suspicious test/demo domain in website URL')

        # Check if domain matches company name (rough check)
        if pd.notna(best_match_name):
            company_name = str(best_match_name).lower()
            try:
                domain = urlparse(url).netloc

                # Extract main part of domain (without TLD)
                domain_parts = domain.split('.')
                main_domain = domain_parts[-2] if len(domain_parts) > 1 else domain

                # Check if company name contains domain or vice versa
                if main_domain not in company_name and company_name not in main_domain:
                    # Only flag if company name is reasonably long
                    if len(company_name) > 4 and len(main_domain) > 3:
                        issues.append('Domain name does not match company name')
            except:
                issues.append('Could not parse website domain')

    return '; '.join(issues) if issues else None

print("✓ All inconsistency detection functions defined\n")

# STEP 4: RUN INCONSISTENCY CHECKS
print("STEP 4: RUNNING INCONSISTENCY CHECKS")
print("---------------------------------")

print("Calculating match confidence scores...")
resolution_df['match_confidence'] = resolution_df.apply(calculate_match_confidence, axis=1)

print("Checking classification inconsistencies...")
resolution_df['classification_issues'] = resolution_df.apply(detect_classification_inconsistency, axis=1)

print("Checking contact information inconsistencies...")
resolution_df['contact_issues'] = resolution_df.apply(detect_contact_inconsistency, axis=1)

print("Checking description inconsistencies...")
resolution_df['description_issues'] = resolution_df.apply(detect_description_inconsistency, axis=1)

print("Checking website inconsistencies...")
resolution_df['website_issues'] = resolution_df.apply(detect_website_inconsistency, axis=1)

print("Compiling results...")
# Create combined inconsistency flags
resolution_df['has_inconsistencies'] = resolution_df.apply(
    lambda row: any([
        pd.notna(row['classification_issues']),
        pd.notna(row['contact_issues']),
        pd.notna(row['description_issues']),
        pd.notna(row['website_issues'])
    ]), axis=1
)

# Combine all issues into a single column for easy review
resolution_df['inconsistency_details'] = resolution_df.apply(
    lambda row: '; '.join(filter(None, [
        str(row['classification_issues']) if pd.notna(row['classification_issues']) else '',
        str(row['contact_issues']) if pd.notna(row['contact_issues']) else '',
        str(row['description_issues']) if pd.notna(row['description_issues']) else '',
        str(row['website_issues']) if pd.notna(row['website_issues']) else ''
    ])), axis=1
)

# Add resolution status
def get_resolution_status(row):
    row_dict = row.to_dict()
    match_confidence = row_dict['match_confidence']
    has_inconsistencies = row_dict['has_inconsistencies']

    if match_confidence >= 0.85 and not has_inconsistencies:
        return 'high_confidence'
    elif match_confidence >= 0.7 and not has_inconsistencies:
        return 'good_confidence'
    elif match_confidence >= 0.6:
        return 'needs_review'
    else:
        return 'low_confidence'

resolution_df['resolution_status'] = resolution_df.apply(get_resolution_status, axis=1)

print("✓ All inconsistency checks completed\n")

# STEP 5: CALCULATE DATA QUALITY SCORE
print("STEP 5: CALCULATING DATA QUALITY SCORES")
print("-------------------------------------")

def calculate_data_quality_score(row):
    """Calculate overall data quality score (0-100)"""
    row_dict = row.to_dict()
    score = 100  # Start with perfect score

    # Deduct points for inconsistencies
    if pd.notna(row_dict['classification_issues']):
        score -= 20
    if pd.notna(row_dict['contact_issues']):
        score -= 15
    if pd.notna(row_dict['description_issues']):
        score -= 15
    if pd.notna(row_dict['website_issues']):
        score -= 10

    # Deduct points for missing critical information
    website_url = convert_to_scalar(row_dict.get('website_url'))
    phone_numbers = convert_to_scalar(row_dict.get('phone_numbers'))
    email_addresses = convert_to_scalar(row_dict.get('email_addresses'))

    if pd.isna(website_url) or not str(website_url).strip() or str(website_url).lower() in ['nan', 'none', '']:
        score -= 10

    phone_available = pd.notna(phone_numbers) and str(phone_numbers).strip() not in ['', 'nan', 'none']
    email_available = pd.notna(email_addresses) and str(email_addresses).strip() not in ['', 'nan', 'none']

    if not phone_available and not email_available:
        score -= 15

    # Deduct points based on confidence score
    confidence_penalty = int((1 - row_dict['match_confidence']) * 25)
    score -= confidence_penalty

    return max(0, min(100, score))

print("Calculating quality scores for all entities...")
resolution_df['data_quality_score'] = resolution_df.apply(calculate_data_quality_score, axis=1)

print("✓ Data quality scores calculated\n")

# STEP 6: GENERATE SUMMARY STATISTICS
print("STEP 6: GENERATING SUMMARY STATISTICS")
print("-----------------------------------")

total_entities = len(resolution_df)
high_confidence = len(resolution_df[resolution_df['resolution_status'] == 'high_confidence'])
good_confidence = len(resolution_df[resolution_df['resolution_status'] == 'good_confidence'])
needs_review = len(resolution_df[resolution_df['resolution_status'] == 'needs_review'])
low_confidence = len(resolution_df[resolution_df['resolution_status'] == 'low_confidence'])

with_issues = len(resolution_df[resolution_df['has_inconsistencies']])
avg_quality = resolution_df['data_quality_score'].mean()

print(f"TOTAL ENTITIES ANALYZED: {total_entities}")
print(f"AVERAGE DATA QUALITY SCORE: {avg_quality:.1f}/100")
print(f"ENTITIES WITH INCONSISTENCIES: {with_issues} ({with_issues/total_entities:.1%})")
print("\nBREAKDOWN BY RESOLUTION STATUS:")
print(f"  - High Confidence (Ready to use): {high_confidence} ({high_confidence/total_entities:.1%})")
print(f"  - Good Confidence (Minor issues): {good_confidence} ({good_confidence/total_entities:.1%})")
print(f"  - Needs Review (Moderate issues): {needs_review} ({needs_review/total_entities:.1%})")
print(f"  - Low Confidence (Major issues): {low_confidence} ({low_confidence/total_entities:.1%})")
print()

# STEP 7: EXPORT RESULTS
print("STEP 7: EXPORTING RESULTS")
print("-----------------------")

# Create output directory if it doesn't exist
output_dir = 'entity_resolution_results'
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Created output directory: {output_dir}")

# Filter accurate matches (high confidence, no inconsistencies)
accurate_matches = resolution_df[
    (resolution_df['match_confidence'] >= 0.85) &
    (~resolution_df['has_inconsistencies'])
].copy()

# Filter entities needing review
needs_review_df = resolution_df[
    (resolution_df['match_confidence'] < 0.85) |
    (resolution_df['has_inconsistencies'])
].copy()

# Prepare columns for export
export_columns = [
    'original_entity_name', 'best_match_name', 'match_confidence',
    'website_url', 'industry_codes', 'naics_code', 'description',
    'phone_numbers', 'email_addresses',
    'data_quality_score', 'has_inconsistencies',
    'inconsistency_details', 'resolution_status'
]

# Handle case where some columns might not exist
available_columns = [col for col in export_columns if col in resolution_df.columns]
if len(available_columns) < len(export_columns):
    missing_cols = set(export_columns) - set(available_columns)
    print(f"⚠️ Warning: Some export columns not found in dataset: {missing_cols}")

# If best_match_name is not available, use original_entity_name as fallback
if 'best_match_name' not in available_columns:
    resolution_df['best_match_name'] = resolution_df['original_entity_name']
    available_columns.append('best_match_name')
    print("✓ Created fallback 'best_match_name' column from 'original_entity_name'")

    # Update the filtered dataframes
    accurate_matches['best_match_name'] = accurate_matches['original_entity_name']
    needs_review_df['best_match_name'] = needs_review_df['original_entity_name']

print(f"Exporting {len(accurate_matches)} accurate matches to CSV...")
accurate_matches[available_columns].to_csv(f'{output_dir}/accurate_entity_matches.csv', index=False)

print(f"Exporting {len(needs_review_df)} entities needing review to CSV...")
needs_review_df[available_columns].to_csv(f'{output_dir}/entities_needing_review.csv', index=False)

print(f"Exporting all results ({len(resolution_df)} entities) to CSV...")
resolution_df[available_columns].to_csv(f'{output_dir}/all_entities_with_analysis.csv', index=False)

# Create Excel file with multiple sheets
excel_path = f'{output_dir}/entity_resolution_summary.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    resolution_df[available_columns].to_excel(writer, sheet_name='All_Results', index=False)
    accurate_matches[available_columns].to_excel(writer, sheet_name='Accurate_Matches', index=False)
    needs_review_df[available_columns].to_excel(writer, sheet_name='Needs_Review', index=False)

    # Create a summary sheet
    summary_data = {
        'Metric': [
            'Total Entities Analyzed',
            'Average Data Quality Score',
            'Entities with Inconsistencies',
            'High Confidence Matches',
            'Good Confidence Matches',
            'Entities Needing Review',
            'Low Confidence Matches',
            'Processing Date'
        ],
        'Value': [
            total_entities,
            f"{avg_quality:.1f}/100",
            f"{with_issues} ({with_issues/total_entities:.1%})",
            f"{high_confidence} ({high_confidence/total_entities:.1%})",
            f"{good_confidence} ({good_confidence/total_entities:.1%})",
            f"{needs_review} ({needs_review/total_entities:.1%})",
            f"{low_confidence} ({low_confidence/total_entities:.1%})",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

print(f"✓ Created Excel summary with multiple sheets: {excel_path}\n")

# STEP 8: GENERATE HTML REPORT
print("STEP 8: GENERATING HTML REPORT")
print("---------------------------")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Entity Resolution Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .high-confidence {{ background-color: #dff0d8; border-left-color: #4CAF50; }}
        .good-confidence {{ background-color: #fcf8e3; border-left-color: #f0ad4e; }}
        .needs-review {{ background-color: #f2dede; border-left-color: #d9534f; }}
        .low-confidence {{ background-color: #f2dede; border-left-color: #d9534f; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .inconsistent {{ border-left: 5px solid #dc3232; }}
        h1, h2 {{ color: #333; }}
        .footer {{ margin-top: 30px; color: #666; font-size: 0.9em; }}
        .status-high {{ color: #28a745; font-weight: bold; }}
        .status-good {{ color: #ffc107; font-weight: bold; }}
        .status-needs {{ color: #dc3545; font-weight: bold; }}
        .status-low {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Entity Resolution Quality Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Entities</h3>
            <p style="font-size: 24px; font-weight: bold;">{total_entities}</p>
        </div>
        <div class="summary-card">
            <h3>Average Quality</h3>
            <p style="font-size: 24px; font-weight: bold;">{avg_quality:.1f}/100</p>
        </div>
        <div class="summary-card">
            <h3>Entities with Issues</h3>
            <p style="font-size: 24px; font-weight: bold;">{with_issues} ({with_issues/total_entities:.1%})</p>
        </div>
        <div class="summary-card">
            <h3>High Confidence</h3>
            <p style="font-size: 24px; font-weight: bold;">{high_confidence} ({high_confidence/total_entities:.1%})</p>
        </div>
    </div>

    <h2>Entities Requiring Review</h2>
    <p>Below are entities that need manual review due to low confidence scores or detected inconsistencies:</p>
    <table>
        <thead>
            <tr>
                <th>Original Entity</th>
                <th>Best Match</th>
                <th>Confidence</th>
                <th>Quality Score</th>
                <th>Status</th>
                <th>Inconsistencies</th>
            </tr>
        </thead>
        <tbody>
"""

# Add top 50 entities needing review to the HTML report
review_entities = needs_review_df.nlargest(50, 'data_quality_score')
for _, row in review_entities.iterrows():
    # Determine row class based on status
    status_class = ''
    if row['resolution_status'] == 'high_confidence':
        status_class = 'high-confidence'
    elif row['resolution_status'] == 'good_confidence':
        status_class = 'good-confidence'
    elif row['resolution_status'] == 'needs_review':
        status_class = 'needs-review'
    else:
        status_class = 'low-confidence'

    # Determine status display text
    status_text = ''
    if row['resolution_status'] == 'high_confidence':
        status_text = '<span class="status-high">High Confidence</span>'
    elif row['resolution_status'] == 'good_confidence':
        status_text = '<span class="status-good">Good Confidence</span>'
    elif row['resolution_status'] == 'needs_review':
        status_text = '<span class="status-needs">Needs Review</span>'
    else:
        status_text = '<span class="status-low">Low Confidence</span>'

    # Get entity names with fallbacks
    orig_name = str(row.get('original_entity_name', '')).strip()
    best_match = str(row.get('best_match_name', '')).strip()
    if pd.isna(best_match) or best_match == '' or best_match.lower() in ['nan', 'none']:
        best_match = orig_name

    # Truncate long inconsistency details
    issues = str(row['inconsistency_details']).strip() if pd.notna(row['inconsistency_details']) else 'None'
    if len(issues) > 80:
        issues = issues[:77] + '...'

    html_content += f"""
            <tr class="{status_class}">
                <td>{orig_name}</td>
                <td>{best_match}</td>
                <td>{row['match_confidence']:.2f}</td>
                <td>{row['data_quality_score']}/100</td>
                <td>{status_text}</td>
                <td>{issues}</td>
            </tr>
    """

html_content += """
        </tbody>
    </table>

    <div class="footer">
        <p><strong>Recommendations:</strong></p>
        <ul>
            <li><strong>High Confidence</strong> entities can be used directly in procurement analytics</li>
            <li><strong>Good Confidence</strong> entities should be spot-checked before use</li>
            <li><strong>Needs Review</strong> entities require manual verification before inclusion</li>
            <li><strong>Low Confidence</strong> entities should be reprocessed or excluded from analysis</li>
        </ul>
        <p>For detailed data, see the exported CSV/Excel files in the '{output_dir}' directory.</p>
    </div>
</body>
</html>
"""

html_path = f'{output_dir}/entity_resolution_report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✓ Generated HTML report: {html_path}\n")

# STEP 9: DISPLAY TOP 10 ENTITIES NEEDING REVIEW
print("STEP 9: TOP ENTITIES NEEDING REVIEW")
print("---------------------------------")

# Handle case where no entities need review
if not needs_review_df.empty:
    print("Top 10 entities with highest quality scores that still need review:")
    top_issues = needs_review_df.nlargest(10, 'data_quality_score')[[
        'original_entity_name', 'best_match_name', 'match_confidence',
        'data_quality_score', 'resolution_status', 'inconsistency_details'
    ]]

    # Format the output nicely
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 1000)
    print(top_issues.to_string(index=False))
else:
    print("No entities require review - all entities have high confidence scores and no inconsistencies!")

print("\n" + "="*50)
print("PROCESS COMPLETED SUCCESSFULLY")
print("="*50)
print(f"All results have been saved to the '{output_dir}' directory:")
print(f"  - Accurate matches: {output_dir}/accurate_entity_matches.csv")
print(f"  - Entities needing review: {output_dir}/entities_needing_review.csv")
print(f"  - Complete analysis: {output_dir}/all_entities_with_analysis.csv")
print(f"  - Excel summary: {output_dir}/entity_resolution_summary.xlsx")
print(f"  - HTML report: {output_dir}/entity_resolution_report.html")
print(f"\nNext steps recommendations:")
print("1. Use the accurate matches for immediate procurement analysis")
print("2. Review the 'entities_needing_review.csv' file for manual verification")
print("3. Open the HTML report for a visual overview of data quality")
print(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
