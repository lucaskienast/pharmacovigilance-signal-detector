"""
FAERS Data Downloader

This utility downloads real FDA Adverse Event Reporting System (FAERS) data
for use in the Pharmacovigilance Signal Detector project.

FAERS is a database that contains information on adverse event and medication 
error reports submitted to FDA. The database is designed to support the FDA's 
post-marketing safety surveillance program for drug and therapeutic biologic products.

Data Source: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html


"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import urllib.parse
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Optional, Tuple
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FAERSDataDownloader:
    """
    Downloads and processes real FAERS data from FDA website.
    
    This class handles downloading quarterly FAERS data files, extracting them,
    and converting them into a format suitable for pharmacovigilance analysis.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the FAERS data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # FAERS data base URL (ASCII format for easier processing)
        self.base_url = "https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip"
        
        # Define FAERS file types we need
        self.file_types = {
            'DEMO': 'demographic data',
            'DRUG': 'drug information', 
            'REAC': 'adverse event reactions',
            'OUTC': 'patient outcomes',
            'RPSR': 'report sources',
            'THER': 'drug therapy dates'
        }
        
        print(f"📁 FAERS Data Downloader initialized")
        print(f"📂 Data directory: {self.data_dir.absolute()}")
    
    def get_available_quarters(self, years: Optional[List[int]] = None) -> List[str]:
        """
        Get list of available FAERS data quarters.
        
        Args:
            years: List of years to check (default: last 2 years)
            
        Returns:
            List of available quarter strings (e.g., ['2024q1', '2024q2'])
        """
        if years is None:
            current_year = datetime.now().year
            years = [current_year - 2, current_year - 1, current_year]  # Last 3 years for safety
        
        quarters = []
        current_date = datetime.now()
        
        for year in years:
            for q in [1, 2, 3, 4]:
                # Don't include future quarters or very recent ones (allow 6 month delay)
                quarter_end = datetime(year, q*3, 28)  # End of quarter (approximately)
                if quarter_end <= current_date - timedelta(days=180):  # 6 month delay
                    quarters.append(f"{year}q{q}")
        
        return quarters
    
    def download_quarter_data(self, quarter: str, force_download: bool = False) -> bool:
        """
        Download FAERS data for a specific quarter.
        
        Args:
            quarter: Quarter string (e.g., '2024q1')
            force_download: Force download even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Downloading FAERS data for {quarter}...")
        
        # Create quarter directory
        quarter_dir = self.data_dir / quarter
        quarter_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        zip_file = quarter_dir / f"faers_ascii_{quarter}.zip"
        if zip_file.exists() and not force_download:
            print(f"✅ Data for {quarter} already exists. Use force_download=True to re-download.")
            return True
        
        # Download the zip file
        url = self.base_url.format(quarter=quarter)
        
        try:
            print(f"🌐 Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save the zip file
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {zip_file.name} ({zip_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Extract the zip file
            self._extract_quarter_data(zip_file, quarter_dir)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading {quarter}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error downloading {quarter}: {e}")
            return False
    
    def _extract_quarter_data(self, zip_file: Path, extract_dir: Path) -> None:
        """Extract FAERS zip file and organize data files."""
        
        print(f"📦 Extracting {zip_file.name}...")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✅ Extracted to {extract_dir}")
            
            # List extracted files for verification
            extracted_files = list(extract_dir.glob("*.txt"))
            print(f"📄 Found {len(extracted_files)} data files:")
            for file in sorted(extracted_files):
                print(f"   • {file.name}")
                
        except Exception as e:
            print(f"❌ Error extracting {zip_file}: {e}")
    
    def load_quarter_data(self, quarter: str, file_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load FAERS data files for a quarter into pandas DataFrames.
        
        Args:
            quarter: Quarter string (e.g., '2024q1')
            file_types: List of file types to load (default: ['DEMO', 'DRUG', 'REAC'])
            
        Returns:
            Dictionary of DataFrames by file type
        """
        if file_types is None:
            file_types = ['DEMO', 'DRUG', 'REAC']  # Core files for signal detection
        
        print(f"📊 Loading FAERS data for {quarter}...")
        
        quarter_dir = self.data_dir / quarter
        data = {}
        
        for file_type in file_types:
            # Find the file (format: DEMO24Q1.txt or similar)
            pattern = f"{file_type}{quarter[2:].upper()}.txt"
            
            # Check in main directory first
            files = list(quarter_dir.glob(pattern))
            
            # If not found, check in ASCII subdirectory
            if not files:
                ascii_dir = quarter_dir / "ASCII"
                if ascii_dir.exists():
                    files = list(ascii_dir.glob(pattern))
            
            if not files:
                print(f"⚠️  File {pattern} not found in {quarter_dir} or {quarter_dir}/ASCII")
                continue
            
            file_path = files[0]
            
            try:
                print(f"📖 Loading {file_path.name}...")
                
                # Load with proper encoding and delimiter
                df = pd.read_csv(file_path, 
                               delimiter='$',  # FAERS uses $ as delimiter
                               encoding='utf-8',
                               low_memory=False,
                               na_values=['', 'NULL', 'null'])
                
                print(f"✅ Loaded {file_type}: {len(df):,} rows, {len(df.columns)} columns")
                data[file_type] = df
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return data
    
    def create_signal_detection_dataset(self, quarters: Optional[List[str]] = None, 
                                       top_n_drugs: Optional[int] = None,
                                       drug_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a processed dataset suitable for signal detection analysis.
        
        Args:
            quarters: List of quarters to process (default: latest available)
            top_n_drugs: If specified, only include top N drugs by case count
            drug_filter: If specified, only include these specific drug names
            
        Returns:
            DataFrame with drug-event pairs and case counts
        """
        if quarters is None:
            # Use the most recent quarter available
            available = self.get_available_quarters()
            quarters = [available[-1]] if available else []
        
        if not quarters:
            print("❌ No quarters specified and none available")
            return pd.DataFrame()
        
        print(f"🔬 Creating signal detection dataset from {len(quarters)} quarter(s)...")
        if top_n_drugs:
            print(f"🎯 Will filter to top {top_n_drugs} drugs by case count")
        elif drug_filter:
            print(f"🎯 Will filter to {len(drug_filter)} specified drugs")
        
        all_drug_events = []
        
        for quarter in quarters:
            print(f"\n📅 Processing {quarter}...")
            
            # Download if needed
            if not self.download_quarter_data(quarter):
                continue
            
            # Load core files
            data = self.load_quarter_data(quarter, ['DEMO', 'DRUG', 'REAC'])
            
            if len(data) < 3:
                print(f"⚠️  Incomplete data for {quarter}, skipping...")
                continue
            
            # Process drug-event combinations with optional filtering
            quarter_events = self._process_drug_events(data, drug_filter)
            all_drug_events.append(quarter_events)
        
        if not all_drug_events:
            print("❌ No data processed successfully")
            return pd.DataFrame()
        
        # Combine all quarters
        combined_df = pd.concat(all_drug_events, ignore_index=True)
        
        # Aggregate by drug-event pairs
        signal_dataset = combined_df.groupby(['drug_name', 'adverse_event']).agg({
            'case_count': 'sum',
            'quarter': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()
        
        # Apply top_n_drugs filter if specified
        if top_n_drugs and not drug_filter:
            print(f"🎯 Applying top {top_n_drugs} drugs filter...")
            drug_totals = signal_dataset.groupby('drug_name')['case_count'].sum()
            top_drugs = drug_totals.nlargest(top_n_drugs).index
            signal_dataset = signal_dataset[signal_dataset['drug_name'].isin(top_drugs)]
            print(f"📊 Filtered to {len(signal_dataset):,} drug-event pairs for top {top_n_drugs} drugs")
        
        # Add background statistics
        signal_dataset = self._add_background_statistics(signal_dataset)
        
        # Save processed dataset
        suffix = f"_top{top_n_drugs}" if top_n_drugs else "_filtered" if drug_filter else ""
        output_file = self.data_dir / f"faers_signal_detection_dataset{suffix}.csv"
        signal_dataset.to_csv(output_file, index=False)
        
        print(f"\n✅ Signal detection dataset created!")
        print(f"📊 Total unique drug-event pairs: {len(signal_dataset):,}")
        print(f"💾 Saved to: {output_file}")
        
        return signal_dataset
    
    def _process_drug_events(self, data: Dict[str, pd.DataFrame], 
                           drug_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Process FAERS data to extract drug-event pairs."""
        
        demo_df = data['DEMO']
        drug_df = data['DRUG'] 
        reac_df = data['REAC']
        
        print(f"🔗 Linking demographic, drug, and reaction data...")
        
        # Clean data and handle missing values
        demo_df = demo_df.dropna(subset=['primaryid'])
        drug_df = drug_df.dropna(subset=['primaryid', 'drugname'])
        reac_df = reac_df.dropna(subset=['primaryid', 'pt'])
        
        # Apply drug filter early to reduce memory usage
        if drug_filter:
            print(f"🎯 Applying drug filter for {len(drug_filter)} drugs during processing...")
            # Convert filter to uppercase for matching
            drug_filter_upper = [drug.upper().strip() for drug in drug_filter]
            # Filter drug records early
            drug_df['drugname_clean'] = drug_df['drugname'].str.upper().str.strip()
            drug_df = drug_df[drug_df['drugname_clean'].isin(drug_filter_upper)]
            print(f"📊 Filtered to {len(drug_df):,} drug records (from filtered drugs)")
        
        # Convert primaryid to consistent type
        demo_df['primaryid'] = demo_df['primaryid'].astype(str)
        drug_df['primaryid'] = drug_df['primaryid'].astype(str)
        reac_df['primaryid'] = reac_df['primaryid'].astype(str)
        
        # Use primaryid as the main key (caseid may not be reliable in some quarters)
        print(f"📊 Starting with {len(demo_df):,} cases, {len(drug_df):,} drug records, {len(reac_df):,} reaction records")
        
        # Link reactions to cases using primaryid
        case_reactions = reac_df.merge(demo_df[['primaryid']], 
                                     on='primaryid', how='inner')
        
        # Link drugs to cases using primaryid
        case_drugs = drug_df.merge(demo_df[['primaryid']], 
                                 on='primaryid', how='inner')
        
        print(f"📊 After linking: {len(case_reactions):,} case-reactions, {len(case_drugs):,} case-drugs")
        
        # Create drug-event combinations
        drug_events = case_reactions.merge(case_drugs, 
                                         on='primaryid', 
                                         how='inner')
        
        print(f"📊 Found {len(drug_events):,} drug-event combinations")
        
        # Clean and standardize names
        if 'drugname_clean' in drug_events.columns:
            drug_events['drug_name'] = drug_events['drugname_clean']
        else:
            drug_events['drug_name'] = drug_events['drugname'].str.upper().str.strip()
        drug_events['adverse_event'] = drug_events['pt'].str.upper().str.strip()
        
        # Filter out invalid entries
        drug_events = drug_events.dropna(subset=['drug_name', 'adverse_event'])
        drug_events = drug_events[
            (drug_events['drug_name'] != '') & 
            (drug_events['adverse_event'] != '') &
            (drug_events['drug_name'] != 'UNKNOWN') &
            (drug_events['adverse_event'] != 'UNKNOWN')
        ]
        
        print(f"📊 After cleaning: {len(drug_events):,} valid drug-event combinations")
        
        # Count unique cases per drug-event pair
        drug_event_counts = drug_events.groupby(['drug_name', 'adverse_event']).agg({
            'primaryid': 'nunique'
        }).reset_index()
        
        drug_event_counts.rename(columns={'primaryid': 'case_count'}, inplace=True)
        
        # Add quarter information
        quarter_info = f"2025q2"  # You could extract this from demo_df dates if needed
        drug_event_counts['quarter'] = quarter_info
        
        print(f"✅ Processed {len(drug_event_counts):,} unique drug-event pairs")
        
        return drug_event_counts
    
    def _add_background_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add background frequency statistics for signal detection."""
        
        print("📈 Calculating background statistics...")
        
        # Calculate totals
        total_reports = df['case_count'].sum()
        
        # Drug totals
        drug_totals = df.groupby('drug_name')['case_count'].sum().to_dict()
        df['total_drug_reports'] = df['drug_name'].map(drug_totals)
        
        # Event totals
        event_totals = df.groupby('adverse_event')['case_count'].sum().to_dict()
        df['total_event_reports'] = df['adverse_event'].map(event_totals)
        
        # Total database reports
        df['total_database_reports'] = total_reports
        
        return df
    
    def get_sample_high_signal_pairs(self, dataset: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get drug-event pairs likely to show strong signals for testing.
        
        Args:
            dataset: Processed FAERS dataset
            top_n: Number of top pairs to return
            
        Returns:
            DataFrame with high-signal drug-event pairs
        """
        
        # Filter for pairs with reasonable case counts (not too rare, not too common)
        filtered = dataset[
            (dataset['case_count'] >= 10) &  # At least 10 cases
            (dataset['case_count'] <= 1000) &  # Not overwhelming common
            (dataset['total_drug_reports'] >= 100)  # Drug has reasonable exposure
        ].copy()
        
        # Calculate simple signal score (case count / background rate)
        filtered['signal_score'] = (
            filtered['case_count'] / 
            (filtered['total_event_reports'] / filtered['total_database_reports'])
        )
        
        # Return top candidates
        top_pairs = filtered.nlargest(top_n, 'signal_score')
        
        print(f"🔍 Top {len(top_pairs)} high-signal candidates:")
        for _, row in top_pairs.iterrows():
            print(f"   • {row['drug_name']} → {row['adverse_event']} ({row['case_count']} cases)")
        
        return top_pairs


def download_faers_data(quarters: Optional[List[str]] = None, 
                      data_dir: str = "data",
                      top_n_drugs: Optional[int] = None,
                      drug_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to download and process FAERS data.
    
    Args:
        quarters: List of quarters to download (default: latest available)
        data_dir: Directory to store data
        top_n_drugs: If specified, only include top N drugs by case count
        drug_filter: If specified, only include these specific drug names
        
    Returns:
        Processed dataset ready for signal detection
    """
    
    print("🏥 FAERS Data Download Utility")
    print("=" * 50)
    
    downloader = FAERSDataDownloader(data_dir)
    
    # Get available quarters if none specified
    if quarters is None:
        available = downloader.get_available_quarters()
        if available:
            quarters = [available[-1]]  # Most recent quarter
            print(f"📅 Using most recent quarter: {quarters[0]}")
        else:
            print("❌ No quarters available")
            return pd.DataFrame()
    
    # Create signal detection dataset with optional filtering
    dataset = downloader.create_signal_detection_dataset(quarters, top_n_drugs, drug_filter)
    
    if len(dataset) > 0:
        print(f"\n🎯 Dataset Summary:")
        print(f"   • Total drug-event pairs: {len(dataset):,}")
        print(f"   • Total adverse event cases: {dataset['case_count'].sum():,}")
        print(f"   • Unique drugs: {dataset['drug_name'].nunique():,}")
        print(f"   • Unique adverse events: {dataset['adverse_event'].nunique():,}")
        
        # Show some high-signal examples
        print(f"\n🔍 Sample high-signal pairs for testing:")
        sample_pairs = downloader.get_sample_high_signal_pairs(dataset, 5)
        
    return dataset


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing FAERS Data Downloader...")
    
    # Download latest quarter
    dataset = download_faers_data()
    
    if len(dataset) > 0:
        print(f"\n✅ Successfully downloaded and processed FAERS data!")
        print(f"📁 Data saved to: data/faers_signal_detection_dataset.csv")
        
        # Show sample of the data
        print(f"\n📊 Real FAERS data preview:")
        print(dataset.head())
    else:
        print(f"❌ Failed to download FAERS data")
        print(f"💡 This might be due to network issues or changes in FDA data structure")
        print(f"💡 Please check network connection and try again")