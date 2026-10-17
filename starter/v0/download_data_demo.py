#!/usr/bin/env python3
"""
FAERS Data Download Demonstration

This script demonstrates how to download real FDA FAERS data for the
Pharmacovigilance Signal Detector project.

Run this script to download real adverse event data that will be used
instead of the sample data in the main analysis notebook.

Usage:
    python download_data_demo.py

Requirements:
    - Internet connection
    - pandas, requests libraries (installed via requirements.txt)

Author: Udacity Life Science Agentic AI Course
"""

import sys
from pathlib import Path
import pandas as pd
import shutil

def clean_data_directory(data_dir: Path, file_to_keep: str):
    """Cleans the data directory, removing all files and folders except the specified file."""
    print(f"🧹 Cleaning data directory: {data_dir}")
    if not data_dir.is_dir():
        print("   Directory not found, skipping cleanup.")
        return

    for item in data_dir.iterdir():
        if item.name != file_to_keep:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"   🗑️ Removed directory: {item.name}")
                else:
                    item.unlink()
                    print(f"   🗑️ Removed file: {item.name}")
            except Exception as e:
                print(f"   ❌ Error removing {item.name}: {e}")

def main():
    """Main function to demonstrate FAERS data download."""
    
    print("🏥 FAERS Data Download Demonstration")
    print("=" * 50)
    print()
    
    try:
        # Import the FAERS downloader
        from faers_data_downloader import download_faers_data, FAERSDataDownloader
        
        print("✅ FAERS downloader imported successfully")
        print()
        
        # Show available quarters
        downloader = FAERSDataDownloader()
        available_quarters = downloader.get_available_quarters()
        
        print("📅 Available FAERS quarters:")
        for quarter in available_quarters[-5:]:  # Show last 5 quarters
            print(f"   • {quarter}")
        print()
        
        # Ask user if they want to proceed
        response = input("📥 Download the latest quarter of FAERS data? (y/n): ").lower()
        
        if response in ['y', 'yes']:
            print("\n🔄 Starting download...")
            print("⏳ This may take several minutes depending on your internet connection...")
            print()
            
            # Download latest quarter
            dataset = download_faers_data()
            
            if len(dataset) > 0:
                print("\n🎉 Success! Real FAERS data is now available for analysis.")
                print(f"📊 Downloaded {len(dataset):,} drug-event pairs")
                
                # Filter the dataset to a smaller, more focused size
                print("\n🔪 Filtering dataset for faster performance...")
                
                drugs_to_keep = [
                    "METFORMIN",
                    "WARFARIN",
                    "ASPIRIN",
                    "SIMVASTATIN",
                    "ACETAMINOPHEN"
                ]
                
                original_size = len(dataset)
                dataset = dataset[dataset['drug_name'].isin(drugs_to_keep)].copy()
                new_size = len(dataset)
                
                # Define output path
                output_path = Path("data/faers_signal_detection_dataset.csv")
                output_path.parent.mkdir(exist_ok=True, parents=True)

                # Clean the data directory before saving the new file
                clean_data_directory(output_path.parent, output_path.name)
                
                # Save the filtered dataset
                dataset.to_csv(output_path, index=False)
                
                print(f"   • Original size: {original_size:,} reports")
                print(f"   • New filtered size: {new_size:,} reports")
                reduction = (1 - new_size / original_size) * 100 if original_size > 0 else 0
                print(f"   • Size reduction: {reduction:.2f}%")
                print(f"📁 Filtered data saved to: {output_path}")

                print()
                print("✨ Next steps:")
                print("   1. Open the main Jupyter notebook")
                print("   2. Run the analysis - it will automatically use the real data")
                print("   3. Compare results with the sample data examples")
                
            else:
                print("\n❌ Download failed. The notebook will use sample data instead.")
                print("💡 This could be due to:")
                print("   • Network connectivity issues")
                print("   • Changes in FDA data structure")
                print("   • Temporary server unavailability")
                print()
                print("📚 The sample data is still valuable for learning!")
                
        else:
            print("\n📚 Skipping download. The notebook will use sample data for learning.")
            print("💡 You can run this script anytime to download real data.")
    
    except ImportError as e:
        print(f"❌ Error importing FAERS downloader: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check your internet connection and try again")
    
    print("\n" + "=" * 50)
    print("🔬 Ready to explore pharmacovigilance with AI!")


if __name__ == "__main__":
    main()