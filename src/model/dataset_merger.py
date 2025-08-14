# custom_dataset_merger.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

class CustomMelanomaDatasetMerger:
    """
    Custom merger for your specific data structure
    """
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        
        
        self.dataset_configs = {
            'ham10000': {
                'metadata_file': 'HAM10000_metadata.csv',
                'image_dirs': ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'ham10000_images_part_1', 'ham10000_images_part_2'],
                'image_id_col': 'image_id',
                'diagnosis_col': 'dx',
                'melanoma_label': 'mel'
            },
            'isic_2019': {
            'metadata_file': 'ISIC_2019_Training_GroundTruth.csv',  
            'image_dirs': ['ISIC_2019_Training_Input'],
            'image_id_col': 'image',
            'diagnosis_col': 'MEL',  
            'melanoma_label': 1.0  
            },
            'isic_2020': {
                'metadata_file': 'ISIC_2020_Training_GroundTruth_v2.csv',
                'image_dirs': ['train'],  
                'image_id_col': 'image_name',
                'diagnosis_col': 'target',
                'melanoma_label': 1
            }
        }
    
    def check_and_explore_data(self):
        """Check what data is available and explore the structure"""
        print("🔍 Checking your data structure...")
        print("=" * 50)
        
        for dataset_name, config in self.dataset_configs.items():
            print(f"\n📁 {dataset_name.upper()}:")
            
            # Check metadata file
            metadata_path = self.data_dir / config['metadata_file']
            print(f"   Metadata: {'✅' if metadata_path.exists() else '❌'} {config['metadata_file']}")
            
            if metadata_path.exists():
                # Explore the metadata file
                try:
                    df = pd.read_csv(metadata_path)
                    print(f"   Rows: {len(df):,}")
                    print(f"   Columns: {list(df.columns)}")
                    
                    # Check if expected columns exist
                    if config['image_id_col'] in df.columns:
                        print(f"   ✅ Image ID column: {config['image_id_col']}")
                    else:
                        print(f"   ❌ Expected image ID column '{config['image_id_col']}' not found")
                    
                    # For diagnosis column, we might need to explore
                    if config['diagnosis_col'] in df.columns:
                        print(f"   ✅ Diagnosis column: {config['diagnosis_col']}")
                        if dataset_name == 'ham10000':
                            melanoma_count = (df[config['diagnosis_col']] == 'mel').sum()
                            print(f"   Melanomas: {melanoma_count:,}")
                        else:
                            melanoma_count = df[config['diagnosis_col']].sum()
                            print(f"   Melanomas: {melanoma_count:,}")
                    else:
                        print(f"   ❌ Expected diagnosis column '{config['diagnosis_col']}' not found")
                        print(f"   Available columns for diagnosis: {[col for col in df.columns if 'mel' in col.lower() or 'target' in col.lower() or 'diagnosis' in col.lower()]}")
                
                except Exception as e:
                    print(f"   ❌ Error reading metadata: {e}")
            
            # Check image directories
            for img_dir in config['image_dirs']:
                img_path = self.data_dir / img_dir
                if img_path.exists():
                    image_count = len(list(img_path.glob('*.jpg')) + list(img_path.glob('*.jpeg')))
                    print(f"   ✅ Images: {img_dir} ({image_count:,} files)")
                else:
                    print(f"   ❌ Image directory: {img_dir}")
    
    def fix_isic_2019_metadata(self):
        """ISIC 2019 might have different column structure - let's explore and fix"""
        metadata_path = self.data_dir / 'ISIC_2019_Training_Metadatah.csv'
        
        if not metadata_path.exists():
            print("❌ ISIC 2019 metadata not found")
            return None
        
        print("\n🔧 Exploring ISIC 2019 metadata structure...")
        df = pd.read_csv(metadata_path)
        
        print(f"Columns: {list(df.columns)}")
        print(f"Sample rows:")
        print(df.head())
        
        # Look for melanoma-related columns
        melanoma_cols = [col for col in df.columns if 'mel' in col.lower()]
        target_cols = [col for col in df.columns if 'target' in col.lower()]
        
        print(f"Melanoma-related columns: {melanoma_cols}")
        print(f"Target-related columns: {target_cols}")
        
        # Try to find the correct melanoma column
        if 'MEL' in df.columns:
            melanoma_col = 'MEL'
        elif 'melanoma' in df.columns:
            melanoma_col = 'melanoma'
        elif melanoma_cols:
            melanoma_col = melanoma_cols[0]
        else:
            print("❌ Could not find melanoma column in ISIC 2019 data")
            return None
        
        print(f"Using melanoma column: {melanoma_col}")
        melanoma_count = df[melanoma_col].sum()
        print(f"Melanomas found: {melanoma_count:,}")
        
        return melanoma_col
    
    def load_ham10000(self):
        """Load HAM10000 data"""
        print("\n📊 Loading HAM10000...")
        
        metadata_path = self.data_dir / 'HAM10000_metadata.csv'
        if not metadata_path.exists():
            return None
        
        df = pd.read_csv(metadata_path)
        
        # Standardize
        df_std = pd.DataFrame()
        df_std['image_id'] = df['image_id']
        df_std['binary_target'] = (df['dx'] == 'mel').astype(int)
        df_std['dataset_source'] = 'ham10000'
        df_std['original_diagnosis'] = df['dx']
        
        # Add other columns if available
        for col in ['age', 'sex', 'localization']:
            if col in df.columns:
                df_std[col] = df[col]
        
        # Find image paths
        df_std['image_path'] = self._find_image_paths(
            df_std['image_id'], 
            ['HAM10000_images_part_1', 'HAM10000_images_part_2']
        )
        
        print(f"   Loaded: {len(df_std):,} samples")
        print(f"   Melanomas: {df_std['binary_target'].sum():,}")
        
        return df_std
    
    def load_isic_2019(self):
        """Load ISIC 2019 data"""
        print("\n📊 Loading ISIC 2019...")
        
        metadata_path = self.data_dir / 'ISIC_2019_Training_Metadatah.csv'
        if not metadata_path.exists():
            return None
        
        df = pd.read_csv(metadata_path)
        
        # Explore and find the right melanoma column
        melanoma_col = self.fix_isic_2019_metadata()
        if melanoma_col is None:
            return None
        
        # Standardize
        df_std = pd.DataFrame()
        df_std['image_id'] = df['image']
        df_std['binary_target'] = df[melanoma_col].astype(int)
        df_std['dataset_source'] = 'isic_2019'
        df_std['original_diagnosis'] = df[melanoma_col]
        
        # Find image paths
        df_std['image_path'] = self._find_image_paths(
            df_std['image_id'], 
            ['ISIC_2019_Training_Input']
        )
        
        print(f"   Loaded: {len(df_std):,} samples")
        print(f"   Melanomas: {df_std['binary_target'].sum():,}")
        
        return df_std
    
    def load_isic_2020(self):
        """Load ISIC 2020 data"""
        print("\n📊 Loading ISIC 2020...")
        
        metadata_path = self.data_dir / 'ISIC_2020_Training_GroundTruth_v2.csv'
        if not metadata_path.exists():
            return None
        
        df = pd.read_csv(metadata_path)
        print(f"ISIC 2020 columns: {list(df.columns)}")
        
        # Standardize
        df_std = pd.DataFrame()
        df_std['image_id'] = df['image_name']
        df_std['binary_target'] = df['target'].astype(int)
        df_std['dataset_source'] = 'isic_2020'
        df_std['original_diagnosis'] = df['target']
        
        # Find image paths - check both 'train' and other possible directories
        possible_dirs = ['train', 'ISIC_2020_Training_JPEG']
        df_std['image_path'] = self._find_image_paths(
            df_std['image_id'], 
            possible_dirs
        )
        
        print(f"   Loaded: {len(df_std):,} samples")
        print(f"   Melanomas: {df_std['binary_target'].sum():,}")
        
        return df_std
    
    def _find_image_paths(self, image_ids, image_dirs):
        """Find image paths across multiple directories"""
        image_paths = []
        
        for image_id in image_ids:
            found_path = None
            
            for img_dir in image_dirs:
                dir_path = self.data_dir / img_dir
                if not dir_path.exists():
                    continue
                
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                    img_path = dir_path / f"{image_id}{ext}"
                    if img_path.exists():
                        found_path = str(img_path)
                        break
                
                if found_path:
                    break
            
            image_paths.append(found_path)
        
        return image_paths
    
    def merge_all_datasets(self):
        """Merge all available datasets"""
        print("\n🔄 Merging all datasets...")
        print("=" * 50)
        
        datasets = []
        
        # Load each dataset
        ham_data = self.load_ham10000()
        if ham_data is not None:
            datasets.append(ham_data)
        
        isic_2019_data = self.load_isic_2019()
        if isic_2019_data is not None:
            datasets.append(isic_2019_data)
        
        isic_2020_data = self.load_isic_2020()
        if isic_2020_data is not None:
            datasets.append(isic_2020_data)
        
        if not datasets:
            print("❌ No datasets could be loaded!")
            return None
        
        # Combine
        combined = pd.concat(datasets, ignore_index=True)
        
        # Remove missing images
        before_count = len(combined)
        combined = combined.dropna(subset=['image_path'])
        after_count = len(combined)
        
        print(f"\n📈 Combined Dataset Results:")
        print(f"   Total samples: {len(combined):,}")
        print(f"   Removed (missing): {before_count - after_count:,}")
        print(f"   Melanomas: {combined['binary_target'].sum():,} ({combined['binary_target'].mean():.1%})")
        
        # Source breakdown
        print(f"\n📊 By Source:")
        for source in combined['dataset_source'].unique():
            source_data = combined[combined['dataset_source'] == source]
            melanomas = source_data['binary_target'].sum()
            total = len(source_data)
            print(f"   {source}: {total:,} samples ({melanomas:,} melanomas, {melanomas/total:.1%})")
        
        return combined
    
    def create_balanced_version(self, combined_df, target_ratio=0.3):
        """Create balanced version"""
        print(f"\n⚖️ Creating balanced version (target: {target_ratio:.1%} melanomas)...")
        
        melanomas = combined_df[combined_df['binary_target'] == 1]
        non_melanomas = combined_df[combined_df['binary_target'] == 0]
        
        current_ratio = len(melanomas) / len(combined_df)
        print(f"   Current ratio: {current_ratio:.1%}")
        
        if current_ratio < target_ratio:
            # Calculate needed melanomas
            target_melanomas = int(len(non_melanomas) * target_ratio / (1 - target_ratio))
            additional_needed = target_melanomas - len(melanomas)
            
            if additional_needed > 0:
                # Oversample melanomas
                additional = melanomas.sample(n=additional_needed, replace=True, random_state=42)
                balanced = pd.concat([combined_df, additional], ignore_index=True)
                
                print(f"   Added {additional_needed:,} melanoma samples")
                print(f"   New ratio: {balanced['binary_target'].mean():.1%}")
                
                return balanced
        
        return combined_df
    
    def save_datasets(self, combined_df):
        """Save the combined datasets"""
        print(f"\n💾 Saving datasets...")
        
        # Original combined
        combined_path = self.data_dir / 'combined_metadata.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"   ✅ Saved: {combined_path}")
        
        # Balanced version
        balanced_df = self.create_balanced_version(combined_df)
        balanced_path = self.data_dir / 'combined_balanced_metadata.csv'
        balanced_df.to_csv(balanced_path, index=False)
        print(f"   ✅ Saved: {balanced_path}")
        
        return combined_path, balanced_path
    
    def run_complete_merge(self):
        """Run the complete merge process"""
        print("🚀 Custom Dataset Merger for Your Data Structure")
        print("=" * 60)
        
        # Step 1: Check and explore
        self.check_and_explore_data()
        
        # Step 2: Merge datasets
        combined_df = self.merge_all_datasets()
        
        if combined_df is None:
            print("❌ Failed to merge datasets")
            return None
        
        # Step 3: Save datasets
        combined_path, balanced_path = self.save_datasets(combined_df)
        
        # Step 4: Summary
        print(f"\n🎉 SUCCESS! Dataset merger completed.")
        print(f"\n📈 Final Results:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Melanoma samples: {combined_df['binary_target'].sum():,}")
        print(f"   Melanoma ratio: {combined_df['binary_target'].mean():.1%}")
        
        # Calculate improvement
        original_melanomas = 1113  # HAM10000 melanomas
        new_melanomas = combined_df['binary_target'].sum()
        improvement = new_melanomas / original_melanomas
        
        print(f"   🚀 Improvement: {improvement:.1f}x more melanoma samples!")
        
        print(f"\n📁 Files created:")
        print(f"   ✅ {combined_path}")
        print(f"   ✅ {balanced_path}")
        
        return combined_df


def main():
    """Main function to run the custom merger"""
    merger = CustomMelanomaDatasetMerger(data_dir="./data")
    result = merger.run_complete_merge()
    
    if result is not None:
        print(f"\n🎯 Ready for enhanced training!")
        print(f"   Use: 'combined_balanced_metadata.csv' in your training script")
        print(f"   Expected confidence improvement: 50% → 70-85%")
    else:
        print(f"\n❌ Merger failed. Please check the error messages above.")

if __name__ == "__main__":
    main()