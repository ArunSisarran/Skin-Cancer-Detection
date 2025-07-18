import pandas as pd
from pathlib import Path
from PIL import Image


def create_binary_dataset():
    # Load metadata
    df = pd.read_csv('data/HAM10000_metadata.csv')

    # Create binary target: 1 for melanoma, 0 for others
    df['binary_target'] = (df['dx'] == 'mel').astype(int)

    print("Binary class distribution:")
    print(df['binary_target'].value_counts())
    print(f"Melanoma percentage: {df['binary_target'].mean():.2%}")

    # Save the processed metadata
    df.to_csv('data/binary_metadata.csv', index=False)

    return df


def check_images():
    df = pd.read_csv('data/binary_metadata.csv')

    # Check if image directories exist
    img_dirs = ['data/HAM10000_images_part_1', 'data/HAM10000_images_part_2']

    missing_images = []
    for idx, row in df.head(10).iterrows():  # Check first 10
        image_id = row['image_id']
        image_found = False

        for img_dir in img_dirs:
            img_path = Path(f"{img_dir}/{image_id}.jpg")
            if img_path.exists():
                image_found = True
                # Try to load the image
                try:
                    img = Image.open(img_path)
                    print(f"✓ {image_id}: {img.size}")
                except Exception as e:
                    print(f"✗ {image_id}: Error loading - {e}")
                break

        if not image_found:
            missing_images.append(image_id)
            print(f"✗ {image_id}: File not found")
    return missing_images


if __name__ == "__main__":
    binary = create_binary_dataset()
    missing = check_images()
