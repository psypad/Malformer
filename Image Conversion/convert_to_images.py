"""
Binary to Image Converter for Malware Detection
Optimized for i5-1360P processor with memory-efficient processing
"""

import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import math
import gc

class BinaryToImageConverter:
    def __init__(self, input_dir="dataset/raw", output_dir="images", image_size=256):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.max_workers = min(8, mp.cpu_count())  # Optimal for i5-1360P
        
    def setup_output_dirs(self):
        """Create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/test", exist_ok=True)
        
    def bytes_to_image(self, file_path, output_path, method='grayscale'):
        """
        Convert binary file to image
        Optimized for memory efficiency on your system
        """
        try:
            # Read binary file in chunks to manage memory
            chunk_size = 1024 * 1024  # 1MB chunks
            byte_data = bytearray()
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    byte_data.extend(chunk)
            
            # Convert to numpy array
            byte_array = np.frombuffer(byte_data, dtype=np.uint8)
            
            # Calculate image dimensions
            if len(byte_array) == 0:
                return False
                
            # Trim or pad to make perfect square
            target_size = self.image_size * self.image_size
            
            if len(byte_array) > target_size:
                byte_array = byte_array[:target_size]
            else:
                # Pad with zeros if needed
                padding = target_size - len(byte_array)
                byte_array = np.pad(byte_array, (0, padding), mode='constant')
            
            # Reshape to image
            img_array = byte_array.reshape(self.image_size, self.image_size)
            
            # Save as image
            img = Image.fromarray(img_array, mode='L')
            img.save(output_path)
            
            # Clean up memory
            del byte_data, byte_array, img_array
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error converting {file_path}: {str(e)}")
            return False
    
    def process_file_batch(self, file_batch):
        """Process a batch of files - for multiprocessing"""
        results = []
        for file_info in file_batch:
            input_path, output_path, file_id = file_info
            success = self.bytes_to_image(input_path, output_path)
            results.append((file_id, success))
        return results
    
    def convert_dataset(self, labels_file=None, batch_size=50):
        """
        Convert entire dataset with memory optimization
        """
        print(f"🔄 Converting binary files to {self.image_size}x{self.image_size} images")
        print(f"Using {self.max_workers} CPU cores for parallel processing")
        
        self.setup_output_dirs()
        
        # Find all .bytes files
        bytes_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.bytes'):
                    bytes_files.append(os.path.join(root, file))
        
        if not bytes_files:
            print("❌ No .bytes files found. Please check your dataset path.")
            return
            
        print(f"Found {len(bytes_files)} binary files to convert")
        
        # Load labels if available
        labels_dict = {}
        if labels_file and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            labels_dict = dict(zip(labels_df['Id'], labels_df['Class']))
            print(f"Loaded labels for {len(labels_dict)} files")
        
        # Prepare file batches for processing
        file_batches = []
        batch = []
        
        for i, file_path in enumerate(bytes_files):
            file_id = os.path.basename(file_path).replace('.bytes', '')
            
            # Determine output path based on labels
            if file_id in labels_dict:
                class_name = labels_dict[file_id]
                output_path = f"{self.output_dir}/train/{class_name}_{file_id}.png"
                os.makedirs(f"{self.output_dir}/train", exist_ok=True)
            else:
                output_path = f"{self.output_dir}/test/{file_id}.png"
                os.makedirs(f"{self.output_dir}/test", exist_ok=True)
            
            batch.append((file_path, output_path, file_id))
            
            if len(batch) >= batch_size:
                file_batches.append(batch)
                batch = []
        
        if batch:  # Add remaining files
            file_batches.append(batch)
        
        # Process batches with multiprocessing
        converted_count = 0
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(bytes_files), desc="Converting files") as pbar:
                futures = [executor.submit(self.process_file_batch, batch) 
                          for batch in file_batches]
                
                for future in futures:
                    try:
                        batch_results = future.result()
                        for file_id, success in batch_results:
                            if success:
                                converted_count += 1
                            else:
                                failed_count += 1
                            pbar.update(1)
                    except Exception as e:
                        print(f"Batch processing error: {e}")
                        failed_count += len(file_batches[0])  # Estimate
                        pbar.update(len(file_batches[0]))
        
        print(f"\n✅ Conversion complete!")
        print(f"   Converted: {converted_count} files")
        print(f"   Failed: {failed_count} files")
        print(f"   Output directory: {self.output_dir}")
    
    def create_sample_images(self, num_samples=100):
        """Create sample images for testing without full dataset"""
        print(f"🎨 Creating {num_samples} sample malware images for testing")
        
        self.setup_output_dirs()
        
        # Malware families for labeling
        families = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda']
        
        for i in tqdm(range(num_samples), desc="Creating samples"):
            # Generate synthetic binary data with patterns
            size = self.image_size * self.image_size
            
            # Create different patterns for different families
            family = families[i % len(families)]
            
            if family == 'Ramnit':
                # Create checkerboard pattern
                data = np.random.randint(0, 256, size, dtype=np.uint8)
                data[::2] = data[::2] * 0.3
            elif family == 'Lollipop':
                # Create gradient pattern
                data = np.linspace(0, 255, size, dtype=np.uint8)
            else:
                # Random pattern
                data = np.random.randint(0, 256, size, dtype=np.uint8)
            
            # Reshape and save
            img_array = data.reshape(self.image_size, self.image_size)
            img = Image.fromarray(img_array, mode='L')
            
            output_path = f"{self.output_dir}/train/{family}_sample_{i:04d}.png"
            img.save(output_path)
        
        print(f"✅ Sample images created in {self.output_dir}/train/")

def main():
    converter = BinaryToImageConverter(image_size=128)  # Smaller for efficiency
    
    print("🖼️  BINARY TO IMAGE CONVERTER")
    print("=" * 40)
    
    # Check if we have real data or create samples
    if os.path.exists("dataset/raw") and os.listdir("dataset/raw"):
        print("Found dataset files. Converting...")
        labels_file = "dataset/raw/trainLabels.csv" if os.path.exists("dataset/raw/trainLabels.csv") else None
        converter.convert_dataset(labels_file=labels_file)
    else:
        print("No dataset found. Creating sample images for testing...")
        converter.create_sample_images(200)
    
    print("\n✨ Next step: Run python train_model.py")

if __name__ == "__main__":
    main()
