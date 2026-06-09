import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
from scipy import ndimage
from scipy.signal import savgol_filter, medfilt
from sklearn.cluster import DBSCAN
import pandas as pd

# Configure matplotlib for better display
matplotlib.use('TkAgg')
plt.ion()

# Start total timer
total_start_time = time.time()
print("=== ENHANCED CRYSTAL ANALYSIS WITH FIBER DISTANCE ===")
print(f"Start time: {time.strftime('%H:%M:%S')}")
print("-" * 40)

# Folder containing images
folder_path = r"C:\Users\ravin\OneDrive\Desktop\images\Hexcel-180-150-5C_B"

# List all files in folder
all_files = os.listdir(folder_path)
print("All files in folder:", all_files)

# Filter only image files (case insensitive) and sort them properly
image_files = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
print(f"Found {len(image_files)} images: {image_files[:5]}...")

if len(image_files) == 0:
    print("No images found. Check that your images are directly inside the folder and have proper extensions.")
    exit()

def analyze_image_sequence_stats(folder_path, image_files, sample_size=10):
    """Analyze a sample of images to understand the dataset characteristics"""
    print("Analyzing image sequence characteristics...")
    
    # Sample images evenly throughout the sequence
    indices = np.linspace(0, len(image_files)-1, min(sample_size, len(image_files)), dtype=int)
    
    intensities = []
    sizes = []
    
    for idx in indices:
        img_path = os.path.join(folder_path, image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            intensities.append(img.mean())
            sizes.append(img.shape)
    
    avg_intensity = np.mean(intensities)
    intensity_std = np.std(intensities)
    
    print(f"Average image intensity: {avg_intensity:.1f} ± {intensity_std:.1f}")
    print(f"Image size: {sizes[0] if sizes else 'Unknown'}")
    
    return avg_intensity, intensity_std

def detect_fibers(img):
    """Detect black fiber lines in the image"""
    height, width = img.shape
    
    # Detect black/dark regions (fibers)
    # Use a low threshold to catch dark fibers
    _, fiber_mask = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the mask - remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fiber_mask = cv2.morphologyEx(fiber_mask, cv2.MORPH_OPEN, kernel)
    
    # Find fiber contours
    contours, _ = cv2.findContours(fiber_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fiber_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter for reasonable fiber sizes (should be elongated, but we'll take center points)
        if area > 100:  # Minimum area for a fiber
            # Get the centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                fiber_centers.append((cx, cy))
    
    return fiber_centers, fiber_mask

def detect_crystal_stable_with_fibers(img, reference_stats):
    """Enhanced crystal detection that also returns fiber information"""
    avg_intensity, intensity_std = reference_stats
    
    # Detect fibers first
    fiber_centers, fiber_mask = detect_fibers(img)
    
    # Original crystal detection logic
    height, width = img.shape
    max_reasonable_radius = min(width, height) // 4
    
    candidates = []
    
    # Method 1: Conservative intensity threshold
    thresh_val = avg_intensity + 0.5 * intensity_std
    _, binary1 = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Method 2: Adaptive threshold
    binary2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # Method 3: Top percentile threshold
    percentile_thresh = np.percentile(img, 90)
    _, binary3 = cv2.threshold(img, percentile_thresh, 255, cv2.THRESH_BINARY)
    
    methods = [
        ("Conservative", binary1, 1.0),
        ("Adaptive", binary2, 0.8),
        ("Percentile", binary3, 0.6)
    ]
    
    for method_name, binary, weight in methods:
        # Remove fiber regions from crystal detection
        binary_clean = cv2.bitwise_and(binary, cv2.bitwise_not(fiber_mask))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        cleaned = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 20 or area > max_reasonable_radius**2 * np.pi:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.3:
                radius = np.sqrt(area / np.pi)
                (x, y), _ = cv2.minEnclosingCircle(contour)
                
                # Calculate distance to nearest fiber
                min_fiber_distance = float('inf')
                if fiber_centers:
                    for fx, fy in fiber_centers:
                        dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                        min_fiber_distance = min(min_fiber_distance, dist)
                else:
                    min_fiber_distance = 0
                
                center_dist = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                center_score = 1.0 - (center_dist / (width/2))
                
                score = weight * circularity * center_score
                
                candidates.append({
                    'radius': radius,
                    'center': (int(x), int(y)),
                    'score': score,
                    'method': method_name,
                    'area': area,
                    'circularity': circularity,
                    'fiber_distance': min_fiber_distance
                })
    
    # Choose best candidate
    if candidates:
        best_candidate = max(candidates, key=lambda x: x['score'])
        return (best_candidate['radius'], best_candidate['center'], best_candidate['method'], 
                best_candidate['fiber_distance'], fiber_centers)
    else:
        return 0, None, "None", 0, fiber_centers

# Analyze sequence characteristics
analysis_start = time.time()
reference_stats = analyze_image_sequence_stats(folder_path, image_files)
analysis_time = time.time() - analysis_start
print(f"Sequence analysis completed in {analysis_time:.1f} seconds")

# Process all images with stable detection
print(f"\n=== PROCESSING {len(image_files)} IMAGES ===")
processing_start = time.time()

radii = []
centers = []
methods_used = []
fiber_distances = []
all_fiber_centers = []

for i, file in enumerate(image_files):
    img_path = os.path.join(folder_path, file)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {file}")
        radii.append(0)
        centers.append(None)
        methods_used.append("Failed")
        fiber_distances.append(0)
        all_fiber_centers.append([])
        continue
    
    radius, center, method, fiber_dist, fiber_centers = detect_crystal_stable_with_fibers(img, reference_stats)
    
    radii.append(radius)
    centers.append(center)
    methods_used.append(method)
    fiber_distances.append(fiber_dist)
    all_fiber_centers.append(fiber_centers)
    
    if i % 25 == 0:
        elapsed = time.time() - processing_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(image_files) - i - 1) / rate if rate > 0 else 0
        print(f"Processed {i+1}/{len(image_files)} images - Last radius: {radius:.1f} - "
              f"Fiber dist: {fiber_dist:.1f} - Rate: {rate:.1f} img/sec - ETA: {remaining:.0f}s")

processing_time = time.time() - processing_start
print(f"\nImage processing completed in {processing_time:.1f} seconds")

# Enhanced data cleaning (keeping your original logic)
print(f"\n=== ENHANCED DATA CLEANING ===")
data_start = time.time()

radii = np.array(radii)
fiber_distances = np.array(fiber_distances)
print(f"Raw data: {len(radii)} points, range {radii.min():.1f} - {radii.max():.1f}")

# Step 1: Handle zero values (failed detections)
zero_mask = radii == 0
zero_percent = np.sum(zero_mask) / len(radii) * 100
print(f"Zero detections: {np.sum(zero_mask)} ({zero_percent:.1f}%)")

# Your original interpolation logic
radii_interpolated = radii.copy()
fiber_distances_interpolated = fiber_distances.copy()

if zero_percent < 50:
    non_zero_indices = np.where(radii > 0)[0]
    if len(non_zero_indices) > 1:
        for i in range(len(radii)):
            if radii[i] == 0:
                left_idx = non_zero_indices[non_zero_indices < i]
                right_idx = non_zero_indices[non_zero_indices > i]
                
                if len(left_idx) > 0 and len(right_idx) > 0:
                    left_val = radii[left_idx[-1]]
                    right_val = radii[right_idx[0]]
                    left_dist = fiber_distances[left_idx[-1]]
                    right_dist = fiber_distances[right_idx[0]]
                    radii_interpolated[i] = (left_val + right_val) / 2
                    fiber_distances_interpolated[i] = (left_dist + right_dist) / 2
                elif len(left_idx) > 0:
                    radii_interpolated[i] = radii[left_idx[-1]]
                    fiber_distances_interpolated[i] = fiber_distances[left_idx[-1]]
                elif len(right_idx) > 0:
                    radii_interpolated[i] = radii[right_idx[0]]
                    fiber_distances_interpolated[i] = fiber_distances[right_idx[0]]

# Your original cleaning steps
radii_median = medfilt(radii_interpolated, kernel_size=5)

def remove_outliers_adaptive(data, window=7, threshold=2.0):
    cleaned = data.copy()
    half_window = window // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        local_data = data[start:end]
        local_data = local_data[local_data > 0]
        
        if len(local_data) > 2:
            local_median = np.median(local_data)
            local_mad = np.median(np.abs(local_data - local_median))
            
            if local_mad > 0:
                z_score = abs(data[i] - local_median) / local_mad
                if z_score > threshold:
                    cleaned[i] = local_median
    
    return cleaned

radii_cleaned = remove_outliers_adaptive(radii_median)

if len(radii_cleaned) > 9:
    radii_smoothed = savgol_filter(radii_cleaned, 9, 3)
else:
    radii_smoothed = radii_cleaned

radii_smoothed = np.maximum(radii_smoothed, 0)

def detect_germination_enhanced(radii, min_sustained_frames=5):
    """Enhanced germination detection with multiple criteria"""
    if len(radii) < min_sustained_frames:
        return None
    
    early_data = radii[:min(20, len(radii)//4)]
    baseline = np.percentile(early_data[early_data > 0], 25) if np.any(early_data > 0) else 1.0
    
    growth_threshold = baseline * 1.3
    
    for i in range(len(radii) - min_sustained_frames):
        window = radii[i:i+min_sustained_frames]
        if np.all(window > growth_threshold):
            pre_window = radii[max(0, i-3):i] if i > 0 else [baseline]
            if len(pre_window) > 0 and np.mean(window) > np.mean(pre_window) * 1.2:
                return i
    
    return None

germination_frame = detect_germination_enhanced(radii_smoothed)

# ENHANCED VISUALIZATION WITH FIBER ANALYSIS
time_frames = np.arange(len(radii))
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Keep your original data processing pipeline
axes[0, 0].plot(time_frames, radii, 'b-', alpha=0.5, label='Raw data', linewidth=1)
axes[0, 0].plot(time_frames, radii_interpolated, 'c-', alpha=0.7, label='Interpolated', linewidth=1)
axes[0, 0].plot(time_frames, radii_median, 'g-', label='Median filtered', linewidth=1.5)
axes[0, 0].plot(time_frames, radii_cleaned, 'm-', label='Outliers removed', linewidth=1.5)
axes[0, 0].plot(time_frames, radii_smoothed, 'r-', label='Final smoothed', linewidth=2)
axes[0, 0].set_xlabel("Time (frames)")
axes[0, 0].set_ylabel("Crystal Radius (pixels)")
axes[0, 0].set_title("Enhanced Data Processing Pipeline")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: NEW - Crystal Growth vs Distance from Fibers
valid_mask = (radii_smoothed > 0) & (fiber_distances_interpolated >= 0)
if np.any(valid_mask):
    # Color points by time
    scatter = axes[0, 1].scatter(fiber_distances_interpolated[valid_mask], 
                                radii_smoothed[valid_mask], 
                                c=time_frames[valid_mask], 
                                cmap='viridis', alpha=0.7, s=30)
    
    # Add trend line
    valid_fibers = fiber_distances_interpolated[valid_mask]
    valid_radii = radii_smoothed[valid_mask]
    if len(valid_fibers) > 1:
        z = np.polyfit(valid_fibers, valid_radii, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_fibers.min(), valid_fibers.max(), 100)
        axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')
    
    axes[0, 1].set_xlabel("Distance from Nearest Fiber (pixels)")
    axes[0, 1].set_ylabel("Crystal Radius (pixels)")
    axes[0, 1].set_title("Crystal Size vs Distance from Fibers")
    axes[0, 1].legend()
    plt.colorbar(scatter, ax=axes[0, 1], label='Time Frame')
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: NEW - Fiber Distance Over Time
axes[1, 0].plot(time_frames, fiber_distances_interpolated, 'orange', linewidth=2, alpha=0.8, label='Distance to fiber')
axes[1, 0].set_xlabel("Time (frames)")
axes[1, 0].set_ylabel("Distance to Nearest Fiber (pixels)")
axes[1, 0].set_title("Crystal-Fiber Distance Evolution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Add average distance line
avg_distance = np.mean(fiber_distances_interpolated[fiber_distances_interpolated > 0])
axes[1, 0].axhline(y=avg_distance, color='red', linestyle=':', alpha=0.7, 
                  label=f'Avg distance: {avg_distance:.1f}px')
axes[1, 0].legend()

# Plot 4: Enhanced final result with germination and fiber info
axes[1, 1].plot(time_frames, radii_smoothed, 'r-', linewidth=3, label='Crystal radius')

if germination_frame is not None:
    axes[1, 1].axvline(x=germination_frame, color='green', linestyle='--', linewidth=2,
                      label=f'Germination (frame {germination_frame})')
    axes[1, 1].plot(germination_frame, radii_smoothed[germination_frame], 'go', markersize=10)

# Add fiber distance as secondary y-axis
ax2 = axes[1, 1].twinx()
ax2.plot(time_frames, fiber_distances_interpolated, 'orange', alpha=0.6, linewidth=1, label='Fiber distance')
ax2.set_ylabel('Distance to Fiber (pixels)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

axes[1, 1].set_xlabel("Time (frames)")
axes[1, 1].set_ylabel("Crystal Radius (pixels)")
axes[1, 1].set_title("Final Crystal Growth Analysis with Fiber Distance")
axes[1, 1].legend(loc='upper left')
ax2.legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)
plt.pause(1)

# Create second figure for detailed fiber analysis
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# Fiber analysis plots
if np.any(valid_mask):
    # Plot 1: Growth rate vs fiber distance
    if len(radii_smoothed) > 1:
        growth_rate = np.gradient(radii_smoothed)
        axes2[0, 0].scatter(fiber_distances_interpolated[valid_mask], growth_rate[valid_mask], 
                           c=time_frames[valid_mask], cmap='plasma', alpha=0.7)
        axes2[0, 0].set_xlabel("Distance from Fiber (pixels)")
        axes2[0, 0].set_ylabel("Growth Rate (pixels/frame)")
        axes2[0, 0].set_title("Growth Rate vs Fiber Distance")
        axes2[0, 0].grid(True, alpha=0.3)
        axes2[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Plot 2: Crystal size distribution by distance bins
    distance_bins = np.linspace(0, np.max(fiber_distances_interpolated[valid_mask]), 5)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    
    mean_radii = []
    std_radii = []
    
    for i in range(len(distance_bins)-1):
        mask = ((fiber_distances_interpolated >= distance_bins[i]) & 
                (fiber_distances_interpolated < distance_bins[i+1]) & 
                valid_mask)
        if np.any(mask):
            mean_radii.append(np.mean(radii_smoothed[mask]))
            std_radii.append(np.std(radii_smoothed[mask]))
        else:
            mean_radii.append(0)
            std_radii.append(0)
    
    axes2[0, 1].errorbar(bin_centers, mean_radii, yerr=std_radii, 
                        marker='o', linewidth=2, markersize=8, capsize=5)
    axes2[0, 1].set_xlabel("Distance from Fiber (pixels)")
    axes2[0, 1].set_ylabel("Average Crystal Radius (pixels)")
    axes2[0, 1].set_title("Crystal Size vs Fiber Distance (Binned)")
    axes2[0, 1].grid(True, alpha=0.3)

    # Plot 3: Time evolution of fiber-crystal relationship
    early_frames = time_frames[:len(time_frames)//3]
    late_frames = time_frames[2*len(time_frames)//3:]
    
    early_mask = np.isin(time_frames, early_frames) & valid_mask
    late_mask = np.isin(time_frames, late_frames) & valid_mask
    
    if np.any(early_mask):
        axes2[1, 0].scatter(fiber_distances_interpolated[early_mask], radii_smoothed[early_mask], 
                           alpha=0.6, label='Early frames', color='blue', s=30)
    if np.any(late_mask):
        axes2[1, 0].scatter(fiber_distances_interpolated[late_mask], radii_smoothed[late_mask], 
                           alpha=0.6, label='Late frames', color='red', s=30)
    
    axes2[1, 0].set_xlabel("Distance from Fiber (pixels)")
    axes2[1, 0].set_ylabel("Crystal Radius (pixels)")
    axes2[1, 0].set_title("Early vs Late Growth Pattern")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    # Plot 4: Crystal detection validation - show sample image with detections
    # Use middle frame for visualization
    mid_frame = len(image_files) // 2
    sample_img_path = os.path.join(folder_path, image_files[mid_frame])
    sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
    
    if sample_img is not None and mid_frame < len(all_fiber_centers):
        axes2[1, 1].imshow(sample_img, cmap='gray')
        
        # Draw fiber centers
        fiber_centers_sample = all_fiber_centers[mid_frame]
        for fx, fy in fiber_centers_sample:
            axes2[1, 1].plot(fx, fy, 'bs', markersize=8, markerfacecolor='blue', 
                           alpha=0.7, label='Fiber' if (fx, fy) == fiber_centers_sample[0] else "")
        
        # Draw crystal
        if centers[mid_frame] is not None:
            cx, cy = centers[mid_frame]
            radius = radii_smoothed[mid_frame]
            circle = plt.Circle((cx, cy), radius, fill=False, color='red', linewidth=2)
            axes2[1, 1].add_patch(circle)
            axes2[1, 1].plot(cx, cy, 'ro', markersize=6)
            
            # Draw line to nearest fiber
            if fiber_centers_sample:
                min_dist = float('inf')
                nearest_fiber = None
                for fx, fy in fiber_centers_sample:
                    dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_fiber = (fx, fy)
                
                if nearest_fiber:
                    axes2[1, 1].plot([cx, nearest_fiber[0]], [cy, nearest_fiber[1]], 
                                   'g--', alpha=0.8, linewidth=2, label='Distance line')
        
        axes2[1, 1].set_title(f'Detection Validation - Frame {mid_frame}')
        axes2[1, 1].legend()
        axes2[1, 1].axis('off')

plt.tight_layout()
plt.show(block=True)

data_time = time.time() - data_start

# Enhanced results with fiber analysis
print(f"\n=== ENHANCED ANALYSIS RESULTS WITH FIBER DISTANCE ===")
print(f"Total frames analyzed: {len(radii)}")
print(f"Data quality: {100-zero_percent:.1f}% successful detections")
print(f"Final radius range: {radii_smoothed.min():.1f} - {radii_smoothed.max():.1f} pixels")
print(f"Average distance to fibers: {np.mean(fiber_distances_interpolated[fiber_distances_interpolated > 0]):.1f} pixels")

if germination_frame is not None:
    print(f"\nGermination analysis:")
    print(f"  Frame: {germination_frame}")
    print(f"  Radius at germination: {radii_smoothed[germination_frame]:.1f} pixels")
    print(f"  Distance to fiber at germination: {fiber_distances_interpolated[germination_frame]:.1f} pixels")

# Correlation analysis
valid_data_mask = (radii_smoothed > 0) & (fiber_distances_interpolated >= 0)
if np.sum(valid_data_mask) > 2:
    correlation = np.corrcoef(radii_smoothed[valid_data_mask], 
                             fiber_distances_interpolated[valid_data_mask])[0, 1]
    print(f"\nCorrelation between crystal size and fiber distance: {correlation:.3f}")

# Save enhanced results
save_start = time.time()
results_file = os.path.join(folder_path, 'enhanced_crystal_fiber_analysis.txt')
with open(results_file, 'w') as f:
    f.write("Enhanced Crystal Growth Analysis with Fiber Distance\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {len(radii)} frames\n")
    f.write(f"Detection success rate: {100-zero_percent:.1f}%\n")
    f.write(f"Germination frame: {germination_frame}\n")
    f.write(f"Final radius: {radii_smoothed[-1]:.2f} pixels\n")
    f.write(f"Average fiber distance: {np.mean(fiber_distances_interpolated[fiber_distances_interpolated > 0]):.2f} pixels\n")
    if np.sum(valid_data_mask) > 2:
        f.write(f"Size-distance correlation: {correlation:.3f}\n")
    f.write("\nFrame\tRadius\tFiber_Distance\tMethod\n")
    for i, (rad, dist, method) in enumerate(zip(radii_smoothed, fiber_distances_interpolated, methods_used)):
        f.write(f"{i}\t{rad:.2f}\t{dist:.2f}\t{method}\n")

save_time = time.time() - save_start
total_time = time.time() - total_start_time

print(f"\n" + "="*60)
print(f"=== TIMING SUMMARY ===")
print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"Processing rate: {len(image_files)/processing_time:.1f} images/second")
print(f"Results saved to: {results_file}")
print("="*60)
print("Enhanced analysis with fiber distance complete!")