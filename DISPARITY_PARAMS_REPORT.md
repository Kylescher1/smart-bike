# Disparity Parameters Investigation Report

## Summary
Out of **42 disparity parameters** defined in `config_setup.py` (lines 87-140), **18 parameters** are now active and working. The remaining **24 parameters** are still inactive.

**Last Updated:** After implementing WLS filtering, morphological operations, and making near/far cutoffs relative

---

## ✅ Parameters That ARE Used (18 total)

### Core Stereo Block Matcher Parameters (9 parameters)
Located in: `src/hal/VISION/VISION.py` lines 105-115

| Parameter | Purpose | Used In |
|-----------|---------|---------|
| `minDisparity` | Minimum possible disparity value | `cv2.StereoSGBM_create()` |
| `numDisparitiesK` | Multiplier for number of disparities (computed as `16 * numDisparitiesK`) | Stereo matcher initialization |
| `numDisparities` | Direct number of disparities (currently redundant with numDisparitiesK) | Config storage |
| `blockSize` | Matched block size (must be odd) | `cv2.StereoSGBM_create()` |
| `preFilterCap` | Truncation value for prefiltered image pixels | `cv2.StereoSGBM_create()` |
| `uniquenessRatio` | Margin in percentage for best match vs second best | `cv2.StereoSGBM_create()` |
| `speckleWindowSize` | Maximum size of smooth disparity regions | `cv2.StereoSGBM_create()` |
| `speckleRange` | Maximum disparity variation within speckle region | `cv2.StereoSGBM_create()` |
| `disp12MaxDiff` | Maximum allowed difference in left-right disparity check | `cv2.StereoSGBM_create()` |

### Pre-processing & Filtering Parameters (4 parameters)
Located in: `src/hal/VISION/depth_processing.py`

| Parameter | Purpose | Implementation |
|-----------|---------|----------------|
| `downSample` | Reduce image size by X% (0-100). Higher = smaller images, faster processing | Resizes frames before rectification (lines 179-190) |
| `crop` | Crop X pixels from all edges after rectification | Crops rectified images (lines 100-107) |
| `nearCutoff` | ✨ **RELATIVE**: Cut off nearest X% of depth range (0-100%) | Filters depth map using percentiles (lines 157-169) |
| `farCutoff` | ✨ **RELATIVE**: Cut off farthest X% of depth range (0-100%) | Filters depth map using percentiles (lines 172-174) |

### Morphological Filtering (2 parameters) ✨ NEW
Located in: `src/hal/VISION/depth_processing.py` lines 138-144

| Parameter | Purpose | Implementation |
|-----------|---------|----------------|
| `useMorph` | Enable morphological closing operation | Toggle for disparity hole-filling |
| `morphIter` | Number of morphological iterations (1-20) | Applied to disparity map to fill holes and smooth |

### WLS Disparity Refinement (3 parameters) ✨ NEW
Located in: `src/hal/VISION/depth_processing.py` lines 62-75, 121-132

| Parameter | Purpose | Implementation |
|-----------|---------|----------------|
| `useWLS` | Enable Weighted Least Squares disparity filtering | Creates WLS filter with left+right matchers |
| `wlsLambda` | WLS filter regularization strength (0-10000) | Higher = smoother but less detail |
| `wlsSigma` | WLS filter edge-aware sigma (0.5-5.0) | Controls edge preservation |

---

## ❌ Parameters That Do NOTHING (24 total)

These parameters are defined in the config but **never used** in the processing pipeline:

### Pre-processing & Scaling (2 parameters remaining)
| Parameter | Intended Purpose | Status |
|-----------|------------------|--------|
| `medianBlurK` | Median blur kernel size | **UNUSED** |
| `farEnhance` | Far region enhancement | **UNUSED** |

### Bilateral Smoothing (2 parameters)
| Parameter | Intended Purpose | Status |
|-----------|------------------|--------|
| `useBilateral` | Enable bilateral filtering | **UNUSED** (only in BOOLEAN_FIELDS) |
| `bilateralStrength` | Bilateral filter strength | **UNUSED** |

### Object Detection Thresholds (3 parameters)
| Parameter | Intended Purpose | Status |
|-----------|------------------|--------|
| `objectThresholdMM` | Object detection threshold in mm | **UNUSED** |
| `wsSigma` | Watershed sigma | **UNUSED** |
| `wsMinArea` | Watershed minimum area | **UNUSED** |

### Edge Enhancement (7 parameters)
| Parameter | Intended Purpose | Status |
|-----------|------------------|--------|
| `edgeEqualize` | Enable histogram equalization on edges | **UNUSED** (only in BOOLEAN_FIELDS) |
| `edgeBilateralD` | Edge bilateral filter diameter | **UNUSED** |
| `edgeBilateralSigma` | Edge bilateral filter sigma | **UNUSED** |
| `edgeCannyKLow` | Canny edge low threshold multiplier | **UNUSED** |
| `edgeCannyKHigh` | Canny edge high threshold multiplier | **UNUSED** |
| `edgeUseScharr` | Use Scharr gradient vs Sobel | **UNUSED** (only in BOOLEAN_FIELDS) |

### Color Segmentation (8 parameters)
| Parameter | Intended Purpose | Status |
|-----------|------------------|--------|
| `colorFocusMM` | Color segmentation focus distance in mm | **UNUSED** |
| `colorSpanMM` | Color segmentation span in mm | **UNUSED** |
| `segMode` | Segmentation mode (0-3) | **UNUSED** |
| `kmK` | K-means K clusters | **UNUSED** |
| `kmSpatialX100` | K-means spatial weight (x100) | **UNUSED** |
| `rgTau` | Region growing tau parameter | **UNUSED** |
| `rgSeedStep` | Region growing seed step | **UNUSED** |

---

## Current Processing Pipeline

The actual depth processing in `src/hal/VISION/depth_processing.py`:

1. **Downsample** frames if `downSample > 0` (reduces size by X%, faster processing)
2. **Rectify** images using calibration maps (`map_x`, `map_y`)
3. **Crop** rectified images if `crop > 0` (removes X pixels from each edge)
4. **Convert to grayscale**
5. **Compute disparity** using the stereo matcher (with the 9 core parameters)
6. ✨ **Apply morphological filtering** if `useMorph = True` (fills holes first with simple closing operation)
7. ✨ **Apply WLS filtering** if `useWLS = True` (final edge-aware refinement using left+right matchers)
8. **Convert disparity to depth** using Q matrix
9. ✨ **Filter depth by relative near/far cutoffs** (percentage-based, adapts to scene depth range)

**Note:** WLS is applied as the final refinement step after morphological filtering for best results.

**Additional filtering (bilateral) and segmentation are still not implemented.**

---

## Recommendations

1. **Remove unused parameters** from `calibrate_disparity.py` trackbars to reduce UI complexity
2. **OR implement the missing processing steps** in `depth_processing.py`:
   - WLS filtering (using `cv2.ximgproc.createDisparityWLSFilter()`)
   - Morphological operations
   - Bilateral filtering
   - Edge enhancement
   - Median blur pre-processing
3. **Document** which parameters are "planned for future use" vs "legacy/deprecated"
4. Consider creating a `numDisparities` trackbar that directly sets the value (currently `numDisparitiesK` is the only way to control this, which is confusing)

---

## Implementation Details

### downSample (lines 179-190 of depth_processing.py)
- Value 0-100 representing percentage to reduce
- Example: `downSample=57` → reduces to 43% of original size (0.43x scale factor)
- Applied before rectification for performance
- Minimum enforced at 10% to prevent too-small images

### crop (lines 100-107 of depth_processing.py)  
- Value in pixels to crop from all edges
- Applied after rectification
- Useful for removing edge artifacts from fisheye undistortion

### nearCutoff (lines 157-169 of depth_processing.py) ✨ CHANGED TO RELATIVE
- Value 0-100 representing **percentage** of depth range from minimum
- Uses 1st percentile of valid depths to establish baseline (ignores outliers)
- Example: `nearCutoff=20` → cuts off nearest 20% of depth range
- **Adapts to scene**: Always relative to actual depth distribution
- Useful for ignoring very close objects (camera mount, bike frame)

### farCutoff (lines 172-174 of depth_processing.py) ✨ CHANGED TO RELATIVE
- Value 0-100 representing **percentage** of depth range from maximum
- Uses 99th percentile of valid depths to establish baseline (ignores outliers)
- Example: `farCutoff=15` → cuts off farthest 15% of depth range
- **Adapts to scene**: Always relative to actual depth distribution
- Useful for limiting depth range to area of interest

### useWLS / wlsLambda / wlsSigma (lines 62-75, 121-132 of depth_processing.py) ✨ NEW
- **WLS (Weighted Least Squares) filtering** significantly improves disparity quality
- Creates both left and right matchers for cross-checking
- `wlsLambda` (0-10000): Regularization strength. Higher = smoother but less detail
- `wlsSigma` (0.5-5.0): Edge-aware smoothing. Controls how much to preserve edges
- Requires `opencv-contrib-python` for `cv2.ximgproc` module
- Falls back gracefully if module not available

### useMorph / morphIter (lines 138-144 of depth_processing.py) ✨ NEW
- **Morphological closing** fills small holes in disparity map
- Uses elliptical 5x5 kernel
- `morphIter` (1-20): Number of iterations. Higher = more aggressive hole filling
- Applied after WLS filtering (if enabled)
- Useful for cleaning up noisy disparities

---

## Files Modified
- `src/hal/VISION/VISION.py` - Updated `_refresh_depth_processor()` to pass 9 parameters (lines 290-306)
- `src/hal/VISION/depth_processing.py` - Major update:
  - Added WLS filtering with left+right matchers
  - Added morphological hole filling
  - Changed near/far cutoffs to relative percentages
  - Implemented downSample and crop
- `config_setup.py` - Updated comments to reflect relative cutoffs
- `calibrate_disparity.py` - Updated trackbar ranges and labels for relative cutoffs

Generated: Saturday, November 8, 2025  
Updated: After activating WLS, morphological filtering, and making cutoffs relative (18 parameters now active)

