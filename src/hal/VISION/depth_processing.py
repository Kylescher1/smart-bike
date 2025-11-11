import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple


class DepthProcessor:
    """Encapsulates stereo rectification, disparity, and depth conversion."""

    def __init__(
        self,
        left_config: Dict,
        right_config: Dict,
        stereo_matcher: Optional[cv2.StereoSGBM] = None,
        q_matrix: Optional[np.ndarray] = None,
        debug: bool = False,
        downSample: int = 0,
        crop: int = 0,
        nearCutoff: float = 0,
        farCutoff: float = 0,
        useMorph: bool = False,
        morphIter: int = 5,
        useWLS: bool = False,
        wlsLambda: float = 8000.0,
        wlsSigma: float = 1.5,
        smoothingKernel: int = 0,
        confidenceWindow: int = 5,
        confidenceThreshold: float = 0.0,
    ) -> None:
        self.left_config = left_config
        self.right_config = right_config
        self.stereo_matcher = stereo_matcher
        self.q_matrix = q_matrix
        self.debug = debug
        
        # Pre-processing & filtering parameters
        self.downSample = max(0, min(100, downSample))  # Clamp to 0-100%
        self.crop = max(0, crop)
        self.nearCutoff = max(0, min(100, nearCutoff))  # 0-100% relative cutoff
        self.farCutoff = max(0, min(100, farCutoff))  # 0-100% relative cutoff
        
        # Morphological filtering
        self.useMorph = useMorph
        self.morphIter = max(1, morphIter)
        
        # WLS filtering
        self.useWLS = useWLS
        self.wlsLambda = wlsLambda
        self.wlsSigma = wlsSigma
        self.wls_filter = None
        self.right_matcher = None

        # Post-processing filters
        self.smoothingKernel = int(max(0, smoothingKernel))
        self._smooth_kernel = self._normalize_kernel(self.smoothingKernel)

        self.confidenceWindow = int(max(0, confidenceWindow))
        self._confidence_kernel = self._normalize_kernel(
            self.confidenceWindow if self.confidenceWindow > 0 else 5
        )
        self.confidenceThreshold = float(max(0.0, min(100.0, confidenceThreshold)))
        
        # Initialize WLS filter if enabled
        if self.useWLS and self.stereo_matcher is not None:
            self._init_wls_filter()

    @staticmethod
    def _normalize_kernel(value: int) -> int:
        """Ensure kernels are odd and >=3. Returns 0 if below usable size."""
        if value <= 0:
            return 0
        if value < 3:
            return 0
        return value if value % 2 == 1 else value + 1

    def update_matcher(self, stereo_matcher: Optional[cv2.StereoSGBM]) -> None:
        self.stereo_matcher = stereo_matcher
        if self.useWLS and stereo_matcher is not None:
            self._init_wls_filter()

    def update_q_matrix(self, q_matrix: Optional[np.ndarray]) -> None:
        self.q_matrix = q_matrix
    
    def _init_wls_filter(self) -> None:
        """Initialize WLS filter for disparity refinement."""
        try:
            # Create right matcher for WLS
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
            # Create WLS filter
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
            self.wls_filter.setLambda(self.wlsLambda)
            self.wls_filter.setSigmaColor(self.wlsSigma)
        except AttributeError:
            if self.debug:
                print("Warning: cv2.ximgproc not available. WLS filtering disabled.")
            self.wls_filter = None
            self.right_matcher = None

    def rectify(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_map_x = self.left_config.get("map_x")
        left_map_y = self.left_config.get("map_y")
        right_map_x = self.right_config.get("map_x")
        right_map_y = self.right_config.get("map_y")

        missing_maps = [
            name
            for name, value in (
                ("left.map_x", left_map_x),
                ("left.map_y", left_map_y),
                ("right.map_x", right_map_x),
                ("right.map_y", right_map_y),
            )
            if value is None
        ]

        if missing_maps:
            raise RuntimeError(f"Calibration maps missing: {', '.join(missing_maps)}")

        rect_left = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
        
        # Apply cropping if specified
        if self.crop > 0:
            h, w = rect_left.shape[:2]
            crop_h = max(0, h - 2 * self.crop)
            crop_w = max(0, w - 2 * self.crop)
            if crop_h > 0 and crop_w > 0:
                rect_left = rect_left[self.crop:self.crop+crop_h, self.crop:self.crop+crop_w]
                rect_right = rect_right[self.crop:self.crop+crop_h, self.crop:self.crop+crop_w]
        
        return rect_left, rect_right

    def compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        if self.stereo_matcher is None:
            raise RuntimeError("Stereo matcher not initialized. Call start() first.")

        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Compute left disparity
        disparity_left = self.stereo_matcher.compute(gray_left, gray_right)
        
        # Apply morphological filtering first (if enabled) - fills holes before refinement
        if self.useMorph and self.morphIter > 0:
            # Use morphological closing to fill small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            disparity_morph = cv2.morphologyEx(disparity_left, cv2.MORPH_CLOSE, kernel, iterations=self.morphIter)
            disparity_left = disparity_morph
        
        # Apply WLS filtering as final refinement step (if enabled)
        if self.useWLS and self.wls_filter is not None and self.right_matcher is not None:
            try:
                # Compute right disparity for WLS
                disparity_right = self.right_matcher.compute(gray_right, gray_left)
                # Apply WLS filter - this refines the disparity with edge-aware smoothing
                disparity = self.wls_filter.filter(disparity_left, gray_left, None, disparity_right)
                disparity = disparity.astype(np.float32) / 16.0
            except Exception as e:
                if self.debug:
                    print(f"WLS filtering failed, using raw disparity: {e}")
                disparity = disparity_left.astype(np.float32) / 16.0
        else:
            disparity = disparity_left.astype(np.float32) / 16.0
        
        disparity[disparity < 0] = 0

        if self.confidenceThreshold > 0 and self._confidence_kernel >= 3:
            valid_mask = (disparity > 0).astype(np.float32)
            neighborhood = cv2.boxFilter(
                valid_mask,
                ddepth=-1,
                ksize=(self._confidence_kernel, self._confidence_kernel),
                normalize=True,
            )
            disparity[neighborhood * 100.0 < self.confidenceThreshold] = 0
        
        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        if self.q_matrix is None:
            raise RuntimeError("Calibration Q matrix not loaded. Call start() first.")

        points = cv2.reprojectImageTo3D(disparity, self.q_matrix)
        depth = points[:, :, 2]
        depth[~np.isfinite(depth)] = 0
        depth = np.maximum(depth, 0)
        
        # Apply relative near/far cutoffs (percentage-based)
        if self.nearCutoff > 0 or self.farCutoff > 0:
            # Get depth range (ignoring zeros)
            valid_depths = depth[depth > 0]
            if valid_depths.size > 0:
                min_depth = np.percentile(valid_depths, 1)  # Use 1st percentile to avoid outliers
                max_depth = np.percentile(valid_depths, 99)  # Use 99th percentile to avoid outliers
                depth_range = max_depth - min_depth
                
                # Apply near cutoff as percentage from minimum
                if self.nearCutoff > 0:
                    near_threshold = min_depth + (depth_range * self.nearCutoff / 100.0)
                    depth[depth < near_threshold] = 0
                
                # Apply far cutoff as percentage from maximum
                if self.farCutoff > 0:
                    far_threshold = max_depth - (depth_range * self.farCutoff / 100.0)
                    depth[depth > far_threshold] = 0
        
        return depth

    def process_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # Apply downsampling if specified (0-100% scale)
        if self.downSample > 0:
            scale_percent = 100 - self.downSample
            if scale_percent < 10:
                scale_percent = 10  # Minimum 10% size
            scale_factor = scale_percent / 100.0
            
            new_width = int(left_frame.shape[1] * scale_factor)
            new_height = int(left_frame.shape[0] * scale_factor)
            
            left_frame = cv2.resize(left_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            right_frame = cv2.resize(right_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        rect_left, rect_right = self.rectify(left_frame, right_frame)
        disparity = self.compute_disparity(rect_left, rect_right)
        depth_map = self.disparity_to_depth(disparity)

        if self._smooth_kernel >= 3:
            depth_map = cv2.GaussianBlur(
                depth_map, (self._smooth_kernel, self._smooth_kernel), 0
            )

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_disparities": int(self.stereo_matcher.getNumDisparities()) if self.stereo_matcher else 0,
        }

        return depth_map, disparity, metadata

