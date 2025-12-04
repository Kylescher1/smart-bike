import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION (Match your Arduino Code)
# ==========================================
MIN_DIST_M = 2   # Meters: Distance for MAX vibration
MAX_DIST_M = 5   # Meters: Distance where vibration starts (MIN)
PAN_ANGLE = 25.0   # Degrees: Angle where signal becomes 100% one-sided

def calculate_haptics(r, theta):
    """
    Calculates Left and Right PWM (0-255) for a given point (r, theta).
    r: distance in meters
    theta: angle in degrees (Negative=Left, Positive=Right)
    """
    # 1. Filter Out of Range
    if r > MAX_DIST_M:
        return 0, 0
    
    # 2. Calculate BASE Intensity (Exponential Curve)
    # 0.0 = Far, 1.0 = Close
    normDist = (MAX_DIST_M - r) / (MAX_DIST_M - MIN_DIST_M)
    normDist = np.clip(normDist, 0, 1) 
    
    # Apply the "Anxious" Exponential Square Curve
    baseIntensity = (normDist ** 2) * 255
    
    # 3. Calculate Balance Factors (Stereo Panning)
    # Left Motor is 100% unless object moves Right (theta > 0)
    # Right Motor is 100% unless object moves Left (theta < 0)
    
    if theta < 0:
        # Object is Left
        leftMix = 1.0
        # Fade Right motor out as we go further left
        rightMix = 1.0 + (theta / PAN_ANGLE) 
    else:
        # Object is Right
        rightMix = 1.0
        # Fade Left motor out as we go further right
        leftMix = 1.0 - (theta / PAN_ANGLE)  
        
    # Clamp mixes to 0.0 - 1.0
    leftMix = np.clip(leftMix, 0, 1)
    rightMix = np.clip(rightMix, 0, 1)
    
    # Apply Mix to Base Intensity
    leftPWM = baseIntensity * leftMix
    rightPWM = baseIntensity * rightMix
    
    return leftPWM, rightPWM

# ==========================================
# GENERATE PLOT DATA
# ==========================================
# Create a grid of points (r, theta)
r_vals = np.linspace(0.1, 5.0, 150)
theta_vals = np.linspace(-90, 90, 150) # Degrees (+/- 60 FOV)

R, THETA = np.meshgrid(r_vals, theta_vals)

# Vectorize the function to apply it to the whole grid at once
calc_vec = np.vectorize(calculate_haptics)
L_PWM, R_PWM = calc_vec(R, THETA)

# Convert Polar (r, theta) to Cartesian (x, y) for the top-down map
# 0 deg is "Up" (Positive Y), +90 deg is Right (Positive X)
THETA_RAD = np.radians(THETA)
X = R * np.sin(THETA_RAD)
Y = R * np.cos(THETA_RAD)

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# 1. LEFT MOTOR PLOT
c1 = axes[0].pcolormesh(X, Y, L_PWM, cmap='magma', vmin=0, vmax=255, shading='auto')
axes[0].set_title("Left Motor Intensity", fontsize=16, fontweight='bold')
axes[0].set_xlabel("Lateral Distance (m)")
axes[0].set_ylabel("Forward Distance (m)")
axes[0].set_aspect('equal')
axes[0].grid(True, linestyle='--', alpha=0.3)
# Draw Robot
axes[0].plot(0, 0, 'ko', markersize=15, markeredgecolor='white', label='Robot')
axes[0].text(0, -0.4, 'You', ha='center', fontsize=12)

# 2. RIGHT MOTOR PLOT
c2 = axes[1].pcolormesh(X, Y, R_PWM, cmap='magma', vmin=0, vmax=255, shading='auto')
axes[1].set_title("Right Motor Intensity", fontsize=16, fontweight='bold')
axes[1].set_xlabel("Lateral Distance (m)")
axes[1].set_aspect('equal')
axes[1].grid(True, linestyle='--', alpha=0.3)
# Draw Robot
axes[1].plot(0, 0, 'ko', markersize=15, markeredgecolor='white')
axes[1].text(0, -0.4, 'You', ha='center', fontsize=12)

# Colorbar
cbar = fig.colorbar(c2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Vibration PWM (0-255)', fontsize=12)

plt.suptitle(f"Haptic Field Visualization\nRange: {MIN_DIST_M}-{MAX_DIST_M}m | Stereo Angle: +/- {PAN_ANGLE}°", fontsize=16)
plt.show()