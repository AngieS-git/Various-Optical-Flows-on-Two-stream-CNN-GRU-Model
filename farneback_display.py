import cv2
import numpy as np

def compute_farneback_flow(prev_frame, next_frame):
    # Resize frames to 320x240
    prev_frame = cv2.resize(prev_frame, (320, 240))
    next_frame = cv2.resize(next_frame, (320, 240))
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Apply slight Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)

    # Calculate Farneback optical flow with adjusted parameters for 320x240 resolution
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, 
        next_gray, 
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=11,        # Reduced window size for smaller resolution
        iterations=5,      # Increased iterations for better accuracy
        poly_n=5,
        poly_sigma=1.1,   # Slightly reduced for sharper detail
        flags=0
    )

    # Convert the flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image to visualize the flow
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # Angle corresponds to Hue (color)
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Enhance the magnitude visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Apply gamma correction to make flow more visible
    gamma = 1.2
    magnitude_norm = np.power(magnitude_norm/255, gamma) * 255
    hsv[..., 2] = magnitude_norm.astype('uint8')

    # Convert HSV to BGR for visualization
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, flow_visualization

def main():
    # Load two consecutive frames from a video
    prev_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0000.jpg')
    next_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0010.jpg')

    if prev_frame is None or next_frame is None:
        print("Error: Could not load images.")
        return

    # Compute Farneback optical flow
    flow, flow_visualization = compute_farneback_flow(prev_frame, next_frame)

    # Create side-by-side comparison
    prev_frame_resized = cv2.resize(prev_frame, (320, 240))
    next_frame_resized = cv2.resize(next_frame, (320, 240))
    
    # Display the original frames and flow visualization
    cv2.imshow('Previous Frame', prev_frame_resized)
    cv2.imshow('Next Frame', next_frame_resized)
    cv2.imshow('Optical Flow', flow_visualization)
    
    # Save the visualization
    cv2.imwrite('optical_flow_visualization.jpg', flow_visualization)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()