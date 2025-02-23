import cv2
import numpy as np

def compute_sf_flow(prev_frame, next_frame):
    # Resize frames to 320x240
    prev_frame = cv2.resize(prev_frame, (320, 240))
    next_frame = cv2.resize(next_frame, (320, 240))
    
    # calcOpticalFlowSF expects color images (CV_8UC3), so we keep them in BGR
    # Set parameters for Simple Flow
    layers = 3
    averaging_block_size = 2
    max_flow = 4

    # Compute optical flow using Simple Flow algorithm
    flow = cv2.optflow.calcOpticalFlowSF(prev_frame, next_frame, layers, averaging_block_size, max_flow)

    # Convert the flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image to visualize the flow
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # Map the angle to hue (color)
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Normalize and apply gamma correction to the magnitude for better visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gamma = 1.2
    magnitude_norm = np.power(magnitude_norm / 255, gamma) * 255
    hsv[..., 2] = magnitude_norm.astype('uint8')

    # Convert HSV to BGR for display
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, flow_visualization

def main():
    # Load two consecutive frames from a video or images
    prev_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0000.jpg')
    next_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0010.jpg')

    if prev_frame is None or next_frame is None:
        print("Error: Could not load images.")
        return

    # Compute optical flow using the Simple Flow algorithm
    flow, flow_visualization = compute_sf_flow(prev_frame, next_frame)

    # Resize frames for display consistency
    prev_frame_resized = cv2.resize(prev_frame, (320, 240))
    next_frame_resized = cv2.resize(next_frame, (320, 240))
    
    # Display the original frames and the flow visualization
    cv2.imshow('Previous Frame', prev_frame_resized)
    cv2.imshow('Next Frame', next_frame_resized)
    cv2.imshow('Optical Flow', flow_visualization)
    
    # Save the flow visualization
    cv2.imwrite('optical_flow_visualization.jpg', flow_visualization)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
