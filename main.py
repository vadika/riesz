import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from multiprocessing import Pool, cpu_count

def riesz_transform(image):
    """Apply the Riesz transform to an image."""
    rows, cols = image.shape
    x = np.fft.fftfreq(cols)
    y = np.fft.fftfreq(rows)
    
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    epsilon = 1e-10  # Small value to avoid division by zero
    fx = -1j * xx / (r + epsilon)
    fy = -1j * yy / (r + epsilon)

    img_fft = np.fft.fft2(image)
    rx = np.real(np.fft.ifft2(img_fft * fx))
    ry = np.real(np.fft.ifft2(img_fft * fy))

    return rx, ry


def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid."""
    pyramid = [image]
    for _ in range(levels - 1):
        image = signal.decimate(signal.decimate(image, 2, axis=0), 2, axis=1)
        pyramid.append(image)
    return pyramid


def laplacian_pyramid(image, levels):
    """Create a Laplacian pyramid."""
    gaussian_pyr = gaussian_pyramid(image, levels)
    laplacian_pyr = []

    for i in range(levels - 1):
        size = gaussian_pyr[i].shape
        upsampled = signal.resample(signal.resample(gaussian_pyr[i + 1], size[0], axis=0), size[1], axis=1)
        laplacian = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(laplacian)

    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


def riesz_pyramid(image, levels):
    """Create a Riesz pyramid."""
    laplacian_pyr = laplacian_pyramid(image, levels)
    riesz_pyr = []

    for lap in laplacian_pyr:
        rx, ry = riesz_transform(lap)
        riesz_pyr.append((rx, ry))

    return riesz_pyr


def load_frames_from_video(video_path, max_frames=None):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and count >= max_frames):
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        count += 1
    
    cap.release()
    return frames

def write_frames_to_video(frames, output_path, fps=30):
    """Write frames to a video file."""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()


def visualize_riesz_pyramid(frame, riesz_pyr, frame_number, amplification_factor=5):
    """Visualize all levels of the Riesz pyramid for a single frame and return the visualizations."""
    visualizations = []
    for i, (rx, ry) in enumerate(riesz_pyr):
        # Resize rx and ry to match frame size
        rx_resized = cv2.resize(rx, (frame.shape[1], frame.shape[0]))
        ry_resized = cv2.resize(ry, (frame.shape[1], frame.shape[0]))
        
        magnitude = np.sqrt(rx_resized**2 + ry_resized**2)
        
        # Amplify the magnitude
        magnitude_amplified = magnitude * amplification_factor
        
        # Normalize amplified magnitude to [0, 1] range
        magnitude_normalized = (magnitude_amplified - magnitude_amplified.min()) / (magnitude_amplified.max() - magnitude_amplified.min())
        
        # Create displacement map
        displacement_x = magnitude_normalized * np.cos(np.angle(rx_resized + 1j*ry_resized))
        displacement_y = magnitude_normalized * np.sin(np.angle(rx_resized + 1j*ry_resized))
        
        # Apply displacement to create moved image
        rows, cols = frame.shape
        row_indices, col_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        moved_row_indices = np.clip(row_indices + displacement_y * rows, 0, rows - 1).astype(int)
        moved_col_indices = np.clip(col_indices + displacement_x * cols, 0, cols - 1).astype(int)
        moved_image = frame[moved_row_indices, moved_col_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original frame
        axes[0, 0].imshow(frame, cmap='gray')
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Rx component
        axes[0, 1].imshow(rx, cmap='gray')
        axes[0, 1].set_title(f'Level {i+1}: Rx')
        axes[0, 1].axis('off')
        
        # Ry component
        axes[1, 0].imshow(ry, cmap='gray')
        axes[1, 0].set_title(f'Level {i+1}: Ry')
        axes[1, 0].axis('off')
        
        # Moved image
        axes[1, 1].imshow(moved_image, cmap='gray')
        axes[1, 1].set_title(f'Level {i+1}: Moved Image')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert the figure to an image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # RGBA to RGB
        
        visualizations.append(img)
        plt.close(fig)
    
    return visualizations


def process_frame(args):
    frame, frame_number, levels, amplification_factor = args
    riesz_pyr = riesz_pyramid(frame, levels)
    visualizations = visualize_riesz_pyramid(frame, riesz_pyr, frame_number, amplification_factor)
    return visualizations

def main():
    # Load frames from a video file
    video_path = "IMG_4101.MOV"  # Replace with your video file path
    frames = load_frames_from_video(video_path, max_frames=None)  # Load all frames

    # Set the number of levels for the pyramid and amplification factor
    levels = 7
    amplification_factor = 10  # Adjust this value to increase or decrease motion amplification

    print(f"Loaded {len(frames)} frames from the video.")

    # Prepare arguments for parallel processing
    args = [(frame, i+1, levels, amplification_factor) for i, frame in enumerate(frames)]

    # Use multiprocessing to process frames in parallel
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_frame, args)

    # Reorganize results by level
    level_frames = [[] for _ in range(levels)]
    for frame_visualizations in results:
        for level, visualization in enumerate(frame_visualizations):
            level_frames[level].append(visualization)

    # Create a video for each level
    for level, frames in enumerate(level_frames):
        output_path = f'riesz_pyramid_level_{level+1}_amplified.mp4'
        write_frames_to_video(frames, output_path)
        print(f"Video created for level {level+1}: {output_path}")

    print(f"Amplified visualization videos created for all levels.")

if __name__ == "__main__":
    main()
