import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def riesz_transform(image):
    """Apply the Riesz transform to an image."""
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols) / cols - 0.5,
                       np.arange(rows) / rows - 0.5)
    r = np.sqrt(x ** 2 + y ** 2)

    fx = -1j * x / (r + np.finfo(float).eps)
    fy = -1j * y / (r + np.finfo(float).eps)

    fx[0, 0] = fy[0, 0] = 0

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


def visualize_riesz_pyramid(frame, riesz_pyr):
    """Visualize the Riesz pyramid for a single frame and return the figure."""
    levels = len(riesz_pyr)
    fig, axes = plt.subplots(levels, 3, figsize=(15, 5*levels))
    
    axes[0, 0].imshow(frame, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    for i, (rx, ry) in enumerate(riesz_pyr):
        magnitude = np.sqrt(rx**2 + ry**2)
        
        axes[i, 0].imshow(rx, cmap='gray')
        axes[i, 0].set_title(f'Level {i}: Rx')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ry, cmap='gray')
        axes[i, 1].set_title(f'Level {i}: Ry')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(magnitude, cmap='viridis')
        axes[i, 2].set_title(f'Level {i}: Magnitude')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img


def main():
    # Load frames from a video file
    video_path = "IMG_4101.MOV"  # Replace with your video file path
    frames = load_frames_from_video(video_path, max_frames=None)  # Load all frames

    # Set the number of levels for the pyramid
    levels = 4

    print(f"Loaded {len(frames)} frames from the video.")

    output_frames = []
    for i, frame in enumerate(frames):
        # Compute the Riesz pyramid for each frame
        riesz_pyr = riesz_pyramid(frame, levels)

        print(f"Frame {i + 1}: Riesz pyramid created with {levels} levels.")
        for j, (rx, ry) in enumerate(riesz_pyr):
            print(f"  Level {j}: Rx shape = {rx.shape}, Ry shape = {ry.shape}")

        # Visualize the Riesz pyramid and get the output frame
        output_frame = visualize_riesz_pyramid(frame, riesz_pyr)
        output_frames.append(output_frame)

    # Write the output frames to a video file
    output_path = "riesz_pyramid_visualization.mp4"
    write_frames_to_video(output_frames, output_path)
    print(f"Visualization video saved to {output_path}")

if __name__ == "__main__":
    main()
