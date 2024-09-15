import numpy as np
from scipy import signal
import cv2


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


# Example usage
def main():
    # Load frames from a video file
    video_path = "path/to/your/video.mp4"  # Replace with your video file path
    frames = load_frames_from_video(video_path, max_frames=10)  # Load up to 10 frames

    # Set the number of levels for the pyramid
    levels = 4

    print(f"Loaded {len(frames)} frames from the video.")

    for i, frame in enumerate(frames):
        # Compute the Riesz pyramid for each frame
        riesz_pyr = riesz_pyramid(frame, levels)

        print(f"Frame {i + 1}: Riesz pyramid created with {levels} levels.")
        for j, (rx, ry) in enumerate(riesz_pyr):
            print(f"  Level {j}: Rx shape = {rx.shape}, Ry shape = {ry.shape}")

        # Here you can add more processing or visualization of the Riesz pyramid if needed


if __name__ == "__main__":
    main()
