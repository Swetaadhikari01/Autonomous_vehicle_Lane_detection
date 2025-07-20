import cv2 
import numpy as np

# Step 1: Apply grayscale filter
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Blur to smooth the image
def gaussian_blur(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)

# Step 3: Apply Canny Edge Detection
def canny_edges(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)

# Step 4: Define the Region of Interest (ROI) to focus on lane area
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width - 100, height - 300),
        (100, height - 300)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Step 5: Use Hough Transform to detect the lanes
def hough_transform(img):
    return cv2.HoughLinesP(
        img,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )

# Step 6: Draw the detected lanes on the original image
def draw_lines(img, lines):
    img_copy = np.copy(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return img_copy

# Step 7: Main function to process video frames
def lane_detection_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = grayscale(frame)
        blurred = gaussian_blur(gray)
        edges = canny_edges(blurred)
        roi = region_of_interest(edges)
        lines = hough_transform(roi)

        if lines is not None:
            result_img = draw_lines(frame, lines)
        else:
            result_img = frame

        cv2.imshow("Lane Detection", result_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
lane_detection_video('C:\\Users\\DELL\\OneDrive\\Desktop\\AIML project\\video\\test_video.mp4')  