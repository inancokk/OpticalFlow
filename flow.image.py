import numpy as np
import cv2
import os

def make_integral_image(image):
    integral_image = np.zeros_like(image, dtype=np.int64)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            integral_image[y, x] = image[y, x]
            if x > 0:
                integral_image[y, x] += integral_image[y, x-1]
            if y > 0:
                integral_image[y, x] += integral_image[y-1, x]
            if x > 0 and y > 0:
                integral_image[y, x] -= integral_image[y-1, x-1]
    return integral_image

def time_structure_matrix(im1, im2, window_size):
    Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
    It = im2 - im1
    Ix2 = cv2.GaussianBlur(Ix**2, (window_size, window_size), 0)
    Iy2 = cv2.GaussianBlur(Iy**2, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ix*Iy, (window_size, window_size), 0)
    Ixt = cv2.GaussianBlur(Ix*It, (window_size, window_size), 0)
    Iyt = cv2.GaussianBlur(Iy*It, (window_size, window_size), 0)
    return Ix2, Iy2, Ixy, Ixt, Iyt

def velocity_image(Ix2, Iy2, Ixy, Ixt, Iyt, stride):
    height, width = Ix2.shape
    V = np.zeros((height, width, 2), dtype=np.float32)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            M = np.array([[Ix2[y, x], Ixy[y, x]], [Ixy[y, x], Iy2[y, x]]])
            b = np.array([-Ixt[y, x], -Iyt[y, x]])
            if np.linalg.det(M) != 0:
                velocity = np.linalg.inv(M).dot(b)
                V[y, x] = velocity
            else:
                V[y, x] = [0, 0]
    return V

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def main():
    # Dosya yolunu ayarlayın
    dog_a_path = 'C:\Project\dog_a.jpg'
    dog_b_path = 'C:\Project\dog_b.jpg'

    # Dosya yollarını yazdırarak kontrol edin
    print("Checking if image files exist...")
    print(f"Does dog_a.jpg exist? {os.path.isfile(dog_a_path)}")
    print(f"Does dog_b.jpg exist? {os.path.isfile(dog_b_path)}")

    # Görüntüleri yükleyin
    dog_a = cv2.imread(dog_a_path, cv2.IMREAD_GRAYSCALE)
    dog_b = cv2.imread(dog_b_path, cv2.IMREAD_GRAYSCALE)

    if dog_a is None or dog_b is None:
        print("Error: Image files not found.")
        return

    # Görüntüleri aynı boyuta getirin
    height, width = dog_a.shape[:2]
    dog_b = cv2.resize(dog_b, (width, height))

    print(f"dog_a shape: {dog_a.shape}")
    print(f"dog_b shape: {dog_b.shape}")

    Ix2, Iy2, Ixy, Ixt, Iyt = time_structure_matrix(dog_a, dog_b, 15)
    velocity = velocity_image(Ix2, Iy2, Ixy, Ixt, Iyt, 8)

    flow_vis = draw_flow(dog_a, velocity, 8)
    cv2.imshow('Flow Visualization', flow_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    optical_flow_webcam(15, 4, 8)

def optical_flow_webcam(window_size, stride, step):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video device.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video device.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        Ix2, Iy2, Ixy, Ixt, Iyt = time_structure_matrix(prev_gray, gray, window_size)
        velocity = velocity_image(Ix2, Iy2, Ixy, Ixt, Iyt, stride)

        flow_vis = draw_flow(gray, velocity, step)
        cv2.imshow('Optical Flow', flow_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
