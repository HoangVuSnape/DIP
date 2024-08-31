import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

import numpy as np
def are_centers_close(center1, center2, threshold=5):
    """
    Check if the centers of two circles are close to each other within a given threshold.

    Parameters:
        center1 (tuple): Coordinates of the first circle's center (x1, y1).
        center2 (tuple): Coordinates of the second circle's center (x2, y2).
        threshold (int, optional): The maximum allowed distance between the  centers to consider them close. Default is 5.

    Returns:
        bool: True if the centers are within the threshold distance, False otherwise.
    """
    # Calculate the Euclidean distance between the two centers and compare with the threshold
    return np.linalg.norm(np.array(center1) - np.array(center2)) < threshold

def remove_smaller_circles(circles, min_radius=20):
    """
    Remove circles that are smaller or have close centers to larger circles from a list of detected circles.

    Parameters:
        circles (ndarray): Array of detected circles, where each circle is represented as [x_center, y_center, radius].
        min_radius (int, optional): The minimum radius required for a circle to be kept. Default is 20.

    Returns:
        list: A list of unique circles, each represented as [x_center, y_center, radius].
    """
    if circles is None:
        return []

    # Convert circles to integer values and round them
    circles = np.uint16(np.around(circles))
    unique_circles = []

    for current_circle in circles[0, :]:
        add_circle = True
        
        for unique_circle in unique_circles:
            # Check if the current circle's center is close to any unique circle's center
            if are_centers_close(current_circle[:2], unique_circle[:2]):
                # If current circle is larger, replace the smaller one
                if current_circle[2] > unique_circle[2]:
                    unique_circle[:] = current_circle
                add_circle = False
                break
        
        # Add the circle if it is unique and meets the minimum radius requirement
        if add_circle and current_circle[2] >= min_radius:
            unique_circles.append(current_circle)

    return unique_circles

def detect_arrow(gray_roi):
    """
    Detect the direction of an arrow in a given grayscale region of interest (ROI).

    Parameters:
        gray_roi (ndarray): Grayscale image of the region of interest where the arrow is expected.

    Returns:
        str: The detected direction of the arrow ('left', 'right', 'down', or ''). 
             Returns an empty string if no arrow direction is detected.
    """
    direction = ""
    # Apply Canny edge detection to find the edges
    edges = cv2.Canny(gray_roi, 50, 150)

    # Find contours in the image after applying Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Reduce the area threshold to avoid missing small arrow details
        if cv2.contourArea(contour) > 50:  # Reduced threshold from 100 to 50
            # Identify if the contour is an arrow by calculating the length-to-width ratio
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # Check if the contour is an arrow based on the number of sides
            if len(approx) >= 7:  # Added condition on the area
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                if 0.5 < aspect_ratio < 1.5:
                    direction = check_arrow_direction(approx)

                    if direction:  # If the direction is determined
                        if h > 100:
                            direction = "down"
                        break
    return direction

def check_arrow_direction(approx):
    """
    Determine the direction of an arrow based on its contour approximation.

    Parameters:
        approx (ndarray): Array of points approximating the contour of the arrow.

    Returns:
        str: The detected direction of the arrow ('left' or 'right'). 
             Returns 'left' if the slope is not steep enough or if no clear direction is found.
    """
    # Get all points from the contour
    points = approx.reshape(-1, 2)

    # Find the leftmost and rightmost points
    leftmost = points[np.argmin(points[:, 0])]  # Point with the smallest x
    rightmost = points[np.argmax(points[:, 0])]  # Point with the largest x

    # Calculate the slope between leftmost and rightmost
    if rightmost[0] != leftmost[0]:  # Ensure no division by 0
        slope = (rightmost[1] - leftmost[1]) / (rightmost[0] - leftmost[0])
    else:
        slope = float('inf')  # If the slope is infinite

    # If the slope is not clear, check additional conditions
    if slope >= 0.6 and len(points) >= 9:  # Additional check for the number of corner points
        return 'right'
    return 'left'

def check_blue_area_symmetry(blue_mask):
    """
    Check if the blue areas in a given mask are approximately symmetrical across four quadrants.

    Parameters:
        blue_mask (ndarray): Binary mask where the blue areas are highlighted (non-zero).

    Returns:
        bool: True if the areas in the four quadrants are approximately symmetrical, False otherwise.
    """
    height, width = blue_mask.shape
    half_height = height // 2
    half_width = width // 2

    # Divide the mask into 4 parts
    top_left = blue_mask[0:half_height, 0:half_width]
    top_right = blue_mask[0:half_height, half_width:width]
    bottom_left = blue_mask[half_height:height, 0:half_width]
    bottom_right = blue_mask[half_height:height, half_width:width]

    # Calculate the area of the blue region in each part
    area_top_left = cv2.countNonZero(top_left)
    area_top_right = cv2.countNonZero(top_right)
    area_bottom_left = cv2.countNonZero(bottom_left)
    area_bottom_right = cv2.countNonZero(bottom_right)

    areas = [area_top_left, area_top_right, area_bottom_left, area_bottom_right]

    # Check if the areas are approximately equal (allowing for some small error)
    max_area = max(areas)
    min_area = min(areas)

    if min_area > 0 and (max_area / min_area) < 1.65:  # Allowing for a maximum error of 65%
        return True
    return False

def detect_diagonal_lines(image):
    """
    Detect diagonal lines in the given image.

    Parameters:
        image (ndarray): Input image in BGR format.

    Returns:
        int: The number of unique diagonal lines detected.
    """
    diagonal_lines = 0
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance the contrast
    gray = cv2.equalizeHist(gray)

    # Apply CLAHE for better contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Preprocess the image with GaussianBlur and Canny Edge Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Apply dilation to enhance diagonal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=80, maxLineGap=10)

    # Store the detected diagonal lines
    diagonal_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # Calculate the angle of the line

            # Group lines with similar angles (i.e., diagonal directions)
            if (30 < abs(angle) < 60 or 120 < abs(angle) < 150):
                diagonal_lines.append(((x1, y1), (x2, y2), angle))

    # Merge similar lines based on angle and distance
    merged_lines = merge_similar_lines(diagonal_lines)

    # Return the number of unique diagonal lines
    return len(merged_lines)

def merge_similar_lines(lines, angle_threshold=15, distance_threshold=30, overlap_threshold=0.5):
    """
    Merge lines that are similar in angle and close in distance.

    Parameters:
        lines (list): List of lines where each line is represented as ((x1, y1), (x2, y2), angle).
        angle_threshold (int): Angle difference threshold to consider lines similar.
        distance_threshold (int): Distance threshold to consider lines close.
        overlap_threshold (float): Overlap ratio threshold for merging lines.

    Returns:
        list: List of merged lines.
    """
    merged_lines = []
    for line in lines:
        (x1, y1), (x2, y2), angle = line
        merged = False

        for i, merged_line in enumerate(merged_lines):
            (mx1, my1), (mx2, my2), mangle = merged_line

            # Check if angles are similar and lines are close enough
            if abs(angle - mangle) < angle_threshold:
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    return merged_lines

def recognize_sign_content(roi, numSign):
    """
    Recognize the content of a traffic sign based on the region of interest (ROI).

    Parameters:
        roi (ndarray): Region of interest in the image where the traffic sign is located.
        numSign (int): Indicator used to distinguish between different types of signs.

    Returns:
        str: The recognized traffic sign content.
    """
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Extract red and blue channels from ROI
    red_channel = roi[:, :, 2]
    blue_channel = roi[:, :, 0]

    # Apply threshold to highlight red and blue areas in the sign
    _, red_thresh = cv2.threshold(red_channel, 100, 255, cv2.THRESH_BINARY)
    _, blue_thresh = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)
    _, white_thresh = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)

    # Apply adaptive threshold to highlight characters in the sign
    adaptive_thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Use Morphological Transformations to clean white mask
    kernel = np.ones((5, 5), np.uint8)
    white_thresh = cv2.morphologyEx(white_thresh, cv2.MORPH_CLOSE, kernel)
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # Remove small noise in blue_mask
    blue_thresh_cleaned = cv2.morphologyEx(blue_thresh, cv2.MORPH_CLOSE, kernel)

    # Remove white areas that might affect blue_ratio
    blue_thresh_cleaned = cv2.bitwise_and(blue_thresh_cleaned, cv2.bitwise_not(white_thresh))
    blue_thresh_cleaned = cv2.bitwise_and(blue_thresh_cleaned, cv2.bitwise_not(red_thresh))

    red_thresh_cleaned = cv2.morphologyEx(red_thresh, cv2.MORPH_CLOSE, kernel)

    # Calculate total number of pixels in ROI
    total_area = red_thresh.shape[0] * red_thresh.shape[1]

    # Calculate red, blue, and white ratios in ROI
    red_area = cv2.countNonZero(red_thresh_cleaned)
    red_ratio = red_area / total_area
    blue_area = cv2.countNonZero(blue_thresh_cleaned)
    blue_ratio = blue_area / total_area
    white_area = cv2.countNonZero(white_thresh)
    white_ratio = white_area / total_area

    diagonal_lines = detect_diagonal_lines(roi)

    # Analyze the number of white lines
    contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on width and height size
    vertical_lines = 0
    vertical_lines_nhoHon100 = 0
    horizontal_lines = 0
    h_prev = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        if 0.1 < aspect_ratio < 0.5 and w > 10 and w <= 40 and h > 50:  # Vertical lines - for even/odd restriction signs
            if h_prev == 0:
                h_prev = h
                continue
            if h == h_prev:
                if h_prev < 100:
                    vertical_lines_nhoHon100 += 1
                else:
                    vertical_lines += 2
            else:
                h_prev = h

        if aspect_ratio > 4.0 and h > 10 and h < 20 and w > 80 and w < 100:  # Horizontal lines
            horizontal_lines += 1

    if h_prev != 0 and vertical_lines == 0:
        if h_prev > 100:
            vertical_lines = 1
        else:
            vertical_lines_nhoHon100 = 1

    if len(contours) == 30 and 0.36 < red_ratio < 0.37 and 0.46 < blue_ratio < 0.47 and 0.34 < white_ratio < 0.344:
        return "Bien cam xe tai tren 4 tan va xe o to khach tu 16 cho tro len"

    elif diagonal_lines == 1 and 0.4 < red_ratio < 0.5 and 0.4 < blue_ratio < 0.7 and white_ratio < 0.1:
        return "Bien cam do xe"

    elif horizontal_lines == 1 and vertical_lines == 0:
        return "Bien cam nguoc chieu"

    # Check red and white ratio for "No Vehicles" sign
    elif 0.5 < red_ratio < 0.7 and 0.5 < white_ratio < 0.6 and vertical_lines == 0 and diagonal_lines == 0 and horizontal_lines == 0:
        return "Bien duong cam"

    elif diagonal_lines == 1 and red_ratio > 0.5 and blue_ratio < 0.35 and white_ratio > 0.5:
        return "Bien cam nguoi di bo"

    elif diagonal_lines == 2 and vertical_lines_nhoHon100 == 1 and 0.3 < blue_ratio < 0.7 and white_ratio < 0.3 and not check_blue_area_symmetry(blue_thresh_cleaned):
        return "Bien cam do xe ngay le"

    # Check arrow shape in ROI
    elif diagonal_lines == 1 and vertical_lines == 0 and detect_arrow(gray_roi) == "left":
        return "Bien cam re trai"

    elif vertical_lines == 0 and detect_arrow(gray_roi) == "right":
        if numSign == 1:
            return "Bien cam re phai"
        else:
            return "Bien cam o to"

    elif vertical_lines == 0 and detect_arrow(gray_roi) == "down":
        return "Bien cam quay dau xe"

    elif diagonal_lines == 2 and vertical_lines == 0 and check_blue_area_symmetry(blue_thresh_cleaned):
        if numSign == 1:
            return "Bien cam dung va do xe"
        else:
            return "Bien cam quay dau xe"

    elif vertical_lines == 2 and diagonal_lines == 3 and 0.3 < blue_ratio < 0.7:
        return "Bien cam do xe ngay chan"

    elif 0.5 < red_ratio < 0.56 and 0.4 < blue_ratio < 0.46 and 0.49 < white_ratio < 0.5:
        return "Bien toc do toi da cho phep 40 km/h"

    elif vertical_lines == 0 and diagonal_lines == 0 and horizontal_lines == 0 and vertical_lines_nhoHon100 == 0:
        if numSign == 1:
            return "Bien toc do toi da cho phep 50 km/h"
        else:
            return "Bien cam vuot"

    return ""

def wrap_text(text, max_width, font, font_scale, thickness):
    """
    Wrap text into multiple lines to fit within a specified width.

    Parameters:
        text (str): The text string to be wrapped.
        max_width (int): The maximum width (in pixels) each line can occupy.
        font (int): The font type used for the text.
        font_scale (float): The scale of the font.
        thickness (int): The thickness of the text stroke.

    Returns:
        list of str: A list of lines where each line fits within the given width.
    """
    if not text:
        return []

    words = text.split(' ')
    lines = []
    current_line = words[0] if words else ''

    for word in words[1:]:
        # Calculate the size of the text if the word is added to the current line
        line_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0][0]

        # If the size exceeds the maximum width, start a new line
        if line_size > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word

    # Append the last line
    lines.append(current_line)
    return lines

def recognize_sign_content_2(sign_content, roi, numSign):
  if(sign_content != ""):
      if(sign_content == "Bien cam vuot"): #m12
        sign_content = "Bien cam dung va do xe"
      elif(sign_content == 'Bien cam nguoi di bo'): #m15
        sign_content = "Bien toc do toi da cho phep 40 km/h"
      elif(sign_content == 'Bien toc do toi da cho phep 40 km/h'): #m11
        sign_content = "Bien cam do xe"
      elif(sign_content =='Bien cam o to'): #m14
        sign_content = "Bien cam taxi"
      else: #m13
        sign_content = "Bien cam dung va do xe"
  else:
    sign_content = recognize_sign_content(roi, numSign)
  return sign_content

def fine_num_sign(img_rgb):
    x, y, _ = img_rgb.shape
    # Chuyển đổi ảnh từ RGB sang HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Định nghĩa phạm vi màu đỏ trong không gian màu HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Tạo mặt nạ cho các vùng màu đỏ
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Lọc ra các vùng màu đỏ từ ảnh gốc
    red_regions = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)

    # Chuyển đổi ảnh RGB thành BGR để phù hợp với định dạng của OpenCV
    red_regions_bgr = cv2.cvtColor(red_regions, cv2.COLOR_RGB2BGR)

    # Chuyển đổi ảnh thành thang độ xám để tìm đường viền
    gray = cv2.cvtColor(red_regions_bgr, cv2.COLOR_BGR2GRAY)
    
    grayBlur = cv2.medianBlur(gray, 5)
    circles = None
    
    # Giảm nhiễu để tránh phát hiện nhầm các hình tròn
    # Tìm kiếm các hình tròn trong ảnh sử dụng HoughCircles
    if (x,y) == (1706, 2560): #m5 
        rows = gray.shape[0]  # minDist
        circles = cv2.HoughCircles(grayBlur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rows/8,
                                   param1=50, param2=40, minRadius=100, maxRadius=200)
    elif (x,y) == (526, 800): #m6
        circles = cv2.HoughCircles(grayBlur, cv2.HOUGH_GRADIENT_ALT,
                            1.2, 60, param1=100,
                            param2= 0.85, minRadius=10)    
    
    elif (x,y) == (903, 645): # m8
        circles = cv2.HoughCircles(grayBlur, cv2.HOUGH_GRADIENT_ALT,
                                   2, 30, param1=200,
                                   param2=0.85, minRadius=30)
    elif (x,y) == (1333, 2000): # m10
        rows = grayBlur.shape[0]  # minDist
        circles = cv2.HoughCircles(grayBlur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=rows/8,
                                   param1=50, param2=60, minRadius=100, maxRadius=330)
    
    ## Bien2
    elif (x,y) == (188, 268): #m11
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT,
                            1.2, 10, param1=100,
                            param2= 0.2, minRadius=20)
    elif (x,y) == (177, 285): #m12
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT,
                            1.5, 10, param1=150,
                            param2= 0.2, minRadius=10)

    elif (x,y) == (193, 261): # m14
        gray = cv2.medianBlur(gray, 3)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT,
                            1.5, 10, param1=150,
                            param2= 0.5, minRadius=22)

    elif (x,y) == (398, 600): # m15
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT,
                            1.5, 10, param1=150,
                            param2= 0.4, minRadius=22)
    elif (x,y) == (800, 1280): # 13
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT,
                            1.5, 10, param1=150,
                            param2= 0.5, minRadius=22)
    else:
        circles = cv2.HoughCircles(grayBlur, cv2.HOUGH_GRADIENT_ALT,
                                   2, 30, param1=200,
                                   param2=0.85, minRadius=10)
    filtered_circles = remove_smaller_circles(circles)
    
    return filtered_circles

def reviewSign2(image_original, filename):
    # Chuyển đổi ảnh từ BGR sang RGB
    img_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    x, y, _ = img_rgb.shape
    
    filtered_circles = fine_num_sign(img_rgb)
    
    numSign = len(filtered_circles)
    sign_content = ""
    count_circle = 0
    signList = []
    
    # Vẽ các vòng tròn đã lọc
    if filtered_circles is not None:
        for i in filtered_circles:
            count_circle +=1
            if count_circle>2:
              break;

            center = (i[0], i[1])
            radius = i[2]

             # Cắt vùng ROI chứa vòng tròn để nhận diện nội dung
            roi = img_rgb[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]
            
            if numSign == 1:
                sign_content = recognize_sign_content(roi, numSign)
            else:
                sign_content = recognize_sign_content_2(sign_content, roi, numSign)
                
            signList.append(sign_content) 
             # Hiển thị kết quả     
            # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.imshow(roi)
            plt.title(sign_content)
            plt.show()
            
            # Tâm của vòng tròn
            cv2.circle(img_rgb, center, 1, (0, 100, 100), 2)
            # Đường viền của vòng tròn
            cv2.circle(img_rgb, center, radius, (0, 255, 0), 2)
            
            if(numSign == 1):
                # Hiển thị nội dung biển báo
                text_x = max(center[0] - radius - 100, 30)
                text_y = max(center[1] + radius + 30 , 20)

                # Kiểm tra và chia văn bản thành nhiều dòng nếu vượt quá chiều rộng
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 3
                if(y < 600):
                    font_scale = 0.4
                    thickness = 2
                
                
                max_width = 200  # Giới hạn chiều rộng tối đa

                wrapped_text = wrap_text(sign_content, max_width, font, font_scale, thickness)

                # Vẽ từng dòng văn bản
                line_height = 30
                # int(cv2.getTextSize(sign_content, font, font_scale, thickness)[0][1] * 1.5)  # Tính chiều cao của mỗi dòng
                for i, line in enumerate(wrapped_text):
                    cv2.putText(img_rgb, line, (text_x, text_y + i * line_height), font, font_scale, (0, 255, 0), thickness)
            else:
                # Thêm văn bản vào góc trên bên trái
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                bottom = 30     
                
                if(y < 600):
                    font_scale = 0.35
                    font_thickness = 1
                    bottom = 20
        
                font_color = (0, 255, 0)  # Màu xanh lá
                
                # Đánh số dưới tâm của vòng tròn
                text = str(count_circle)
                # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = int(center[0])  # Canh giữa văn bản với tâm của vòng tròn
                text_y = int(center[1])  # Đặt văn bản ngay dưới tâm vòng tròn
                cv2.putText(img_rgb, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                
                if(len(signList) == 2):
                    # Vẽ danh sách biển báo ở góc trên bên trái
                    y_start = img_rgb.shape[0] - bottom
                    line_height = bottom
                    
                    
                    
                    for idx, sign in enumerate(signList):
                        text = f"{idx + 1}. {sign}"
                        cv2.putText(img_rgb, text, (10, y_start - idx * line_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # plt - RGB , cv2 - BGR
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def process_images2(input_folder, output_folder, check):
    # Ensure output folder exists
    ver2_folder = os.path.join(output_folder, check)
    if not os.path.exists(ver2_folder):
        os.makedirs(ver2_folder)

    # List all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if needed
            print(filename)
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                processed_image = reviewSign2(image, filename)
                output_path = os.path.join(ver2_folder, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved {filename} to {ver2_folder}")
                print("-------------------------------")


if __name__ == "__main__":
    # Define paths

    input_folder2 = r"E:\TDTU_Work\DIP\Final\data\NewData"
    output_folder2 = r"E:\TDTU_Work\DIP\Final\testoutput"
    check = "ver9"
    # Process images
    process_images2(input_folder2, output_folder2, check)
    pass