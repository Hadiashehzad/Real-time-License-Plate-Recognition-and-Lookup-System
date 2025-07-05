
import cv2
from ultralytics import YOLO
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


model = YOLO(r'E:\computer vision\PA2_CNN1\PA2\best.torchscript',task='detect')

def detect_Np(image_path):
  img = cv2.imread(image_path) # Read the image
  results = model(img) # Run YOLO model on the image
  x1, y1, x2, y2 = results[0].boxes[0].xyxy[0].cpu().numpy() # Process YOLO results
  cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # Draw rectangle around the detected object
  label = "License Plate"
  cv2.putText(img, f"{label}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img_rgb,x1, y1, x2, y2

def crop_image(img, x1, x2, y1, y2):
  height, width = img.shape[:2]
  x1 = max(0, x1)
  y1 = max(0, y1)
  x2 = min(width, x2)
  y2 = min(height, y2)
  cropped_img = img[y1:y2, x1:x2]
  return cropped_img

# helper function for countors identification
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or 15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    img_res = []
    widths = []
    heights = []
    contours = []  # Store contour coordinates here

    # Calculate the middle line of the license plate
    middle_line = img.shape[0] // 2
    # Sort contours based on x-coordinate and then on y-coordinate
    sorted_cntrs = sorted(cntrs, key=lambda c: (c[0][0][0], c[0][0][1]))

    # Separate contours above and below the middle line
    above_middle = []
    below_middle = []
    for cntr in sorted_cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        if y < middle_line:
            above_middle.append((x, y, w, h, cntr))
        else:
            below_middle.append((x, y, w, h, cntr))

    # Sort contours from left to right
    above_middle = sorted(above_middle, key=lambda c: c[0])
    below_middle = sorted(below_middle, key=lambda c: c[0])

    width_margin = 0.85
    height_margin = 0.85
    # Process contours above the middle line
    for x, y, w, h, cntr in above_middle:
        # if w > lower_width and w < upper_width and h > lower_height and h < upper_height:
        if (lower_width * width_margin < w < upper_width / width_margin and
        lower_height * height_margin < h < upper_height / height_margin):
            char = img[y:y+h, x:x+w]
            white_pixels = np.sum(char == 255)
            total_pixels = char.size
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage >= 25:
                x_cntr_list.append((x, y))  # stores the (x, y) coordinates of the character's contour
                widths.append(w)
                heights.append(h)
                contours.append((x, y, x+w, y+h))  # Append contour coordinates

                char_copy = np.zeros((44, 24))
                # Extracting each character using the enclosing rectangle's coordinates.
                char = cv2.resize(char, (20, 40))
                cv2.rectangle(ii, (x, y), (x+w, y+h), (50, 21, 200), 2)
                plt.imshow(ii, cmap='gray')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with a black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0
                # print("binary image:")
                # cv2_imshow(char_copy)
                # print("end")
                img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
                # appends L E 1 D 3

    # Process contours below the middle line
    # count=0
    for x, y, w, h, cntr in below_middle:
        if (lower_width * width_margin < w < upper_width / width_margin and
        lower_height * height_margin < h < upper_height / height_margin):
        # if w > lower_width and w < upper_width and h > lower_height and h < upper_height:
            # count+=1
            # print("yo",count)
            char = img[y:y+h, x:x+w]
            white_pixels = np.sum(char == 255)
            total_pixels = char.size
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage >= 25:
                x_cntr_list.append((x, y))  # stores the (x, y) coordinates of the character's contour
                widths.append(w)
                heights.append(h)
                contours.append((x, y, x+w, y+h))  # Append contour coordinates

                char_copy = np.zeros((44, 24))
                # Extracting each character using the enclosing rectangle's coordinates.
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(ii, (x, y), (x+w, y+h), (50, 21, 200), 2)
                plt.imshow(ii, cmap='gray')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with a black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0
                # print("binary image:")
                # cv2_imshow(char_copy)
                # print("end")
                img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
                # appends 6 0 0 1

    # Calculate median width and height
    median_width = st.median(widths)
    median_height = st.median(heights)

    # Filter characters based on width and height deviation from the median
    filtered_img_res = []
    filtered_contours = []
    for char, contour in zip(img_res, contours):
        x1, y1, x2, y2 = contour
        if ((x2 - x1) >= 0.70 * median_width) and ((x2 - x1) <= 1.3 * median_width) and ((y2 - y1) >= 0.70 * median_height) and ((y2 - y1) <= 1.3 * median_height):
            filtered_img_res.append(char)
            filtered_contours.append(contour)
            # print("filtered image:")
            # cv2_imshow(char)
            # print("end")
            # appends L E D 3 6 0 0 1

    # Remove contours with a distance of more than 15 pixels between them
    remaining_contours = []
    remaining_filtered_img_res = []
    if len(filtered_contours) > 1:
        center_x = img.shape[1] / 3
        distance = 0
        count = 1
        for i in range(1, len(filtered_contours)):
            if count >= len(filtered_contours):
                break
            x1_prev, _, x2_prev, _ = filtered_contours[count - 1]
            x1_curr, _, x2_curr, _ = filtered_contours[count]
            distance = x1_curr - x2_prev
            # print (f"Distance between contours {count} and {count+1}: {distance}")
            if distance <= -175:
                count += 1
                continue
            elif distance <= 15:
                # print("if (distance <= 15):")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count += 1
            elif x1_curr < center_x:
                # print("elif x1_curr < center_x:")
                remaining_contours.append(filtered_contours[count])
                remaining_filtered_img_res.append(filtered_img_res[count])
                count +=2
            elif x1_curr >= (center_x * 2):
                # print("elif x1_curr >= center_x:")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count +=2
            else:
                # print("else:")
                remaining_contours.append(filtered_contours[count - 1])
                remaining_filtered_img_res.append(filtered_img_res[count - 1])
                count += 1
        # print("Last contour")
        remaining_contours.append(filtered_contours[-1])
        remaining_filtered_img_res.append(filtered_img_res[-1])
    # plt.show()
    # for img in remaining_filtered_img_res:
    #   print("remaining filtered image:")
    #   cv2_imshow(img)
    #   print("end")
    # returns L E D 6 0 0 1
    return np.array(remaining_filtered_img_res)

# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    LP_HEIGHT=256
    LP_WIDTH = 256
    # resize image
    # code here
    img_resized = cv2.resize(image, (LP_WIDTH, LP_HEIGHT))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray)

    _, img_binary_lp = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    closed_image = cv2.morphologyEx(img_binary_lp, cv2.MORPH_CLOSE, kernel)
    img_binary_lp = closed_image
    # Make borders white
    # code here
    img_binary_lp[0:5, :] = 255  # Top border
    img_binary_lp[:, 0:5] = 255  # Left border
    img_binary_lp[LP_HEIGHT-5:LP_HEIGHT, :] = 255  # Bottom border
    img_binary_lp[:, LP_WIDTH-5:LP_WIDTH] = 255  # Right border
    image_rgb = cv2.cvtColor(img_binary_lp, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/10,
                  2*LP_WIDTH/2.5,
                  LP_HEIGHT/20,
                  2*LP_HEIGHT/2.5]

    cv2.imwrite('contour.jpg', img_binary_lp)
    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)
    # char_list = img_binary_lp

    return char_list
def fix_dimension(img):
    img = cv2.resize(img, (28, 28))  # Resize to model input size
    return img

def show_results(model, char_images, idx_to_class, device='cpu'):
    model.eval()
    output = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Match training preprocessing
    ])

    with torch.no_grad():
        for img in char_images:
            img = fix_dimension(img)

            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)

            if len(img.shape) == 2:  # (H, W)
                img = np.expand_dims(img, axis=-1)  # (H, W, 1)

            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0).float().to(device)

            # Predict
            logits = model(img_tensor)
            _, pred = torch.max(logits, dim=1)
            pred_char = idx_to_class[pred.item()][-1]
            output.append(pred_char)

    return ''.join(output)
