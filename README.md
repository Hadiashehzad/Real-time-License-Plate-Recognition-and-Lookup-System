# Real-time-License-Plate-Recognition-and-Lookup-System
A full-stack application that detects, segments, and recognizes license plates from vehicle images using custom CNN and image processing techniques. This project presents a complete end-to-end License Plate Recognition System, designed for smart city surveillance and law enforcement. It integrates computer vision, deep learning, and web technologies to detect, recognize, and log vehicle license plates from real-world images.

<img width="482" alt="Screenshot 2025-06-26 013650" src="https://github.com/user-attachments/assets/8a6e7e95-4c74-43bd-b42c-9b87bdb819fc" />


# 🔍 Project Overview
The system automatically detects license plates from car images, segments and recognizes the characters using a trained Convolutional Neural Network (CNN), and logs the plate number with a timestamp. A Streamlit web application allows users to either search by license plate number or upload a new image for real-time plate recognition and lookup.

![cars](https://github.com/user-attachments/assets/b16dc0e8-e389-4931-a2d9-25ca7369f69a)


# Methodology
I used a pre-trained YOLO v8 model to detect lisence plates from images of cars and draw bounding boxes around them. I used the YOLO v8 backbone and trained it on a custom dataset of license plates obtained from [Roboflow](https://drive.google.com/drive/folders/1o9m2tCkLVJcJOesEX7AsMwQI37Ulq6ev?usp=sharing). Then I segmented out individual characters from the plate images using contouring and then used a custom-trained CNN model to classify the characters and obtain the plate number. Then I logged the numbers along with the timestamps into a file, to be conveniently retrieved through the StreamLit application.

Here is an outline of the technologies used:
- YOLOv5-based License Plate Detection
- Character Segmentation: Processes the cropped plate using adaptive filtering and morphological operations to isolate individual characters.
- CNN Character Recognition: Recognizes each character using a custom-trained CNN model (trained on alphanumeric plate characters).
- Image Logging & Timestamping: Saves logs of each recognized license plate along with timestamp and image label.
- Interactive Streamlit Web App

![image](https://github.com/user-attachments/assets/e07cf552-bcd2-450a-8efd-d540c7afe345)

# Preprocessing
I did the following preprocessing steps on the cropped images of lisence plates obtained from the bounding boxes detected by the YOLO v8 model:

- Resize
- Convert to grayscale
- Binary thresholding
- Morph Close
- Make borders white

![plate](https://github.com/user-attachments/assets/59bb04b0-1000-4592-939c-a251f9ccb65e)
![contour](https://github.com/user-attachments/assets/dc27251a-5004-4365-a72e-b31bd277bf43)
  
These steps would allow easy segmentation and contouring on the plate images to extract out the characters.
  
# Contouring

Having preprocessed the image I found Contours in the preprocessed cropped lisence plate and filtered them out based on size and distance to extract out separate characters from the license plate for classification. The contouring worked as follows:

- Find all contours in the binary license plate image
- Keep the 15 largest contours assuming they're more likely to be actual characters
- Split contours into those above and below the plate’s horizontal center
- Basic dimensional filter: rejects contours that are too narrow or short to be characters
- Resize character to 20x40, adds a black border to make 24x44 total
- Remove characters that are too big/small compared to average.
- Distance-based filter: remove characters that are too far apart or are on the extreme left/right

![contourss](https://github.com/user-attachments/assets/98258a49-eaae-4df6-a75e-ff61ca604bf5)

# CNN

Then I trained a Convolutional Neural Network (CNN) on a custom dataset of individual characters to be able to classify the images of separate characters extracted from the license plate. 
The dataset used for training can be found [here](https://drive.google.com/drive/folders/1jaASSDN5juH2mRfw2HCvHO4oc_ntZ9c3?usp=sharing).

<img width="609" alt="output" src="https://github.com/user-attachments/assets/3df7599b-e927-4c9f-9ed2-dfecb92c7b8d" />

## Evaluation metrics:

| Metric  | Score |
| ------------- | ------------- |
| Accuracy  | 98.9%  |
| Precision  | 99.09%  |
| Recall  | 99.10%  |
| F1 score  | 99.06%  |

# StreamLit
Finally, I deployed the model on a streamlit app that inputs license plate numbers and displays the corresponding car image along with the corresponding timestamp of when the car image was logged into the system.

🔎 Plate Lookup Mode: Enter a license plate number to retrieve associated timestamps and vehicle images.

📤 Image Upload Mode: Upload a vehicle image and automatically detect, recognize, and display the plate number, timestamp, and plate-highlighted image.

<img width="482" alt="Screenshot 2025-06-26 013650" src="https://github.com/user-attachments/assets/c50db99b-0a5f-4e50-b4f0-495ba0cb84d0" />









