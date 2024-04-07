import cv2
import os

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

# Create a main folder to store the captured images
main_folder_path = "C:\\Users\\tuanb\\Downloads\\EE4208"
if not os.path.exists(main_folder_path):
    os.makedirs(main_folder_path)

# Define the number of people
num_people = 9

# Define the number of images per person
images_per_person = 100

# Capture images for each person
for person_id in range(1, num_people + 1):
    # Create a subfolder for each person
    person_folder_path = os.path.join(main_folder_path, f"Person_{person_id}")
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path)

    print(f"Capturing images for Person {person_id}")

    def detect_bounding_box(vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(100, 100))
        cropped_faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 0)
            cropped_faces.append(vid[y:y+h, x:x+w])
        return faces, cropped_faces

    def capture_images():
        image_count = 0
        total_image_count = 0
        while total_image_count < images_per_person:
            result, video_frame = video_capture.read()  # read frames from the video
            if result is False:
                break  # terminate the loop if the frame is not read successfully

            faces, _ = detect_bounding_box(video_frame)  # apply the function we created to the video frame

            for (x, y, w, h) in faces:
                # Save the face region as an image
                face_img = video_frame[y:y+h, x:x+w]
                image_path = os.path.join(person_folder_path, f"image_{total_image_count}.jpg")
                cv2.imwrite(image_path, face_img)
                total_image_count += 1
                image_count += 1

            cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame
            cv2.waitKey(200)  # add a minimal delay to allow imshow to refresh

            # Check if it's time to prompt for 'p' key press
            if image_count >= 100:
                print("Press 'p' key to continue capturing images for the next 100.")
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('p'):
                        image_count = 0
                        break

        print(f"Images captured successfully for Person {person_id}")

    # Capture images for the current person
    capture_images()

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

