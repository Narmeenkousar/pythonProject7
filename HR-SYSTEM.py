import cv2
import dlib

def recommend_hairstyles(image_path):
    # Placeholder code for recommending hairstyles
    recommended_hairstyles = [
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/1.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/2.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/3.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/4.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/5.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/6.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/7.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/8.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/9.png',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Hairstyles/10.png'
    ]
    return recommended_hairstyles

def apply_hair_filter(image_path, hairstyle_paths):
    # Load the input image
    image = cv2.imread(image_path)

    # Initialize the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/Users/ADMIN/PycharmProjects/pythonProject7/shape_predictor_68_face_landmarks.dat')

    for hairstyle_path in hairstyle_paths:
        # Load the current hairstyle image
        hairstyle = cv2.imread(hairstyle_path, -1)

        # Convert the hairstyle image to RGBA format
        hairstyle_rgba = cv2.cvtColor(hairstyle, cv2.COLOR_BGR2BGRA)

        # Create a copy of the input image to apply the hairstyle
        result_image = image.copy()

        # Detect faces in the input image
        gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_image)

        for face in faces:
            # Get the facial landmarks for the current face
            landmarks = predictor(gray_image, face)

            # Calculate the center of the forehead
            forehead_center = ((landmarks.part(21).x + landmarks.part(22).x) // 2,
                               (landmarks.part(21).y + landmarks.part(22).y) // 2)

            # Calculate the position to place the hairstyle
            hairstyle_x1 = forehead_center[0] - 250  # Half the width of the hairstyle (500 / 2)
            hairstyle_y1 = forehead_center[1] - 250  # Half the height of the hairstyle (500 / 2)

            # Ensure the hairstyle is within the image boundaries
            if hairstyle_x1 < 0:
                hairstyle_x1 = 0
            if hairstyle_y1 < 0:
                hairstyle_y1 = 0

            # Calculate the position to place the hairstyle
            hairstyle_x2 = hairstyle_x1 + 500  # Width of the hairstyle
            hairstyle_y2 = hairstyle_y1 + 500  # Height of the hairstyle

            if hairstyle_x2 > result_image.shape[1]:
                hairstyle_x2 = result_image.shape[1]
                hairstyle_x1 = hairstyle_x2 - 500
            if hairstyle_y2 > result_image.shape[0]:
                hairstyle_y2 = result_image.shape[0]
                hairstyle_y1 = hairstyle_y2 - 500

            # Extract the alpha channel from the hairstyle image
            alpha_channel = hairstyle_rgba[:, :, 3] / 255.0

            # Calculate the region of interest (ROI) for placing the hairstyle
            roi = result_image[hairstyle_y1:hairstyle_y2, hairstyle_x1:hairstyle_x2]

            # Perform image blending to overlay the hairstyle on the face
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + hairstyle_rgba[:, :, c] * alpha_channel

        # Display the result image with the overlayed hairstyle
        cv2.imshow("Hairstyle Overlay", result_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # List of user image paths
    user_image_paths = [
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Test img/b.jpg',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Test img/c.jpg',
        'C:/Users/ADMIN/PycharmProjects/pythonProject7/Images/Test img/d.jpg',
    ]

    # Recommend hairstyles
    recommended_hairstyles = recommend_hairstyles(user_image_paths[0])  # Use the first user image for hairstyle recommendations

    # Display recommended hairstyles
    print("Recommended hairstyles:")
    for i, hairstyle in enumerate(recommended_hairstyles):
        print(f"{i + 1}. {hairstyle}")

    # Selectively apply hairstyles to each user image
    for user_image_path in user_image_paths:
        image = cv2.imread(user_image_path)

        cv2.imshow('Image', image)
        cv2.waitKey(0)

        selected_hairstyles = []
        while True:
            selected_hairstyle = input("Enter the number of the hairstyle you want to apply (or 'q' to quit): ")
            if selected_hairstyle.lower() == 'q':
                break
            try:
                selected_hairstyle = int(selected_hairstyle)
                if 1 <= selected_hairstyle <= len(recommended_hairstyles):
                    hairstyle_path = recommended_hairstyles[selected_hairstyle - 1]
                    selected_hairstyles.append(hairstyle_path)
                else:
                    print("Invalid hairstyle number. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a valid number or 'q' to quit.")

        # Apply selected hairstyles to the current user image
        if selected_hairstyles:
            apply_hair_filter(user_image_path, selected_hairstyles)
        else:
            print("No hairstyles selected for this image.")
