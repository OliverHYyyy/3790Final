import cv2
import os
import numpy as np

# Helper functions for PCA and image processing
def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)  # Load RGB image
        if image is not None:
            image_resized = cv2.resize(image, (50, 50))  # Resize to normalize size
            images.append(image_resized)
    return np.array(images)

def flatten_images(images):
    num_images, height, width, channels = images.shape
    return images.reshape(num_images, height * width * channels)  # Flatten to 1D array

def calculate_mean_face(flattened_images):
    return np.mean(flattened_images, axis=0)

def demean_faces(flattened_images, mean_face):
    return flattened_images - mean_face

def calculate_covariance_matrix(demeaned_faces):
    num_faces = demeaned_faces.shape[0]
    return np.dot(demeaned_faces.T, demeaned_faces) / num_faces

def pca(demeaned_faces, num_components):
    # Compute covariance matrix
    covariance_matrix = calculate_covariance_matrix(demeaned_faces)
    # Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
    # Sort by eigenvalues
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    # Select top num_components eigenvectors
    eig_vecs = eig_vecs[:, :num_components]
    return eig_vecs

def project_faces(demeaned_faces, eig_vecs):
    return np.dot(demeaned_faces, eig_vecs)

def recognize_face(test_image, mean_face, eig_vecs, projected_train_faces, labels):
    # Preprocess the test image
    test_image = cv2.resize(test_image, (50, 50))  # Ensure test image size is 50x50
    test_image_flat = test_image.flatten()  # Flatten to 1D array
    # Compute demeaned test image
    test_image_demeaned = test_image_flat - mean_face
    # Project to feature space
    test_image_projected = np.dot(test_image_demeaned, eig_vecs)
    # Calculate Euclidean distances
    distances = np.linalg.norm(projected_train_faces - test_image_projected, axis=1)
    # Find index of nearest training image
    nearest_face_index = np.argmin(distances)
    return labels[nearest_face_index]

def get_name(file_path):
    index = len(file_path) - 1
    while index >= 0:
        if file_path[index] == '\\':
            break
        index -= 1

    # 输出 slicing 的内容
    return file_path[index+1:len(file_path) - 9]

def process_images_in_folder(folder_path, mean_face, eig_vecs, projected_train_faces, labels, image_paths):
    correct = 0.0
    total = 0.0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") and not (file.endswith("1.jpg") or file.endswith("2.jpg") or file.endswith("3.jpg") or file.endswith("4.jpg")\
                        or file.endswith("5.jpg")):  # Support multiple image formats
                test_image_path = os.path.join(root, file)
                test_image = cv2.imread(test_image_path)
                if test_image is not None:
                    label = recognize_face(test_image, mean_face, eig_vecs, projected_train_faces, labels)
                    # Compare complete path and predicted label

                    is_correct = get_name(test_image_path) ==  get_name(label)
                    if is_correct:
                        correct += 1
                    print(f"图像: {get_name(test_image_path)} -> 预测标签: {get_name(label)} -> 结果: {is_correct}")
                    total += 1

    print(correct, ', ', total)
    print('correct rate: ', (correct / total) * 100, '%')


# Example usage
if __name__ == "__main__":
    # Load training data
    training_folder = "E:\\24Spring\\lfw"
    training_image_paths = []
    training_labels = []

    label_index = 0
    for root, _, files in sorted(os.walk(training_folder)):
        count = 0
        for file in files:
            count += 1

        if count >5:
            for file in files:
                if file.endswith("1.jpg") or file.endswith("2.jpg") or file.endswith("3.jpg") or file.endswith("4.jpg")\
                        or file.endswith("5.jpg"):  # Support multiple image formats
                    image_path = os.path.join(root, file)
                    training_image_paths.append(image_path)
                    training_labels.append(image_path)  # Assign labels based on folder traversal order

    images = load_images(training_image_paths)
    flattened_images = flatten_images(images)

    # Calculate mean face
    mean_face = calculate_mean_face(flattened_images)

    # Compute demeaned faces
    demeaned_faces = demean_faces(flattened_images, mean_face)

    # Compute eigenvectors
    num_components = 100  # Select number of principal components
    eig_vecs = pca(demeaned_faces, num_components)

    # Project training images to feature space
    projected_train_faces = project_faces(demeaned_faces, eig_vecs)

    # Test recognizer
    test_folder = "E:\\24Spring\\lfw"
    process_images_in_folder(test_folder, mean_face, eig_vecs, projected_train_faces, training_labels, training_image_paths)