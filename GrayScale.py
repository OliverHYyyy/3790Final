import cv2
import os
import numpy as np

# Helper functions for PCA and image processing
#1,2
def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                   # Load grayscale image
        if image is not None:
            image_resized = cv2.resize(image, (50,50))           # Resize to normalize size
            images.append(image_resized)
    return np.array(images)
#3
def flatten_images(images):
    num_images, height, width = images.shape
    return images.reshape(num_images, height * width)                # Flatten to 1D array
#4
def calculate_mean_face(flattened_images):
    return np.mean(flattened_images, axis=0)
#5
def demean_faces(flattened_images, mean_face):
    return flattened_images - mean_face
#6,7
def calculate_covariance_matrix(demeaned_faces):
    num_faces = demeaned_faces.shape[0]
    return np.dot(demeaned_faces.T, demeaned_faces) / num_faces
def pca(demeaned_faces, num_components):
    covariance_matrix = calculate_covariance_matrix(demeaned_faces)
    eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    eig_vecs = eig_vecs[:, :num_components]
    return eig_vecs

def project_faces(demeaned_faces, eig_vecs):
    return np.dot(demeaned_faces, eig_vecs)
#8
def recognize_face(test_image, mean_face, eig_vecs, projected_train_faces, labels):

    test_image = cv2.resize(test_image, (50, 50))
    test_image_flat = test_image.flatten()
    test_image_demeaned = test_image_flat - mean_face
    test_image_projected = np.dot(test_image_demeaned, eig_vecs)
    distances = np.linalg.norm(projected_train_faces - test_image_projected, axis=1)
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
            if file.endswith(".jpg"):  # Support multiple image formats
                test_image_path = os.path.join(root, file)
                test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
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

    for root, _, files in sorted(os.walk(training_folder)):
        #count = 0
        #for file in files:
            #count += 1

        #if count > 5:
            for file in files:
                if file.endswith(".jpg"):  # Support multiple image formats
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