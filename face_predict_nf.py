import numpy as np
import cv2
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from skimage.feature import local_binary_pattern
from scipy import ndimage

# Define function to divide image into grids and concatenate pixel values of each grid
def grid_concatenation(image, grid_size):
    height, width = image.shape
    grid_height = height // grid_size[0]
    grid_width = width // grid_size[1]
    concatenated_grids = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid = image[i * grid_height: (i + 1) * grid_height, j * grid_width: (j + 1) * grid_width]
            concatenated_grids.extend(grid.ravel())

    return concatenated_grids
def tan_trigs_preprocessing(image, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0):    #Convert to floating point:
    X = image.astype(np.float32)    #Start preprocessing:
    X = np.power(X, gamma)    
    X = np.asarray(ndimage.gaussian_filter(X, sigma1) - ndimage.gaussian_filter(X, sigma0))
    X = X/np.power(np.mean(np.power(np.abs(X), alpha)), 1.0/alpha)    
    X = X/np.power(np.mean(np.power(np.minimum(np.abs(X), tau), alpha)), 1.0/alpha)
    X = tau*np.tanh(X/tau)    #return ((X - X.min()) / (X.max() - X.min())*255).astype('uint8')
    return (X - X.min()) / (X.max() - X.min())

def get_lpb_image(image):
    # Load the image
    nimage = np.uint8(image*255)
    # Define LBP parameters
    radius = 3
    n_points = 8 * radius

    # Apply LBP
    lbp_image = local_binary_pattern(nimage, n_points, radius, method='uniform')
    return (lbp_image / np.max(lbp_image))

def rank_array(arr):
    # Enumerate the array to keep track of the original indices
    indexed_arr = list(enumerate(arr))
    
    # Sort the array based on the values
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])
    
    # Create a new array to store the ranks
    rankings = [0] * len(arr)
    
    # Assign ranks based on the sorted order
    for i, (index, value) in enumerate(sorted_arr):
        rankings[index] = i
    
    return rankings
def discriminant(x, means, Vs, lamdas):
    return [np.sum([np.real(np.square(np.subtract(np.dot(V[:,i].T,x),np.dot(V[:,i].T,mean))))/lamda[i] for i in range(len(lamda))]) for V,mean,lamda in zip(Vs,means,lamdas)]

def min_divided_by_avg(input_list):
    # Step 1: Find the minimum value in the list
    min_value = min(input_list)
    
    # Step 2: Create a new list without the minimum value
    remaining_values = [value for value in input_list if value != min_value]
    
    # Step 3: Calculate the average of the remaining values
    avg_remaining_values = sum(remaining_values) / len(remaining_values) if remaining_values else 0
    
    # Step 4: Divide the minimum value by the average of remaining values
    result = min_value / avg_remaining_values if avg_remaining_values != 0 else float('inf')
    
    return (1 - result)

def nearest_dist(input_list):
    # Step 1: Find the minimum value in the list
    min_value = min(input_list)
    
    # Step 2: Create a new list without the minimum value
    remaining_values = [value for value in input_list if value != min_value]
    
    # Step 3: Calculate the average of the remaining values
    next_min = min(remaining_values)
    
    # Step 4: Divide the minimum value by the average of remaining values
    result = min_value / (min_value + next_min)
    
    return (1 - result)

#########################################################################################################################
# Prepare
# names = ['Xuewei', 'Xavier', 'Nhan', 'Yong', 'Xuan', 'Zhuoxun', 'Tuan', 'Minh', 'Bowen']
names = ['nf', 'f']

# LDA model
base_folder = 'nf3/' # Location of the np arrays
dr = np.load(f'{base_folder}dimension_reduction.npy')
vec = np.load(f'{base_folder}eigenvectors_list_lda.npz')
vec_list = [vec[key] for key in vec.files]
val = np.load(f'{base_folder}eigenvalues_list_lda.npz')
val_list = [val[key] for key in val.files]
mean = np.load(f'{base_folder}means_lda.npz')
mean_list = [mean[key] for key in mean.files]
# CNN model
model = tf.keras.models.load_model(f'{base_folder}cnn_1.h5', compile = False)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1 = tf.keras.models.load_model(f'{base_folder}nn.h5', compile = False)
model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Haar Face Detect Model
faceDetect = cv2.CascadeClassifier('C:/users/thanh/miniconda3/envs/gprMax/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

#########################################################################################################################
count = 0
basearr = [0 for i in range(len(names))]
sum_array = basearr
sum_array1 = basearr
pi = basearr
pinn = basearr
# Real time detection

# Initialize video capture
video = cv2.VideoCapture(0)

# Create a window to display LPB image
cv2.namedWindow('LPB Image', cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
    lpb = frame
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        sub_face_img = frame[y : y + h, x : x + w]
        height, width, _ = sub_face_img.shape
        crop = 3
        data = sub_face_img[crop:height-crop, crop:width-crop]
        data = cv2.resize(data, (90, 90))
        # if count < 50:
        #     # Save the face image
        #     cv2.imwrite(f'faces{count}.png', data)
        #     count = count + 1
        # Convert RGB to grayscale
        gray_data = rgb2gray(data)

        # Preprocessing
        lpb = get_lpb_image(gray_data)
        # lpb = gray_data
        # lpb = tan_trigs_preprocessing(gray_data)

        # d = np.array(grid_concatenation(lpb, (3,3)))
        d = lpb.ravel()
        
        # Reshape the array to a 2D array
        reshaped_array = d.reshape(-1, 1)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit and transform the data using the scaler
        scaled_array = scaler.fit_transform(reshaped_array)

        # Reshape the scaled data back to a 1D array
        d = scaled_array.ravel()

        d = d.reshape(1,-1)

        x_dr = np.real(np.transpose(np.dot(dr.T, d.T)))
        pix = discriminant(x_dr[0], mean_list, vec_list,val_list)
        nnx = model1.predict(x_dr, verbose = False)
        cnnx = model.predict(d[0].reshape(1, 90, 90, 1), verbose = False)
        
        sum_array = np.array(sum_array) + np.array(pix)
        sum_array1 = np.array(sum_array1) + np.array(nnx)

        count = count + 1
        if count > 10:
            print('Updated Prediction')
            pi = np.array(sum_array)/count
            pinn = np.array(sum_array1)/count

            # pi[0] = pi[0]*8/5
            # pinn[0] = pinn[0]*5/8

            sum_array = basearr
            sum_array1 = basearr
            print(pi)
            count = 0

        prediction = np.argmin(pi)
        confidence = min_divided_by_avg(pi)
        prob_nn = np.max(pinn)
        nn_pred = np.argmax(pinn)
        cnn_pred = np.argmax(cnnx, axis=1)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(frame, f'DR Mahalanobis {names[prediction]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # cv2.putText(frame, f'DR NN {names[nn_pred]}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # cv2.putText(frame, f'CNN {names[cnn_pred[0]]}', (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if np.min(pi) < 1e+16:
            cv2.putText(frame, f'DR Mahalanobis {names[prediction]} {confidence:.2f}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'CNN {names[cnn_pred[0]]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'DR NN {names[nn_pred]} {prob_nn:.2f}', (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif np.min(pi) > 1.3e+16:
            cv2.putText(frame, f'Non-face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, f'CNN {names[cnn_pred[0]]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'Possible {names[prediction]} or {names[nn_pred]}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Display the LPB image
    cv2.imshow('LPB Image', lpb)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()