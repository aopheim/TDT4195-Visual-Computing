import numpy as np      #n-D arrays
import matplotlib.pyplot as plt             # For plotting
from PIL import Image
import math

def img_save(img, filename):
    print("Saving...")
    img.save(filename, "PNG")

def img_show(img):
    plt.figure()
    plt.imshow(img, cmap='gray')     # have to use the gray color map.
    plt.axis('off')
    plt.show()





def conv2Grey_avg(img):
    img_arr = np.array(img) # convert from image object to array
    r = img_arr[:,:, 0] # equivalent to img[...,0]
    g = img_arr[:,:, 1]
    b = img_arr[:,:, 2]

    grey = np.zeros((len(r), len(r[0])), np.uint8)

    print("Converting to grey...")
    for row in range(0, len(r)-1):
        for col in range(0, len(r[0])-1):
            grey[row][col] = (int(r[row][col]) + int(g[row][col]) + int(b[row][col])) / 3          # grey tone with the average of the RGB values

    img = Image.fromarray(grey) # converting back to image object

    return img

def conv2Grey_lum(img):
    img_arr = np.array(img) # convert from image object to array
    r = img_arr[:,:, 0] # equivalent to img[...,0]
    g = img_arr[:,:, 1]
    b = img_arr[:,:, 2]

    grey = np.zeros((len(r), len(r[0])), np.uint8)

    for row in range(0, len(r)-1):
        for col in range(0, len(r[0])-1):
            grey[row][col] = int(0.2126*r[row][col] + 0.7152*g[row][col] + 0.0722*b[row][col])

    img = Image.fromarray(grey) # converting back to image object
    return img

def intensity_tran(img):
    # img is already a greyscale image.
    img_arr = np.array(img) #array of the greyscale image

    p_k = np.amax(img_arr)

    # Creating a dictionary for the intensity transformation to reduce processing time
    T_dict = {}
    for i in range(0, p_k + 1):
        T_dict[i] = p_k - i

    print("Doing intensity transformation...")


    for row in range(0, len(img_arr)):
        for col in range(0, len(img_arr[0])):
            img_arr[row][col] = T_dict[img_arr[row][col]]


    img = Image.fromarray(img_arr)
    return img

def gamma_tran(img, c, gamma):
    img_arr = np.array(img)

    #Normalizing values to [0,1]:
    max = np.amax(img_arr)
    img_arr_float = img_arr.astype(float)
    conv_dict={}
    for i in range(0, max + 1):
        conv_dict[i] = i / max
    print("Performing gamma transformation...")
    for i in range(0, len(img_arr)):
        for j in range(0, len(img_arr[0])):
            img_arr_float[i][j] = conv_dict[img_arr[i][j]]          # using the dictionary
            img_arr_float[i][j] = c * math.pow(img_arr_float[i][j], gamma) #applying the gamma function
            img_arr[i][j] = img_arr_float[i][j] * max       #converting back to original range for saving the image correctly.

    img = Image.fromarray(img_arr)
    return img

def convolution(kernel, img):
    kernel_height = len(kernel)
    ker_h_2 = int(kernel_height/2)

    kernel_width = len(kernel[0])
    ker_w_2 = int(kernel_width/2)

    img_arr = np.array(img)

    if (img_arr.ndim == 3):     #color picture
        print("Applying convolution filtering for color image...")
        img_conv = np.zeros((len(img_arr), len(img_arr[0]), img_arr.shape[2]), dtype=np.uint8)
        for i in range(ker_h_2, len(img_arr) - ker_h_2):
            for j in range(ker_w_2, len(img_arr[0]) - ker_w_2):     #Skipping the outer row/coloumn to not go out of index when applying convolution filter.
                for k in range(0, img_arr.shape[2]):        # shape returns e.g. (50, 50, 3)
                    img_conv[i, j, k] = np.sum(np.multiply(kernel, img_arr[i - ker_h_2 : i + ker_h_2 + 1, j - ker_w_2 : j + ker_w_2 + 1, k]))

    else:       #greyscale picture
        img_conv = np.zeros((len(img_arr), len(img_arr[0])), dtype=np.uint8)
        print("Applying convolution filtering for greyscale image...")
        for i in range(ker_h_2, len(img_arr) - ker_h_2):
            for j in range(ker_w_2, len(img_arr[0]) - ker_w_2):     #Skipping the outer row/coloumn to not go out of index when applying convolution filter.
                img_conv[i, j] = np.sum(np.multiply(kernel, img_arr[i - ker_h_2 : i + ker_h_2 + 1, j - ker_w_2 : j + ker_w_2 + 1]))

    img = Image.fromarray(img_conv)
    return img



def main():
    img_anna = Image.open("./images/DSC_1242_reduced.jpg")
    img_anna_grey_avg = conv2Grey_avg(img_anna)
    '''
    img_anna_grey_lum = conv2Grey_lum(img_anna)
    img_save(img_anna_grey_avg, "./img_proc/img_anna_grey_avg.png")
    img_save(img_anna_grey_lum, "./img_proc/img_anna_grey_lum.png")
    '''

    #img_std = Image.open("./images/terraux.png")
    #img_std_grey_avg = conv2Grey_avg(img_std)
    #img_std_grey_lum = conv2Grey_lum(img_std)


    # Inversion transformation:
    #img_std_intens = intensity_tran(img_std_grey_avg)
    #img_anna_intens = intensity_tran(img_anna_grey_avg)

    # Gamma transformation:
    #img_anna_gamma = gamma_tran(img_anna_grey_avg, 1, 2.5)
    #img_std_gamma = gamma_tran(img_std_grey_avg, 1, 2.5)


    # Convolution filtering:
    '''
    kernel_test = np.array([
        [1, 2, 3],
        [4, 5, 4],
        [3, 2, 1]
        ])
    kernel_test = np.multiply(kernel_test, 1/25)


    kernel_avg = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel_avg = np.multiply(1/9, kernel_avg)

    kernel_gauss = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    kernel_gauss = np.multiply(1/256, kernel_gauss)
    '''
    gradient_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    gradient_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    #img_std_conv_avg = convolution(kernel_avg, img_std_grey_avg)
    #img_anna_conv_avg = convolution(kernel_avg, img_anna_grey_avg)
    #img_anna_conv_gauss = convolution(kernel_gauss, img_anna)
    #img_anna_conv_test = convolution(kernel_test, img_anna_grey_avg)
    img_anna_gradient_x = convolution(gradient_x, img_anna_grey_avg)
    img_anna_gradient_y = convolution(gradient_y, img_anna_grey_avg)

    img_anna_gradient_x = np.array(img_anna_gradient_x)
    img_anna_gradient_y = np.array(img_anna_gradient_x)

    img_anna_gradient_x  = img_anna_gradient_x.astype(np.int32)
    img_anna_gradient_y  = img_anna_gradient_y.astype(np.int32)

    x_sq = np.square(img_anna_gradient_x)
    y_sq = np.square(img_anna_gradient_y)
    img_sum = np.add(x_sq, y_sq)
    img_sqrt = np.sqrt(img_sum)
    img_sqrt = img_sqrt.astype(np.uint8)
    print(img_sqrt)
    img_sqrt = Image.fromarray(img_sqrt)


    #img_save(img_anna_gradient_x, "./img_proc/img_anna_gradient_x.png")
    img_save(img_sqrt, "./img_proc/img_anna_gradient_magn.png")
    #img_save(img_std_conv_avg, "./img_proc/img_std_conv_avg.png")
    #img_save(img_std_gamma, "./img_proc/img_std_gamma.png")


if __name__ == "__main__":
	main()
