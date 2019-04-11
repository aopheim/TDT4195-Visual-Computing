import numpy as np      #n-D arrays
import matplotlib.pyplot as plt             # For plotting
from mpl_toolkits.mplot3d import axes3d # For 3D plotting
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


def conv2Grey_lum(img):
    img_arr = np.array(img) # convert from image object to array
    r = img_arr[:,:, 0] # equivalent to img[...,0]
    g = img_arr[:,:, 1]
    b = img_arr[:,:, 2]

    grey = np.zeros((len(r), len(r[0])), np.uint8)
    print("Converting to grey...")
    for row in range(0, len(r)-1):
        for col in range(0, len(r[0])-1):
            grey[row][col] = int(0.2126*r[row][col] + 0.7152*g[row][col] + 0.0722*b[row][col])

    img = Image.fromarray(grey) # converting back to image object
    return img



def freq_filter(img):

    def pad_image(img_arr):
        print("Padding image...")
        img_padded = np.zeros((P, Q), dtype = np.uint8)
        for i in range(0, M):
            for j in range(0, N):
                img_padded[i][j] = img_arr[i][j]
        return img_padded

    def plot_wireframe(kernel, filename):

        u = np.zeros((P, Q), dtype=np.int)
        v = np.zeros((P, Q), dtype=np.int)
        j_counter = 0
        for i in range(0, P):
            for j in range(0, Q):
                u[i] = np.arange(Q)
                v[i].fill(j_counter)
            j_counter += 1

        #print("u: ", u, "\nv: ", v, "\nkernel: ", kernel)
        #print("u: ", u.shape, "v: ", v.shape, "kernel: ", kernel.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lines = ax.plot_wireframe(u, v, kernel)
        plt.setp(lines, color='black', linewidth=0.5)
        plt.show()
        fig.savefig(filename, format='pdf') #saving as pdf


    def D(u, v): # Function for calculating the distance between a point (u, v) in the frequency domain and the center of the frequency rectangle
        term1 = (u - P/2)*(u - P/2)
        term2 = (v - Q/2)*(v - Q/2)
        #print(math.sqrt(term1 + term2))
        ans = math.sqrt(term1 + term2)
        # Not possible to divide by zero:
        if (ans < 10e-7):
            return 10e-7
        else:
            return ans

    def lowpass_gauss(D0):
        H = np.zeros((P, Q), dtype=np.float) # H is the kernel --> H(u, v)
        print("Creating Gauss kernel...")
        for u in range(0, P):
            for v in range(0, Q):
                teller = -(D(u, v) * D(u, v))
                nevner = 2 * (D0 * D0)
                eksp = teller / nevner
                H[u][v] = math.exp(eksp)

        #print(H.shape)
        #plot_wireframe(H)
        #print("H: ", H)
        #img = Image.fromarray(np.real(H))
        #img = img.convert('RGB')
        #img_save(img, "./img_proc/lowpass_gauss.png")
        return H


    def highpass_butterworth(D0, n):
        H = np.zeros((P, Q), dtype=np.float) # H is the kernel --> H(u, v)
        print("Creating high-pass Butterworth kernel...")
        for u in range(0, P):
            for v in range(0, Q):
                teller = 1
                nevner = D0 / (D(u, v))
                nevner = math.pow(nevner, 2*n)
                nevner = 1 + nevner
                H[u][v] = teller / nevner

        plot_wireframe(H, './plots/highpass_butterworth.pdf')
        return H





    def center_transform(img):
        print("Centering the image...")
        #print("Image before centering: ", img)
        centered = np.zeros((P, Q), dtype=np.float) # Have to allow negative values
        for x in range(0, P):
            for y in range(0, Q):
                eksp = x + y
                centered[x][y] = img[x][y] * math.pow(-1, eksp)
        #print("Image after centering: ", centered)
        return centered

    def inv_fourier(G):
        print("Inverse Fourier transform...")
        g_p = np.fft.ifft2(G) # performing inverse ourier transform...
        g_p = np.real(g_p) # extracting the real part...
        g_p = center_transform(g_p) # Centering
        g_p = g_p.astype(np.uint8) # Converting back to uint8 type
        return g_p

    def plot_fourier(F):
        print(F)
        # Extracting the real part of the transform:
        F = np.fft.fftshift(F)
        F_log = np.log2(1 + np.abs(F)**2)

        print(F_log)
        print("max: ", np.max(F_log))

        img = Image.fromarray(F_log)
        img = img.convert('RGB')
        img_show(img)





    img_arr = np.array(img)
    img_arr = img_arr[:, :, 0] # Removing color bands that are still there.

    # Getting image parameters: Original image of size (M, N)
    M = len(img_arr)
    N = len(img_arr[0])
    # Size of the filter kernel:
    P = 2*M
    Q = 2*N
    print("Dimensions of the image: \nM: ", M, "N: ", N, "P: ", P, "Q: ", Q)

    # Padding the image:
    img_padded = pad_image(img_arr)

    # Centering its transform:
    img_centered = center_transform(img_padded)
    #img = Image.fromarray(img_centered)
    #img = img.convert('RGB')
    #img_save(img, "./img_proc/img_sushi_centered.png")


    # Computing the Fourier transform:
    print("Computing Fourier transform...")
    img_fourier = np.fft.fft2(img_padded)
    #plot_wireframe(img_fourier, './plots/fourier_F.pdf')
    plot_fourier(img_fourier)
    #print("img_fourier: ", img_fourier)
    #img = Image.fromarray(np.real(img_fourier))
    #img = img.convert('RGB')
    #img_show(img)
    #img_save(img, "./img_proc/img_sushi_fourier.png")


    # Creating a kernel:
    kernel = lowpass_gauss(1e2)
    #kernel = highpass_butterworth(3e2, 3)

    # Multiplying element-wise kernel with Fourier transformed image:
    print("Multiplying kernel with Fourier image...")
    img_G = np.multiply(kernel, img_fourier)
    plot_fourier(img_G)
    #plot_wireframe(img_G, './plots/G_highpass.pdf' )
    #img = Image.fromarray(np.real(img_G))
    #img = img.convert('RGB')
    #img_show(img)


    # Obtaining the finished processed image:
    img_proc = inv_fourier(img_G)
    #img = Image.fromarray(img_proc)
    #img_save(img, "./img_proc/img_sushi_proc_PQ.png")

    # Clipping out the original (M, N) sized picture:
    img_proc = img_proc[0:M, 0:N]
    img = Image.fromarray(img_proc)
    img = img.convert('RGB')
    img_show(img)

    # Subtracting the low-pass image from the original grayscale:
    g_mask = np.subtract(img_arr, img_proc)
    img = Image.fromarray(g_mask)
    img = img.convert('RGB')
    img_show(img)
    img_save(img, "./img_proc_2/img_g_mask.png")

    print("img_proc: ", img_proc)
    print("img_arr: ", img_arr)
    print("g_mask: ", g_mask)

    # Multiplying the mask with a k value
    g_mask = np.multiply(1, g_mask)

    g = np.add(img_arr, g_mask)
    img = Image.fromarray(g)
    img_save(img, "./img_proc_2/img_g_k1.png")
    #img = img.convert('RGB')
    #img_show(img)

    #img = Image.fromarray(img_proc)
    #img_save(img, "./img_proc/img_sushi_proc_MN_highpass.png")
    return img


def main():
    '''
    # Creating the greyscale image:
    img = Image.open("./images/DSC_0784.JPG")
    img_grey = conv2Grey_lum(img)
    img_save(img_grey, "./img_proc_2/img_grey.png")
    '''

    img = Image.open("./img_proc/img_sushi_grey_mini.png")
    #img = Image.open("./img_proc_2/img_grey_mini.png")
    #img = Image.open("./images/noise-a.tiff")


    ## Filtering in the frequency domain: ##
    img_freq_filter = freq_filter(img)





if __name__ == "__main__":
	main()
