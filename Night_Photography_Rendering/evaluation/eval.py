import skimage
from skimage.metrics import structural_similarity as ssim
import cv2
from math import log10, sqrt
import csv
import glob
import numpy as np

# Create a CSV file
evaluation_metrics = "evaluation_MBPNet.csv"
fields = ['IMAGE', 'SSIM', 'MSE', 'PSNR']

dir1 = "/home/suresh/Desktop/Night_Photography_Rendering/evaluation/proposed_model/GT"   #ground truth images dataset
dir2 = "/home/suresh/Desktop/Night_Photography_Rendering/evaluation/proposed_model/Shivam"  #model trained images dataset

image_files1 = glob.glob(f'{dir1}/*')
image_files2 = glob.glob(f'{dir2}/*')

if not image_files1 or not image_files2:
    print("No images found in one or both directories.")
else:
    image_files1.sort()
    image_files2.sort()

    with open(evaluation_metrics, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i in range(len(image_files1)):
            #print("i=", i)
            # Read images
            before_img = cv2.imread(image_files1[i])
            after_img = cv2.imread(image_files2[i])

            # Convert to grayscale
            before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

            # SSIM
            score, diff = ssim(before_gray, after_gray, full=True)

            # MSE
            def mse(before_gray, after_gray):
                h, w = before_gray.shape
                diff_mse = cv2.subtract(before_gray, after_gray)
                err = np.sum(diff_mse ** 2)
                mse = err / (float(h * w))
                return mse

            error = mse(before_gray, after_gray)

            # PSNR
            def PSNR(original, compressed):
                mse = np.mean((original - compressed) ** 2)
                if mse == 0:
                    return 100
                max_pixel = 255.0
                psnr = 20 * log10(max_pixel / sqrt(mse))
                return psnr

            value = PSNR(before_gray, after_gray)

            # Write results to CSV
            writer.writerow([image_files1[i], score, error, value])
    print("Values added successfully!")
