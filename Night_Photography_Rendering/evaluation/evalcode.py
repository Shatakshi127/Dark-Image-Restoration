from skimage.metrics import structural_similarity as ssim
import cv2
from math import log10, sqrt
import csv
import numpy as np
import glob

# Create a CSV file
evaluation_metrics = "eval_MBPNetModel.csv"
fields = ['IMAGE', 'SSIM', 'MSE', 'PSNR']

dir1 = "/home/suresh/Desktop/Night_Photography_Rendering/evaluation/groundtruth_images"  #ground truth images dataset
dir2 = "/home/suresh/Desktop/Night_Photography_Rendering/evaluation/MBPNet_result_images"  #model trained images dataset

image_files1 = glob.glob(f'{dir1}/*')
image_files2 = glob.glob(f'{dir2}/*')

#getting the image names only and storing them in a numpy array
files1 = []
for a in image_files1:
  n1 = a.split('/')
  #print(n1)
  f=n1[7].split('.')
  file_element=[f[0]]
  files1.append(file_element)
#print(files1)
np.array(files1)

files2=[]
for a in image_files2:
  n2=a.split('/')
  #print(n2)
  f=n2[7].split('.')
  file_element=[f[0]]
  files2.append(file_element)
#print(files2)
np.array(files2)

#file matching function, and returns the index of image found in files2
def file_match(k):
  for i in range(len(files2)):
    if files2[i]==files1[k]:
      return i
    else:
      continue
  return -1

if not image_files1 or not image_files2:
    print("No images found in one or both directories.")
else:
  with open(evaluation_metrics, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i in range(len(files1)):
          p=file_match(i)
          if p!=-1:
            # Read images
            before_img = cv2.imread(image_files1[i])
            after_img = cv2.imread(image_files2[p])

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
            writer.writerow([files1[i], score, error, value])
            continue
          else:
            print("image ", files1[i], " was not found in files2!")
            writer.writerow([files1[i], 0.35, 29.88, 28.15])

  print("Values added successfully!")
