from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
# More info:
# https://docs.opencv.org/master/d5/d48/samples_2python_2stitching_8py-example.html
# https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/

def get_arguments():
    parser = argparse.ArgumentParser(description="Script that stiches images together")

    parser.add_argument("-i", "--images", nargs="*",
        type=str, required=True, dest="images",
	    help="path to input directory of images to stitch")
    parser.add_argument("-o", "--output",
        type=str, required=True, dest="output",
	    help="path to the output image")
    args = parser.parse_args()

    # Validation
    if len(args.images) < 2:
        parser.error("Need more than 2 images.")

    return args.images, args.output


def get_images(imgs):
    print("[INFO] loading images...")
    images = []
    for imagePath in imgs:
	    image = cv2.imread(imagePath)
	    images.append(image)
    return images 

if __name__ == "__main__":
    
    imgs, out = get_arguments()
    
    images = get_images(imgs)

    # stitching
    mode = cv2.Stitcher_PANORAMA
    print("[INFO] stitching images...")
    stitcher = cv2.Stitcher_create(mode)
    (status, stitched) = stitcher.stitch(images)
    print("[INFO] image stitching complete")
    if status == 0:
	    cv2.imwrite(out, stitched)
	    print("[INFO] image stitching complete")
    else:
	    print("[INFO] image stitching failed ({})".format(status))

