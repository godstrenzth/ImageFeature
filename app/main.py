import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import base64
from app.hog import hog1
# Load the image as grayscale
# img_gray = cv2.imread("image_data", 0)
# cv2.imshow("Image",image_data)
# cv2.waitKey(0)
# win_size = (32,32)
# # win_size = image_data.shape
# cell_size = (8, 8)
# block_size = (16, 16)
# block_stride = (8, 8)
# num_bins = 9
# # Set the parameters of the HOG descriptor using the variablesdefined above
# hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
# # Compute the HOG Descriptor for the gray scale image
# hog_descriptor = hog.compute(image_data)


# def hog1(img_data):
#     size=(128,128)
#     new_img =cv2.resize(img_data,size,interpolation=cv2.INTER_AREA)
#     # img_gray = cv2.imread("img", 0)
#     win_size = new_img.shape
#     # win_size = image_data.shape
#     cell_size = (8, 8)
#     block_size = (16, 16)
#     block_stride = (8, 8)
#     num_bins = 9
#     # Set the parameters of the HOG descriptor using the variablesdefined above
#     hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
#     # Compute the HOG Descriptor for the gray scale image
#     hog_descriptor = hog.compute(new_img)
#     return hog_descriptor


def read64(uri):
    img_data =uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(img_data),np.uint8)
    img =cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


app =FastAPI()

@app.get("/")
def root():
    return {"message": "This is my api"}
@app.get("/api/gethog")
async def read_str(request: Request):
    item = await request.json()
    item_str = item['img']
   
    img =read64(item_str)
    hog =hog1(img)
    return {"HOG Descriptor": hog.tolist()}

