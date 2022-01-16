import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import scipy

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def find_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 220])
    upper_white = np.array([179, 35, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(image, image, mask= mask)

    # cv2.imshow('image', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return res

# input: colored RGB image
# output: a new image, where any pixel that touches (orthogonally or diagonally) a pure white pixel
#         is set to be pure white and all other pixels are set to black
def keep_only_white( image ):
    height = image.shape[0]
    width = image.shape[1]
    dstImage = np.zeros( ( height, width ), np.uint8 )
    for row in range( height ):
        for col in range( width ):
            # touchesPureWhite = False
            # for dR in range( -1, 2 ):
            #     for dC in range( -1, 2 ):
            #         r = min( height - 1, max( 0, row + dR ) )
            #         c = min( width - 1, max( 0, col + dC ) )
            #         p = image[r][c]
            #         if p[0] == 255 and p[1] == 255 and p[2] == 255:
            #             touchesPureWhite = True
            #             break
            # if touchesPureWhite:
            #     dstImage[row][col] = 255
            p = image[row][col]
            if p[0] == 255 and p[1] == 255 and p[2] == 255:
                dstImage[row][col] = 255

    return dstImage

if __name__ == "__main__":
    filename = "test_images/test2.png"
    img = cv2.imread( filename )
    if img is None:
        print( "Could not load image '" + filename + "'" )
        exit( 0 )

    img2 = find_white( img )
    imgGray = get_grayscale( img2 )
    #cv2.imwrite( "test_images/test2_gray3.png", imgGray )

    threshVal, threshImg = cv2.threshold( imgGray, 230, 255, cv2.THRESH_BINARY )
    #threshImg = keep_only_white( img )

    #plt.imshow( threshImg, cmap='gray' )
    #plt.show()

    # no idea what this config does, just copied from the blog
    custom_config = r'--oem 3 --psm 6'
    res = pytesseract.image_to_string( threshImg, config=custom_config )
    print( res )