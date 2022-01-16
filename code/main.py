from email.mime import image
import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab

def load_image( filename ):
    img = cv2.imread( filename )
    if img is None:
        print( "Could not load image '" + filename + "'" )
        exit( 0 )
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    return img

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

def invert_image( image ):
    res = cv2.bitwise_not(image)
    return res

def take_screenshot():
    snapshot = ImageGrab.grab()
    snapshot.save( "test_images/LT_fullscreen_3.png" )

def get_names( image ):
    height = image.shape[0]
    width = image.shape[1]

    first_name_col_center = 0.2289
    last_name_col_center = 0.7707
    name_width = (last_name_col_center - first_name_col_center)/11

    first_name_row_center = 0.1764
    last_name_row_center = 0.5618
    name_height = 0.0139
    name_height_separation = (last_name_row_center - first_name_row_center)/4

    # no idea what this config does, just copied from the blog
    custom_config = r'--oem 3 --psm 6'
    all_names = []
    all_images = []

    current_row_center = first_name_row_center
    for name_row_index in range(5):
        all_names.append( 12 * [ None ] )
        all_images.append( 12 * [ None ] )
        name_row_start = int(height * (current_row_center - name_height/2))
        name_row_end = int(height * (current_row_center + name_height/2))

        current_col_center = first_name_col_center
        for name_col_index in range(12):
            name_col_start = int(width * (current_col_center - name_width/2))
            name_col_end = int(width * (current_col_center + name_width/2))
            all_images[name_row_index][name_col_index] = image[name_row_start:name_row_end, name_col_start:name_col_end]
            all_names[name_row_index][name_col_index] = pytesseract.image_to_string(all_images[name_row_index][name_col_index])

            # if all_names[name_row_index][name_col_index].strip() != "":
            print(all_names[name_row_index][name_col_index].strip())
            current_col_center += name_width
        current_row_center += name_height_separation

    cv2.imshow('image', all_images[4][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return all_names

# returns 5 x 12 arrary of all cubicles
def get_cropped_images( img ):
    height = img.shape[0]
    width = img.shape[1]

    startX = 539 / 2560
    endX = 2020 / 2560
    deltaX = (endX - startX) / 12
    startY = 250 / 1440
    endY = 928 / 1440
    deltaY = (endY - startY) / 5
    # for r in range( 6 ):
    #     y = int( height * (startY + r*deltaY) )
    #     cv2.line( img, (0, y), (width, y), (255, 0, 0), thickness=2 )
    # for c in range( 13 ):
    #     x = int( width * (startX + c*deltaX) )
    #     cv2.line( img, (x, 0), (x, height), (255, 0, 0), thickness=2 )

    croppedImages = []
    for r in range( 5 ):
        croppedImages.append( 12 * [ None ] )
        for c in range( 12 ):
            x1 = int( width * (startX + c*deltaX) )
            x2 = int( width * (startX + (c+1)*deltaX) )
            y1 = int( height * (startY + r*deltaY) )
            y2 = int( height * (startY + (r+1)*deltaY) )
            croppedImages[r][c] = img[y1:y2, x1:x2]

    # for r in range( 5 ):
    #     for c in range( 12 ):
    #         plt.imshow( croppedImages[r][c] )
    #         plt.show()

    return croppedImages

def create_final_display_image( croppedImgs, trolls ):
    if len( trolls ) == 0:
        img = load_image( "no_trolls.png" )
        cv2.imshow('cropped', img )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for troll in trolls:
            row = troll[0]
            col = troll[1]

if __name__ == "__main__":
    filename = "test_images/LT_fullscreen_1.png"
    img = load_image( filename )

    croppedImgs = get_cropped_images( img )

    img2 = find_white( img )
    imgGray = cv2.cvtColor( img2, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite( "test_images/test2_gray3.png", imgGray )
    threshVal, threshImg = cv2.threshold( imgGray, 230, 255, cv2.THRESH_BINARY )
    # cv2.imwrite( "test_images/test2_gray3.png", threshImg )

    names = get_names(threshImg)

    # #threshImg = keep_only_white( img )

    # plt.imshow( threshImg, cmap='gray' )
    # plt.show()

    # # no idea what this config does, just copied from the blog
    # custom_config = r'--oem 3 --psm 6'
    # res = pytesseract.image_to_string( threshImg, config=custom_config )
    # print( res )