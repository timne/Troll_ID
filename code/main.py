from email.mime import image
import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab
from difflib import SequenceMatcher
from pynput.keyboard import Key, Controller

def str_similarity( a,b ):
    return SequenceMatcher( None, a, b ).ratio()

def load_image( filename ):
    img = cv2.imread( filename )
    if img is None:
        print( "Could not load image '" + filename + "'" )
        exit( 0 )
    #img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    return img

def find_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 220])
    upper_white = np.array([179, 35, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(image, image, mask= mask)

    res = cv2.cvtColor( res, cv2.COLOR_RGB2GRAY )
    threshVal, res = cv2.threshold( res, 230, 255, cv2.THRESH_BINARY )

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
    snapshot.save( "tmpScreenshot.png" )

def get_names( inputRGBImage ):
    image = find_white( inputRGBImage )

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
            name = pytesseract.image_to_string( all_images[name_row_index][name_col_index], config=custom_config ).strip()
            all_names[name_row_index][name_col_index] = name

            # if name != "":
            #     print( name )
            current_col_center += name_width
        current_row_center += name_height_separation

    # cv2.imshow('image', all_images[4][0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

    cubicleWidth = int( width * deltaX )
    cubicleHeight = int( height * deltaY )
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
            x2 = x1 + cubicleWidth
            y1 = int( height * (startY + r*deltaY) )
            y2 = y1 + cubicleHeight
            croppedImages[r][c] = img[y1:y2, x1:x2]

    # for r in range( 5 ):
    #     for c in range( 12 ):
    #         plt.imshow( croppedImages[r][c] )
    #         plt.show()

    return croppedImages

def create_final_display_image( croppedImgs, names ):
    lines = open( "trolls.txt" ).readlines()
    L = len( lines )
    trollNames = L * [ "" ]
    trollReasons = L * [ "" ]
    for i in range( L ):
        s = lines[i].strip().split( ',' )
        trollNames[i] = s[0]
        if len( s ) > 1:
            trollReasons[i] = s[1]

    foundTrolls = []
    for r in range( 5 ):
        for c in range( 12 ):
            name = names[r][c]
            if name == "":
                continue
            for i in range( L ):
                strSimilarity = str_similarity( name, trollNames[i] )
                #print( "Similarity " + name + " vs " + trollNames[i] + ":", strSimilarity )
                if strSimilarity >= 0.8:
                    foundTrolls.append( (r, c) )

    img = None
    numTrolls = len( foundTrolls )
    if numTrolls == 0:
        img = load_image( "no_trolls.png" )
    else:
        cH = croppedImgs[0][0].shape[0]
        cW = croppedImgs[0][0].shape[1]
        effectiveWidth = cW + 6
        totalW = numTrolls * effectiveWidth - 6
        img = np.zeros( (cH, totalW, 3), dtype=np.uint8 )
        count = 0
        for troll in foundTrolls:
            r = troll[0]
            c = troll[1]
            img[:,count*effectiveWidth:count*effectiveWidth + cW] = croppedImgs[r][c]
            count += 1

    cv2.imshow( 'Trolls', img )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #filename = "test_images/LT_fullscreen_1.png"
    # # cv2.imwrite( "test_images/test2_gray3.png", threshImg )

    take_screenshot()
    img = load_image( "tmpScreenshot.png" )
    names = get_names( img )
    croppedImgs = get_cropped_images( img )
    create_final_display_image( croppedImgs, names )

    # keyboard = Controller()
    # keyboard.press( Key.tab )
    # keyboard.release( Key.tab )
    # take_screenshot()
    # keyboard.press( Key.tab )
    # keyboard.release( Key.tab )
