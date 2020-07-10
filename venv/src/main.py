from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2



def main():
    # This is just the transposition of your code in python
    img = cv2.imread('../data/Photo_Turlom_C1_1.jpg')
    #img = cv2.imread('../data/L1ZzA.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4, dst=100)
    blur2 = cv2.medianBlur(thresh, 3)
    ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur3 = cv2.GaussianBlur(th2, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # cv2.imwrite("../data/outputs/th3.jpeg", thresh)
    #
    # # noise removal
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # #opening = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel, iterations=2)
    #
    # cv2.imwrite("../data/outputs/opening.jpeg", opening)
    #
    # # sure background area
    # sure_bg = cv2.dilate(opening, kernel, iterations=1)
    #
    # cv2.imwrite("../data/outputs/sure_bg.jpeg", sure_bg)
    #
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    #
    # cv2.imwrite("../data/outputs/unknown.jpeg", unknown)
    #
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    #
    # cv2.imwrite("../data/outputs/markers.jpeg", markers)
    #
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    #
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    #
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]
    #
    # cv2.imwrite("../data/outputs/Out.jpeg", img)
    # END



    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # ret, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # # thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
    # # blur2 = cv2.medianBlur(thres, 3)
    # # ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # blur3 = cv2.GaussianBlur(th2, (3, 3), 0)
    # # ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # th3 = thres
    #
    # img_erode = cv2.erode(th3, np.ones((3, 3), np.uint8), iterations=1)
    #
    # th3 = img_erode
    #
    # # Get contours
    # contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # output = img.copy()
    #
    # for idx, contour in enumerate(contours):
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    #     # hierarchy[i][0]: the index of the next contour of the same level
    #     # hierarchy[i][1]: the index of the previous contour of the same level
    #     # hierarchy[i][2]: the index of the first child
    #     # hierarchy[i][3]: the index of the parent
    #     if hierarchy[0][idx][3] == 0:
    #         cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
    #
    # cv2.imwrite("../data/outputs/Input.jpeg", img)
    # cv2.imwrite("../data/outputs/Enlarged.jpeg", th3)
    # cv2.imwrite("../data/outputs/Output.jpeg", output)
    # cv2.waitKey(0)


    # Find connected components and extract the mean height and width
    output_orig = cv2.connectedComponentsWithStats(255 - th3, 6, cv2.CV_8U)
    output = cv2.findContours(255-th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mean_h = np.mean(output[2][:, cv2.CC_STAT_HEIGHT])
    mean_w = np.mean(output[2][:, cv2.CC_STAT_WIDTH])

    # Find empty rows, defined as having less than mean_h/2 pixels
    empty_rows = []
    for i in range(th3.shape[0]):
        if np.sum(255 - th3[i, :]) < mean_h / 2.0:
            empty_rows.append(i)

            # Group rows by labels
    d = np.ediff1d(empty_rows, to_begin=1)

    good_rows = []
    good_labels = []
    label = 0

    # 1: assign labels to each row
    # based on whether they are following each other or not (i.e. diff >1)
    for i in range(1, len(empty_rows) - 1):
        if d[i + 1] >= 1:
            good_labels.append(label)
            good_rows.append(empty_rows[i])

        elif d[i] > 1 and d[i + 1] > 1:
            label = good_labels[len(good_labels) - 1] + 1

    # 2: find the mean row value associated with each label, and color that line in green in the original image
    for i in range(label):
        frow = np.mean(np.asarray(good_rows)[np.where(np.asarray(good_labels) == i)])
        img[int(frow), :, 1] = 255

    cv2.imwrite("../data/outputs/test.jpeg", img)

if __name__ == "__main__":
    main()