from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import pickle
import copy
import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def getAngle(p0, p1, p2):
    return math.acos((dist(p0, p1) ** 2 + dist(p0, p2) ** 2 - dist(p1, p2) ** 2) / (2 * (dist(p0, p1) ** 2) * (dist(p0, p2) ** 2)))

def getNormalisedCoordinates(x, y, brightest_index, second_brightest_index, lines=[]):
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)

    lines = np.array(lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            pass

    # Shifting the brightest star to origin
    for i in range(len(x)):
        x[len(x) - i - 1] -= x[brightest_index]
        y[len(x) - i - 1] -= y[brightest_index]

    # Distance between brightest and second brightest star
    distance_brightest = math.sqrt((x[brightest_index] - x[second_brightest_index]) ** 2 + (y[brightest_index] - y[second_brightest_index]) ** 2)

    # Normalising the distance between all stars
    for i in range(len(x)):
        x[i] = x[i] / distance_brightest
        y[i] = y[i] / distance_brightest
    lines = lines / distance_brightest

    # Finding the angle of rotation to rotate second brightest star to (1, 0)
    p0 = (x[brightest_index], y[brightest_index])
    p1 = (x[second_brightest_index], y[second_brightest_index])
    p2 = (1, 0)
    theta = getAngle(p0, p1, p2)
    if round(x[second_brightest_index] * math.cos(theta) - y[second_brightest_index] * math.sin(theta), 2) != float(1) or round(x[second_brightest_index] * math.sin(theta) + y[second_brightest_index] * math.cos(theta), 2) != float(0):
        theta = -theta

    # Updating the new coordinates of each star
    for i in range(1, len(x)):
        x_new = x[i] * math.cos(theta) - y[i] * math.sin(theta)
        y_new = x[i] * math.sin(theta) + y[i] * math.cos(theta)
        x[i], y[i] = x_new, y_new

    for line in lines:
        for x1, y1, x2, y2 in line:
            pass

    return np.array(x), np.array(y), lines

def iterateArea(contours, lines=[], iterate=False):
    lines = np.array(lines)

    # Finding the coordinates of the contours and their area
    coordinates = {}
    for i in range(len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(cnt)
        if area in coordinates:
            coordinates[area].append((cx, cy))
        else:
            coordinates[area] = [(cx, cy)]

    sorted_area = []
    for i in reversed(sorted(coordinates.keys())):
        sorted_area.append(i)

    # Creating coordinates for each star in the constellation
    x = []
    y = []
    for area in sorted_area:
        x.append(coordinates[area][0][0])
        y.append(coordinates[area][0][1])

    threshold = min(150, sorted_area[2])
    count = 0
    for area in sorted_area:
        if area >= threshold:
            count += 1

    coordinates_list = []
    if not iterate:
        return getNormalisedCoordinates(x, y, 0, 1, lines)
    else:
        for i in range(count):
            coordinates_list.append(getNormalisedCoordinates(x, y, 0, i + 1, lines))
        return coordinates_list

# Finding edges using Canny edge detection
def findEdges(image, thresh1, thresh2):
    out = cv2.Canny(np.array(image), thresh1, thresh2)
    return out

def invertImage(image):
    return cv2.bitwise_not(image)

# Converting it back to black and white image
def getGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying median filter to get rid of noise left from words in the image
def applyMedian(image, size):
    return cv2.medianBlur(image, size)

# Plot the specified image with the name given
def plotImage(image, imagename="figure"):
    cv2.imshow(imagename, image)

def binariseImage(img, thresholds):
    # Thresholding the Image to binarise it
    output_thresh = []
    for threshold in thresholds:
        ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        output_thresh.append(thresh)
    return output_thresh

def getGreenChannel(img):
    image_copy_green = img.copy()
    # Setting Green and Red channel 0
    image_copy_green[:, :, 0] = 0
    image_copy_green[:, :, 2] = 0
    return image_copy_green

def getBlueChannel(img):
    image_copy_blue = img.copy()
    # Setting Green and Red channel 0
    image_copy_blue[:, :, 1] = 0
    image_copy_blue[:, :, 2] = 0
    return image_copy_blue

def getRedChannel(img):
    # Setting Green and Blue channel 0
    image_copy = img.copy()
    image_copy[:, :, 1] = 0
    image_copy[:, :, 0] = 0
    return image_copy

def makeTemplates():
    # Directory where the templates are stored
    template_directory = "./Templatess"
    templates_coordinates = {}

    # Iterate through each file in the template directory to process one template at a time
    for filename in os.listdir(template_directory):
        img = cv2.imread("./Templatess/" + filename)
        red_channel = getRedChannel(img)
        # Process the red channel image to find coordinates
        # Add the coordinates to templates_coordinates

    # Save the normalised coordinates for all templates
    with open("Template Coordinates", "wb") as fp:
        pickle.dump(templates_coordinates, fp)

def test_normaliser(test_path):
    # Process and find the normalised coordinate for each template present in the Templates directory
    # makeTemplates()

    img = cv2.imread(test_path)
    img = getGrayscale(img)
    cv2.imshow('test_img', img)

    a = 50
    thresh = binariseImage(img, [190])
    # Subtracting to get only stars
    final = thresh[0]
    plotImage(final, "final")
    stars = applyMedian(final, 5)
    final_stars_inverted = invertImage(stars)
    edged = findEdges(final_stars_inverted, 30, 200)
    plotImage(edged, "edges")

    edge_copy = edged.copy()
    contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    final_contours = []
    for contour in contours:
        final_contours.append(contour)

    coordinates_list = iterateArea(final_contours, [], True)
    return coordinates_list

def score(x, y, template_x, template_y):
    test_coordinates = {}
    template_coordinates = {}

    for i in range(len(x)):
        test_coordinates[dist((0, 0), (x[i], y[i]))] = (x[i], y[i])

    for i in range(len(template_x)):
        template_coordinates[dist((0, 0), (template_x[i], template_y[i]))] = (template_x[i], template_y[i])

    sorted_x = []
    sorted_y = []
    sorted_template_x = []
    sorted_template_y = []

    for i in sorted(test_coordinates.keys()):
        sorted_x.append(test_coordinates[i][0])
        sorted_y.append(test_coordinates[i][1])

    for i in sorted(template_coordinates.keys()):
        sorted_template_x.append(template_coordinates[i][0])
        sorted_template_y.append(template_coordinates[i][1])

    index_test = 0
    index_template = 0

    count = 0
    matched_coord = []

    while index_test < len(sorted_x):
        if dist((sorted_x[index_test], sorted_y[index_test]), (sorted_template_x[index_template], sorted_template_y[index_template])) < 0.05:
            count += 1
            matched_coord.append((sorted_x[index_test], sorted_y[index_test]))
            index_template += 1
        index_test += 1

    return count, matched_coord

def simillarity_error(train, test):
    threshold = 0.05 * 1
    error = 1e-100
    count = 0
    for i in range(train[0].shape[0]):
        if dist(train[:, i], test[:, i]) < threshold:
            count += 1
        else:
            error += dist(train[:, i], test[:, i])
    return count, error

def test_runner(constellation):
    test_coordinates = test_normaliser('test_data/' + constellation + '.png')
    true_label = constellation[:]

    file = open('Template Coordinates', 'rb')
    template_coordinate = pickle.load(file)

    score = -1
    pred_label = 'None'

    plot_points = []

    for bright_perm in range(len(test_coordinates)):
        for constellation in template_coordinate:
            x_template, y_template, n_stars, normalised_lines = template_coordinate[constellation]

            e = simillarity_error((x_template, y_template), test_coordinates[bright_perm])
            cur_score = e[0] * (e[0] - 2) / (n_stars * e[1])

            if e[0] > 2 and score < cur_score < 1e+3:
                pred_label = constellation
                score = cur_score
                plot_points = (x_template, y_template, test_coordinates, normalised_lines)

    plt.figure('Matched ' + true_label + " " + pred_label)
    plt.scatter(plot_points[0], plot_points[1])
    plt.scatter(plot_points[2][bright_perm][0], plot_points[2][bright_perm][1])
    for line in plot_points[3]:
        for x1, y1, x2, y2 in line:
            plt.plot([x1, x2], [y1, y2], color='red')
    plt.savefig("./Predicted_images/" + true_label + " " + pred_label)
    plt.close()

    return pred_label

if __name__ == "__main__":
    d = ['Andromeda', 'Aquila', 'Auriga', 'CanisMajor', 'Capricornus', 'Cetus', 'Columba', 'Gemini', 'Grus', 'Leo', 'Orion', 'Pavo', 'Pegasus', 'Phoenix', 'Pisces', 'PiscisAustrinus', 'Puppis', 'UrsaMajor', 'UrsaMinor', 'Vela']
    count = 0
    for i in d:
        pred = test_runner(i)
        if pred == i:
            count += 1
            print(f"Constellation matched: {pred}")
        else:
            print(f"Expected: {i}, Predicted: {pred}")
    print(f"Accuracy: {count / len(d)}")
    cv2.waitKey(0)