from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
np.seterr(divide='ignore', invalid='ignore')

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])



def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    equal_distant_numbers = np.linspace(0, 1, 100)

    ediff1ds = np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2)
    cumsum_d = np.cumsum(ediff1ds)
    sum_d = cumsum_d[-1]
    distance_normalization = cumsum_d / sum_d

    interp1d_X = interp1d(distance_normalization, points_X, kind='linear')
    interp1d_Y = interp1d(distance_normalization, points_Y, kind='linear')

    sample_points_X, sample_points_Y = interp1d_X(equal_distant_numbers), interp1d_Y(equal_distant_numbers)
    return sample_points_X, sample_points_Y


template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)



def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    threshold = 20

    gesture_points_X = gesture_points_X[0]
    gesture_points_Y = gesture_points_Y[0]
    gesture_start = np.array([gesture_points_X[0], gesture_points_Y[0]])
    gesture_end = np.array([gesture_points_X[-1], gesture_points_Y[-1]])

    num_templates = len(template_sample_points_X)
    template_start_points = np.array(
        [[template_sample_points_X[i][0], template_sample_points_Y[i][0]] for i in range(num_templates)])
    template_end_points = np.array(
        [[template_sample_points_X[i][-1], template_sample_points_Y[i][-1]] for i in range(num_templates)])

    gesture_start_point = np.reshape(gesture_start, (1, -1))
    gesture_end_point = np.reshape(gesture_end, (1, -1))

    start_distances = np.sqrt(np.sum((gesture_start_point-template_start_points)**2,axis=1))
    end_distances = np.sqrt(np.sum((gesture_end_point - template_end_points) ** 2, axis=1))

    valid_indices= []
    for i in range(len(start_distances)):
        if start_distances[i] <= threshold and end_distances[i] <= threshold:
            valid_indices.append(i)

    valid_template_sample_points_X = np.array(template_sample_points_X)[valid_indices]
    valid_template_sample_points_Y = np.array(template_sample_points_Y)[valid_indices]
    valid_words = [words[valid_index] for valid_index in valid_indices]

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return np.array(scaled_X), np.array(scaled_Y)

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''

    L = 200#Enter Value Here

    valid_normalized_template_sample_points_X = np.zeros(valid_template_sample_points_X.shape)
    valid_normalized_template_sample_points_Y = np.zeros(valid_template_sample_points_Y.shape)
    normalized_gesture_sample_points_X = np.zeros(gesture_sample_points_X.shape)
    normalized_gesture_sample_points_Y = np.zeros(gesture_sample_points_Y.shape)

    for i in range(len(valid_template_sample_points_X)):
        valid_normalized_template_sample_points_X[i],valid_normalized_template_sample_points_Y[i] = get_scaled_points(valid_template_sample_points_X[i],valid_template_sample_points_Y[i], L)

    normalized_gesture_sample_points_X[0],normalized_gesture_sample_points_Y[0] = get_scaled_points(gesture_sample_points_X[0],gesture_sample_points_Y[0], L)
    x_diff = (valid_normalized_template_sample_points_X - np.reshape(normalized_gesture_sample_points_X, (1, -1))) ** 2

    y_diff = (valid_normalized_template_sample_points_Y - np.reshape(normalized_gesture_sample_points_Y, (1, -1))) ** 2

    distances = (x_diff + y_diff) ** 0.5
    shape_scores = np.sum(distances, axis=1) / 100

    return shape_scores


def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15

    alphas = np.zeros((100))
    mid_point = 100 // 2
    for i in range(mid_point):
        x = i / 2450
        alphas[mid_point - i - 1], alphas[mid_point + i] = x, x

    location_scores = np.zeros(valid_template_sample_points_X.shape[0])
    gesture_points = [[gesture_sample_points_X[0][j], gesture_sample_points_Y[0][j]] for j in range(100)]

    for i in range(valid_template_sample_points_X.shape[0]):
        template_points = [[valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]] for j in range(100)]
        distances = euclidean_distances(gesture_points, template_points)
        min_d_between_template_gesture = np.min(distances, axis=0)
        min_d_between_gesture_template = np.min(distances, axis=1)
        if np.any(min_d_between_gesture_template > radius) or np.any(min_d_between_template_gesture > radius):
            deltas = np.diagonal(distances)
            location_scores[i] = np.sum(np.multiply(alphas, deltas))

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []

    shape_coef = 0.2#Enter Value Here#

    location_coef = 0.8#Enter Value Here#
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'

    n = 3

    best_words = []
    min_score = np.min(np.array(integration_scores))
    min_score_indices = np.where(integration_scores == min_score)[0]
    # Create a list of words having minimum scores
    for min_score_index in min_score_indices:
        best_words.append(valid_words[min_score_index])
        if(len(best_words)>=n):
            break
    return ' '.join(best_words)


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)#Generate Sample Points

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y)#Do Pruning

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)#Get Shape Scores

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)#Get Location Scores

    integration_scores = get_integration_scores(shape_scores, location_scores)#Get Integration Scores

    best_word = get_best_word(valid_words, integration_scores)#Get Best Word

    end_time = time.time()
    
    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()
