import os
import imutils
import numpy as np
import cv2
from math import ceil, floor
from model import CNN_Model
from collections import defaultdict
import pandas as pd
import random


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):
    # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # apply canny edge detection algorithm
    img_canny = cv2.Canny(blurred, 100, 200)
    # find contours
    cnts = cv2.findContours(
        img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr +
                                                    h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks


def process_ans_blocks(ans_blocks):
    """
        this function process 2 block answer box and return a list answer has len of 200 bubble choices
        :param ans_blocks: a list which include 2 element, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    biggest_block = max(
        ans_blocks, key=lambda block: block[1][2] * block[1][3])
    ans_block_img = np.array(biggest_block[0])
    # image_path = os.path.join(
    #     "./output", f"answer{random.randint(1,50)}.png")
    # cv2.imwrite(image_path, ans_block_img)
    ans_block_img = ans_block_img[2:ans_block_img.shape[0]-2, :]
    offset2 = floor(ans_block_img.shape[0] / 25)
    width = ceil(ans_block_img.shape[1])
    print(width)
    # Loop over each line in the answer block
    for j in range(25):
        answer_img = ans_block_img[j * offset2:(j + 1) * offset2, :]
        list_answers.append(answer_img)

    return list_answers, width


def process_list_ans(filename, list_answers, width):
    list_choices = []
    offset = ceil(width / 6)
    start = offset
    index = 0
    for answer_img in list_answers:
        index = index+1
        # image_path = os.path.join("./output", f"answer{index}.png")
        # cv2.imwrite(image_path, answer_img)
        for i in range(5):
            bubble_choice_raw = answer_img[:, start +
                                           i * offset:start + (i + 1) * offset]
            # image_path = os.path.join(
            #     "./output", f"answer{filename}{index}{i}.png")
            # cv2.imwrite(image_path, bubble_choice_raw)
            box_offset = floor(width/12 - 12/592*width)
            bubble_choice = bubble_choice_raw[:, box_offset:-box_offset]

            bubble_choice = cv2.threshold(
                bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            image_path = os.path.join(
                "./output", f"answer{filename}{index}{i}.png")
            cv2.imwrite(image_path, bubble_choice)
            list_choices.append(bubble_choice)

    # if len(list_choices)  480:
    #     raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 5 == 0:
        answer_circle = "A"
    elif idx % 5 == 1:
        answer_circle = "B"
    elif idx % 5 == 2:
        answer_circle = "C"
    elif idx % 5 == 3:
        answer_circle = "D"
    else:
        answer_circle = "E"
    return answer_circle


def remove_empty_strings_from_dict(dictionary):
    for question, answers in dictionary.items():
        non_empty_answers = [answer for answer in answers if answer != '']
        dictionary[question] = non_empty_answers or ['']
    return dictionary


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 5

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)
        else:
            results[question + 1].append('')
    return remove_empty_strings_from_dict(results)


def read_correct_answers(file_path):
    df = pd.read_excel(file_path, header=None)
    df = df.drop(columns=0)  # Exclude the first column (question number)

    # Convert each row to a set of trimmed correct answers and create the dictionary
    correct_answers_dict = {
        index+1: set(answer.strip() for answer in row.dropna().tolist()) for index, row in df.iterrows()}
    return correct_answers_dict


if __name__ == '__main__':
    # model = CNN_Model()
    # model.train()
    folder_path = './images'
    images = []
    data = []
    correct_answers = read_correct_answers('answers.xlsx')

    total_questions = len(correct_answers)

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            list_ans_boxes = crop_image(img)
            list_ans, width = process_ans_blocks(list_ans_boxes)
            list_ans = process_list_ans(filename, list_ans, width)
            answers = get_answers(list_ans)
            correct_count = 0
            for question, correct_option in correct_answers.items():
                if question in answers:
                    student_options = answers[question]
                    correct_options_count = len(correct_option)
                    correct_student_options = [
                        option for option in student_options if option in correct_option]
                    student_options_count = len(correct_student_options)
                    question_score = student_options_count / correct_options_count
                    correct_count += question_score

            score = correct_count / total_questions * 10
            print(answers)
            data.append([filename] + [', '.join(map(str, ans_list))
                        for ans_list in answers.values()]+[correct_count]+[score])

    # Convert data list to a DataFrame
    columns = ['ID'] + list(answers.keys())+['Corrects']+["Score"]
    df = pd.DataFrame(data, columns=columns)
    df.to_excel('results.xlsx', index=False)
