import imutils
import numpy as np
import cv2
import pandas as pd
from math import ceil
from model import CNN_Model
from collections import defaultdict
import os

output_folder = "./output"


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
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h > 100000:
            ans_block = gray_img[y:y + h, x:x + w]
            ans_block = cv2.threshold(
                ans_block, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            ans_block = cv2.resize(ans_block, (28, 28), cv2.INTER_AREA)
            ans_block = ans_block.reshape((28, 28, 1))
            ans_blocks.append((ans_block, [x, y, w, h]))

    sorted_ans_blocks = sorted(ans_blocks, key=get_x)

    return sorted_ans_blocks


def process_ans_blocks(ans_blocks):
    """
        this function process 2 block answer box and return a list answer has len of 200 bubble choices
        :param ans_blocks: a list which include 2 element, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    ans_block_img = np.array(ans_blocks[0][0])
    ans_block_img = ans_block_img[2:ans_block_img.shape[0]-2, :]
    offset2 = ceil(ans_block_img.shape[0] / 25)

    # loop over each line in the answer block
    for j in range(25):
        answer_img = ans_block_img[j * offset2:(j + 1) * offset2, :]
        list_answers.append(answer_img)
        # Save the answer image as a separate file
        # image_path = os.path.join(output_folder, f"answer{j + 1}.png")
        # cv2.imwrite(image_path, answer_img)

    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    offset = 185
    start = 185

    for answer_img in list_answers:
        for i in range(5):
            bubble_choice = answer_img[:, start +
                                       i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(
                bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    elif idx % 4 == 3:
        answer_circle = "D"
    else:
        answer_circle = "E"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    return results


if __name__ == '__main__':
    folder_path = './images'
    images = []
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            list_ans_boxes = crop_image(img)
            list_ans = process_ans_blocks(list_ans_boxes)
            list_ans = process_list_ans(list_ans)
            answers = get_answers(list_ans)
            data.append([filename] + [', '.join(map(str, ans_list))
                        for ans_list in answers.values()])

    # Convert data list to a DataFrame
    columns = ['ID'] + list(answers.keys())
    df = pd.DataFrame(data, columns=columns)
    df.to_excel('answers.xlsx', index=False)
