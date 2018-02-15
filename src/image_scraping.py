"""
Before running this script, create 'Training_Data' and 'Validation_Data' folders. Once created just run the script and you're golden.
"""

import os
import requests
import pathlib
import numpy as np
import pandas as pd
import shutil



def get_images_from_url(url,label, image_number):
    """
    Helper function to save .jpg images into the specified sub directory(labeling purposes)

    Input....
    url: URL where the image is taken from
    label: The specified category taken from category name in pandas data frame
    image_number: Unique number for image

    Output.....
    The saved .jpg images from the image urls
    """
    image_url = str(url)
    img_data = requests.get(image_url).content
    with open(os.path.join('../data/Training_Data/' + label, label + image_number + '.jpg'), 'wb') as handler:
        handler.write(img_data)

def make_training_directory_label(df):
    """
    This helper function will create sublabels for the images to into for training purposes

    Input.....
    df = Pandas DataFrame consisting of all category labels

    Output....
    Sub directories within the training data folder, sub directory labels are the category labels
    """
    for element in df[0]:
        pathlib.Path('../data/Training_Data/' + element).mkdir(parents=True, exist_ok=True)

def make_validation_directory_label(df):
    """
    This helper function will create sublabels for the images to into for training purposes

    Input.....
    df = Pandas DataFrame consisting of all category labels

    Output....
    Sub directories within the validation data folder, sub directory labels are the category labels
    """
    for element in df[0]:
        pathlib.Path('../data/Validation_Data/' + element).mkdir(parents=True, exist_ok=True)

def populate_training_directory_label(df):
    """
    Using the DataFrame containing training data, separate and get images into their respective Category labels

    Input.....
    df = Pandas DataFrame containing training data

    Output.....
    Based on your sub directories, all images are organized in their respective Category label
    """
    for num,element in enumerate(df.CategoryName):
        print('Currently populating {} sub directory'.format(element))
        url = df['ImageUrl'][num]
        get_images_from_url(url, element, str(num))

def populate_validation_directory_label(df):
    """
    Helper function to populate the validation set using about 20 percent of training data

    Input.....
    df = DataFrame that contains all the category labels
    """
    for element in df[0]:
        print('Currently moving {} images to Validation'.format(element))
        num = 0
        source = '../data/Training_Data/' + element + '/'
        dest1 = '../data/Validation_Data/' + element + '/'
        files = os.listdir(source)
        for f in files:
            if num == 20:
                break
            if not f.startswith('.'):
                shutil.move(source + f, dest1)
                num += 1


if __name__ == "__main__":
    df = pd.read_excel('../data/training_data.xlsx')
    categories = pd.read_excel('../data/categories.xlsx', header=None)
    make_training_directory_label(categories)
    make_validation_directory_label(categories)
    populate_training_directory_label(df)
    populate_validation_directory_label(categories)
