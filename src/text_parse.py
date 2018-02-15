import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path

train_images = '../data/Training_Data'
validation_images = '../data/Validation_Data'


def get_existing_indices(df):
    train_indices, val_indices = [], []
    for path, array in [[train_images, train_indices], [validation_images, val_indices]]:
        for (dirpath, dirnames, filenames) in os.walk(path):
            if len(filenames) > 0:
                for f in filenames:
                    asin = f.split('.jpg')[0]
                    idx = df.index[df['ASIN'] == asin]
                    array.extend(idx.values)

    return train_indices, val_indices

def place_vec_text_in_folder(df):
    for path in [train_images, validation_images]:
        for (dirpath, dirnames, filenames) in os.walk(path):
            if len(filenames) > 0:
                for f in filenames:
                    asin = f.split('.jpg')[0]
                    idx = df.index[df['ASIN'] == asin].values
                    vec_text = vectorized_text[idx]
                    if len(train_classes[idx]) > 0:
                        p = Path('{}/{}/{}.npy'.format(path, train_classes[idx][0], asin))
                        np.save(p, vec_text)

if __name__ == '__main__':
    df = pd.read_excel('../data/training_data.xlsx')
    train_indices, val_indices = get_existing_indices(df)

    titles = np.array(training_df['Title'].values)
    train_titles, val_titles = titles[train_indices], titles[val_indices]
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    vectorized_text = vectorizer.fit_transform(titles).toarray()
    bag_of_words_vocab = vectorizer.vocabulary_

    print('Length of vocab : ', len(bag_of_words_vocab))

    place_vec_text_in_folder(training_df)
    print('Placed Vectorized Text in folders')
