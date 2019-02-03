import pandas as pd
import numpy as np
import random


def load_news():
    df = pd.read_csv('news_dataset/fake-news/train.csv', encoding='utf8')
    train_data = df['text'].values.tolist()          # 'text' column contains articles
    train_labels = df['label'].values.tolist()       # 'label' column contains label

    # Randomly shuffle data and labels together
    zipped = list(zip(train_data, train_labels))
    random.shuffle(zipped)
    train_data, train_labels = zip(*zipped)
    del df      # clear the memory

    return np.asarray(train_data).tolist(), np.asarray(train_labels).tolist()


train_data, train_labels = load_news()