import pandas as pd


train_data = pd.read_csv('data/train_data.csv', index_col='id')
train_labels = pd.read_csv('data/train_label.csv', index_col='id')
test_data = pd.read_csv('data/test_data.csv', index_col='id')
test_labels = pd.read_csv('data/test_label.csv', index_col='id')
