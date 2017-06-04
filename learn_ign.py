# import pandas as pd
# from collections import Counter

#
# df = pd.read_csv('data_scaled.fix.csv')
#
# df['title']
titles = [
    'the quick brown fox jumps over the lazy dog',
    'this is a test story'
]

for title_num, r in enumerate(titles):
    parts = r.split(' ')
    word2indices = {}
    doc2vec = []
    for i, word in enumerate(parts):
        if word not in word2indices:
            word2indices[word] = i
        doc2vec.append(word2indices[word])

    print(parts, '->', doc2vec)

    if title_num == 6:
        break
