import pandas as pd

df = pd.read_csv('data_scaled.fix.csv')

titles = df['title'][:]

for title_num, r in enumerate(titles):
    parts = r.split(' ')
    # track first occurences of words in the titles
    word2indices = {}

    # assemble the index representation in place
    doc2vec = []

    for i, word in enumerate(parts):
        if word not in word2indices:
            word2indices[word] = i
        doc2vec.append(word2indices[word])

    # write it back (is this slower?)
    titles[title_num] = doc2vec

# I'm not sure about copy-by-reference, so we'll check that here
print('titles:', titles[:5])

print('df[\'title\']', df['title'].head())

print(titles[11868]) # our longest entry is 15 cells. We can pad that in tflearn... :)
