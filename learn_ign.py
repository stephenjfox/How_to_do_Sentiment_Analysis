import pandas as pd
from collections import Counter


df = pd.read_csv('data_scaled.fix.csv')

titles = df['title']

for r in titles:
    print(r)
    c = Counter(r.split(' '))
    print(c)
    break
