import mailbox
import pandas as pd
import plac
import pdb
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def main(mbox_path: ('MBox Path', 'option', 'm')):
    messages = []
    texts = []
    for idx, message in enumerate(mailbox.mbox(mbox_path)):
        content = message.get_payload()[0].get_payload()
        stripped_content = striphtml(content).replace('\r', '').replace('\n', '').replace('=2C', '').replace('=', '')

        matches = re.findall(r'"([^"]*)"', stripped_content)
        if len(matches) == 0:
            print("{}: Failed to extract message.".format(idx))
            continue

        messages.append(message)
        texts.append({'text': matches[0]})
    df = pd.DataFrame(texts)
    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(df['text'].values).toarray()
    indexes_to_keep = np.flip(vectorized.argsort(axis=-1), -1)[:, :5]
    arr = np.zeros(vectorized.shape)
    for idx, set_one_idxs in enumerate(indexes_to_keep):
        arr[idx][set_one_idxs] = 1.

    terms_per_document = vectorizer.inverse_transform(arr)
    all_terms = []

    for terms in terms_per_document:
        all_terms += terms.tolist()

    count = Counter(all_terms)
    print(count.most_common(20))

if __name__ == '__main__':
    plac.call(main)

