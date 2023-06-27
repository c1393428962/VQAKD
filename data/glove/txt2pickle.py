import pandas as pd
import pickle
import csv

import requests

df = pd.read_csv('/root/autodl-tmp/msvd-data/glove/glove.840B.300d.txt', sep =" ", quoting=3, header=None, index_col=0)
print("glove file loaded!")
glove = {key: val.values for key, val in df.T.items()}

with open('/root/autodl-tmp/feature-data/glove/glove.840.300d.pkl', 'wb') as fp:
    pickle.dump(glove, fp)

### weixin token
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                     json={
                         "token": "69fcd3bf894d",
                         "title": "run status",
                         "name": "preprocess_question",
                         "content": "run success!!!"
                     })
print(resp.content.decode())