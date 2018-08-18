import tarfile
import json
import MeCab
from collections import Counter
import re
import click
import pickle
def make_sparse():
  m = MeCab.Tagger('-Owakati')
  tar = tarfile.open('./parsed.tar.gz', 'r:gz')
  # tag, word embedding
  tag_index = json.load(fp=open('tag_index.json'))
  word_index = json.load(fp=open('word_index.json'))
  data = []
  for member in tar.getmembers():
    #print(member)
    if not member.isfile():
      continue
    fp = tar.extractfile(member)
    datum = fp.read().decode()

    obj = json.loads(datum)

    text, hash, img_url, clk_num, target_url, tags = obj
    words = m.parse(text).strip().split()
    words = Counter(words)
    i_freq = { word_index[word]:freq for word, freq in words.items() if word_index.get(word) }
    t_hot  = { tag_index[tag]:1 for tag in tags if tag_index.get(tag) }
    if re.search(r'\d{1,}', clk_num) is None:
      continue
    print(clk_num)
    clk_num = int( re.search(r'\d{1,}', clk_num.replace(',','') ).group(0))
    print(clk_num)
    data.append( (clk_num, i_freq, t_hot) )
  pickle.dump(data, open('data.pkl', 'wb'))

import numpy as np
import pandas as pd
def make_dense():
  data = pickle.load(open('data.pkl', 'rb'))

  clks = [x[0] for x in data] 
  clks = np.nan_to_num( np.log( np.array(clks) + 2 ) )
  clks = (clks - clks.min()) / clks.max()
  height = len(clks)
  w_freqs = np.zeros( (height, 3000) ).astype(np.float16)
  t_hots = np.zeros( (height, 3000) ).astype(np.float16)
  for h, x in enumerate(data):
    if h%100 == 0:
      print('now', h)
    ii = x[1]
    tt = x[2]
    for index, freq in ii.items():
      w_freqs[h, index] = 1.0 # freq
    for index, freq in tt.items():
      t_hots[h, index]  = 1.0
  
  np.savez('dense.npz', clks=clks, t_hots=t_hots, w_freqs=w_freqs)
  
  #pd.DataFrame(w_freqs).to_csv('test.csv')
@click.command()
@click.option('--mode', default='make_sparse', help='mode spec')
def main(mode):
  if mode == 'make_sparse':
    make_sparse()
  if mode == 'make_dense':
    make_dense()
if __name__ == '__main__':
  main()
