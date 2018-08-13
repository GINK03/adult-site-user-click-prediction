import tarfile
import json
import MeCab
from collections import Counter
import re
m = MeCab.Tagger('-Owakati')
tar = tarfile.open('./parsed.tar.gz', 'r:gz')

# tag, word embedding
tag_index = json.load(fp=open('tag_index.json'))
word_index = json.load(fp=open('word_index.json'))

for member in tar.getmembers():
  print(member)
  if not member.isfile():
    continue
  fp = tar.extractfile(member)
  datum = fp.read().decode()

  obj = json.loads(datum)

  text, hash, img_url, clk_num, target_url, tags = obj
  print(obj)

  words = m.parse(text).strip().split()
  words = Counter(words)

  i_freq = { word_index[word]:freq for word, freq in words.items() }
  t_hot  = { tag_index[tag]:1 for tag in tags }

  if re.search(r'\d{1,}', clk_num) is None:
    continue

  clk_num = int(re.search(r'\d{1,}', clk_num).group(0))
  print(clk_num)
  print(i_freq)
  print(t_hot)
