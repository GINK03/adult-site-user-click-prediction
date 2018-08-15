
import tarfile
import json
import MeCab

m = MeCab.Tagger('-Owakati')
tar = tarfile.open('./parsed.tar.gz', 'r:gz')

# lable encoding
word_freq, tag_freq = {}, {}
for member in tar.getmembers():
  print(member)
  if not member.isfile():
    continue
  fp = tar.extractfile(member)
  datum = fp.read().decode()

  obj = json.loads(datum)

  text, hash, img_url, clk_num, target_url, tags = obj
  #print(obj)
  print(m.parse(text).strip())

  for tag in tags:
    if tag_freq.get(tag) is None:
      tag_freq[tag] = 0
    tag_freq[tag] += 1

  for word in m.parse(text).strip().split():
    if word_freq.get(word) is None:
      word_freq[word] = 0
    word_freq[word] += 1


tag_index = { tag:index for index, (tag, freq) in enumerate( list(reversed(sorted(tag_freq.items(), key=lambda x:x[1])))[:3000] )}
word_index = { word:index for index, (word, freq) in enumerate( list(reversed(sorted(word_freq.items(), key=lambda x:x[1])))[:3000] )}
json.dump(tag_index, fp=open('tag_index.json', 'w'), indent=2, ensure_ascii=False)
json.dump(word_index, fp=open('word_index.json', 'w'), indent=2, ensure_ascii=False)
