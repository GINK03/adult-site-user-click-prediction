# アダルトサイトの人気予想

具体的なアダルトサイトとは、`https://movie.eroterest.net/`。  

各動画のクリック数がカウントされているため、予想するタスクとしてやりやすい。  


## スクレイピングとデータ・セット

スクレイピング済みのファイル
 - https://www.dropbox.com/s/yzmvw2a9yp8kjhn/parsed.tar.gz?dl=0
 - https://www.dropbox.com/s/86h3h8zgpzrvjpz/imgs.tar.gz?dl=0
 

新規にスクレイピングする場合
 - https://github.com/GINK03/scraping-designs/tree/master/eroterest

## テキスト情報でからがクリック数を予想する

Bag of Wordsに変換し、予想するアプローチ(1)

**単語にインデックスをふり、疎行列の変換の準備ファイルを作成**  

```
$ python3 preparation.py 
```

**疎行列を作成する**  
```
$ python3 bow_embed.py --mode=make_sparse
```

**DeepLearningに学習できるように密行列に変換**  
```
$ python3 bow_embed.py --mode=make_dense
```

## DeepLearningで学習

**5fold-cvで学習する**  
```
$ python3 keras_dense.py
```
<div align="center">
 <img widht="450px" src="https://user-images.githubusercontent.com/4949982/44300069-e67ab700-a33b-11e8-8d82-0777be0af924.png">
</div>

