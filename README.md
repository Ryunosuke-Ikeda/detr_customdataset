## DETR Customdataset

detrをカスタムデータで学習
本リポジトリはBDD100Kをcocoに変換した想定です。
本家コードをちょいといじっただけなのでデータセット名等はcocoのままのなってますが気にしないでください

本家のdetrは[こちら](https://github.com/facebookresearch/detr)

# ほかのデータで学習する際の注意
- models/detr.pyの307行目、num_classesを使用するデータに合わせて下さい。

#　準備
- 学習データをダウンロード
フォルダ内に学習データをダウンロードしてください。
今回は以下のようなディレクトリ構造を想定しています
```
path/to/small_bdd/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
-　学習済みモデルをダウンロード
今回は[detr本家の学習済みモデル](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)を使用します。
(detrは学習スピードがめちゃくちゃ遅いので1つのGPU等で1から学習するときついみたいです。)
ダウンロードしたらweigthフォルダーを作って保存してください。
```
path/weigth/detr-r50-e632da11.pth
```





