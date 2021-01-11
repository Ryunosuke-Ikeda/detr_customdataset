# DETR Customdataset

detrをカスタムデータで学習
本リポジトリはBDD100Kをcocoに変換した想定です。
本家コードをちょいといじっただけなのでデータセット名等はcocoのままのなってますが気にしないでください

環境構築等は本家をご覧ください。
本家のdetrは[こちら](https://github.com/facebookresearch/detr)


## ほかのデータで学習する際の注意
- models/detr.pyの307行目、num_classesを使用するデータに合わせて下さい。

## 準備
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
ダウンロードしたらweightフォルダーを作って保存してください。
```
path/weight/detr-r50-e632da11.pth
```

## 学習
以下のコマンドを実行(本家の実行コマンド使うと1つのGPUで学習する際エラー吐きます。)
```
python main.py --batch_size 1 --coco_path ./to/small_bdd --output_dir ./output_small_bdd --resume weight/detr-r50-e632da11.pth
```

## 評価
以下のコマンドを実行
```
python main.py --batch_size 1 --no_aux_loss --eval --resume output_small_bdd/checkpoint.pth --coco_path ./to/small_bdd --output_dir ./output_small_bdd
```

## テスト
```
python main.py --inf --coco_path ./to/small_bdd --resume output_small_bdd/checkpoint.pth
```
注意！
テストしたい場合テストしたい画像のパスをinf_test.pyの74行目のパスを書き換えて下さい。
ここで入力するcoco_pathは本来関係ないですがこれないと動かないため必要



