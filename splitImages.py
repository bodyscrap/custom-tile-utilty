''' splitImages.py
画画を同サイズのタイルに分割する。  
設定上オーバーラップはあり、削除するマージンも設定可能。  
ディープラーニングモデル等での推論結果のマージなどが主な用途。  
jsonファイルで読み込んだ設定に従って、画像を分割する。

分割用設定ファイルの項目は以下

tile_width : 基本となるタイルの幅(pixel)
tile_height : 基本となるタイルの高さ(pixel)
margin : タイルの外側の余白(pixel)
また、タイルのstrideは、tile_width/tile_heightの半分に設定される。  
(tile_width, tile_whitghtは偶数pixelとする。)

正のmarginを設定した場合、実際のタイルサイズは、
上下左右にmarginで指定した分拡張される。  
負のmarginを設定した場合は、
tile_width/tile_heightはそれぞれ、2 * margin分縮小され、
marginの符号が反転した設定と同じとして扱う。  

分割時はmarginも含んだサイズで切り出される。  
これは機械学習モデル等では一般的に領域外周の情報の信頼性が低い傾向が高く、
あえて、広めに入力してから信頼性の高い中央部のみを使用するような使い方を想定している。
'''
import cv2
import json
import os
import numpy as np
import glob

def normalizeSetting(setting):
    '''設定の正規化
    正のmarginはそのまま、負のmarginはtile_width/tile_heightを縮小し、marginの符号を反転させる。
    stride_h/stride_vはtile_width/tile_heightの半分に設定する。
    '''
    if setting["margin"] < 0:
        setting["tile_width"] += 2 * setting["margin"]
        setting["tile_height"] += 2 * setting["margin"]
        setting["margin"] = -setting["margin"]
    setting["stride_h"] = setting["tile_width"] // 2
    setting["stride_v"] = setting["tile_height"] // 2
    setting["tile_width"] = setting["stride_h"] * 2
    setting["tile_height"] = setting["stride_v"] * 2
    return setting

def splitImage(img, setting, use_only_inner=False):
    '''設定に従って画像を分割する
    args:
        img : 分割対象の画像
        setting : 分割設定
        use_only_inner : 
            Trueの場合、画像の内側のみからタイルを切り出し、端数は切り捨てる。
            Falseの場合、元画像が切り捨てられないように、上下左右に均等にマージンを追加する。  
    return:
        tiles : 分割設定、分割されたタイルのリスト
    '''
    src_h, src_w = img.shape[:2]
    # 切り出し元の画像サイズの決定
    if use_only_inner:
        # 画像の内側のみを使用し端数を捨てる場合
        test_w = src_w - setting["margin"] * 2
        test_h = src_h - setting["margin"] * 2
        rows = (test_h - setting["tile_height"]) // setting["stride_v"] + 1
        cols = (test_w - setting["tile_width"]) // setting["stride_h"] + 1
        # 切り出し用バッファサイズの算出
        buffer_w = setting["tile_width"] + (cols - 1) * setting["stride_h"] + setting["margin"] * 2
        buffer_h = setting["tile_height"] + (rows - 1) * setting["stride_v"] + setting["margin"] * 2
        # 値が0のバッファを確保
        src_buf = np.zeros((buffer_h, buffer_w, 3), dtype=np.uint8)
        # バッファの中心に元画像を配置
        offset_x = 0
        offset_y = 0
        src_buf = img[offset_y:offset_y+buffer_h, offset_x:offset_x+buffer_w]
    else:
        # 前元画像を含めるようにマージンを付加する場合
        test_w = src_w
        test_h = src_h
        rows = (test_h - setting["tile_height"]) // setting["stride_v"] + 1
        cols = (test_w - setting["tile_width"]) // setting["stride_h"] + 1
        # 端数がある場合は、タイルを追加する
        rows += 1 if (test_h - setting["tile_height"]) % setting["stride_v"] > 0 else 0
        cols += 1 if (test_w - setting["tile_width"]) % setting["stride_h"] > 0 else 0
        # 切り出し用バッファサイズの算出
        buffer_w = setting["tile_width"] + (cols - 1) * setting["stride_h"] + setting["margin"] * 2
        buffer_h = setting["tile_height"] + (rows - 1) * setting["stride_v"] + setting["margin"] * 2
        # 値が0のバッファを確保
        src_buf = np.zeros((buffer_h, buffer_w, 3), dtype=np.uint8)
        # バッファの中心に元画像を配置
        offset_x = (buffer_w - src_w) // 2
        offset_y = (buffer_h - src_h) // 2
        src_buf[offset_y:offset_y+src_h, offset_x:offset_x+src_w] = img
    # タイル分割情報の記録
    tile_setting = setting.copy()
    tile_setting["rows"] = rows
    tile_setting["cols"] = cols
    tile_setting["buffer_w"] = buffer_w
    tile_setting["buffer_h"] = buffer_h
    tile_setting["offset_x"] = offset_x
    tile_setting["offset_y"] = offset_y
    tile_setting["src_w"] = src_w
    tile_setting["src_h"] = src_h
    # バッファをタイル分割
    tiles = []
    crop_width =  setting["tile_width"] + setting["margin"] * 2
    crop_height = setting["tile_height"] + setting["margin"] * 2
    for r in range(rows):
        for c in range(cols):
            x = c * setting["stride_h"]
            y = r * setting["stride_v"]
            tile = src_buf[y:y+crop_height, x:x+crop_width]
            tiles.append(tile)
    return tiles, src_buf, tile_setting

def saveTileImages(tiles, tile_setting, dst_dir):
    '''タイル画像を保存する
    args:
        tiles : 分割されたタイルのリスト
        tile_settin : 分割設定
        dst_dir : 保存先ディレクトリ
    '''
    os.makedirs(dst_dir, exist_ok=True)
    # 桁数を計算
    num = len(tiles)
    digit = len(str(num))
    for idx, img in enumerate(tiles):
        filename = f"tile_{idx:0{digit}d}.png"
        path = os.path.join(dst_dir, filename)
        cv2.imwrite(path, img)
    # 分割情報の保存
    json.dump(tile_setting, open(os.path.join(dst_dir, "tile_setting.json"), "w"), indent=4)

def createTiledDataset(src_dir, dst_dir, setting):
    '''指定ディレクトリ内の画像をタイル分割し、保存する
    args:
        src_dir : 分割対象の画像が格納されているディレクトリ
        dst_dir : 保存先ディレクトリ
        setting : 分割設定
    '''
    os.makedirs(dst_dir, exist_ok=True)
    # 画像ファイルの取得
    img_files = glob.glob(os.path.join(src_dir, "*.jpg"))
    img_files += glob.glob(os.path.join(src_dir, "*.png"))
    for path_img in img_files:
        # 画像の読み込み
        img = cv2.imread(path_img)
        # 画像の分割
        tiles, _, tile_setting = splitImage(img, setting, use_only_inner=False)
        # タイル画像の保存
        base_name = os.path.splitext(os.path.basename(path_img))[0]
        dst_subdir = os.path.join(dst_dir, base_name)
        saveTileImages(tiles, tile_setting, dst_subdir)

if __name__ == "__main__":
    # 設定ファイルの読み込み
    setting = json.load(open("sample-input/split_setup.json", "r"))
    setting = normalizeSetting(setting)
    # 画像の読み込み
    path_img = "debug_data/beers.jpg"
    img = cv2.imread(path_img)
    # 出力ディレクトリ設定
    dst_dir = "result3"
    os.makedirs(dst_dir, exist_ok=True)
    # 画像の分割
    tiles, src_buf, tile_setting = splitImage(img, setting, use_only_inner=True)
    # タイル画像の保存
    saveTileImages(tiles, tile_setting, dst_dir)
