'''タイル画像の統合
splitImages.pyで分割されたタイル画像を元の画像サイズに統合する。
統合時は、タイル画像の外側の余白部分は削除され、中央部のみを使用して統合する。
'''

import cv2
import json
import os
import numpy as np
import glob

def createMergeCoeffMap(tile_setting):
    '''タイルのマージ用の加重加算マップの作成
    '''
    buffer_h = tile_setting["tile_height"] + tile_setting["stride_v"] * (tile_setting["rows"] - 1)
    buffer_w = tile_setting["tile_width"] + tile_setting["stride_h"] * (tile_setting["cols"] - 1)
    buffer_map = np.zeros((buffer_h, buffer_w, 1), dtype=np.float32)
    rows = tile_setting["rows"]
    cols = tile_setting["cols"]
    for r in range(rows):
        y = r * tile_setting["stride_v"]
        for c in range(cols):
            x = c * tile_setting["stride_h"]
            buffer_map[y:y + tile_setting["tile_height"], x:x + tile_setting["tile_width"]] += 1.0
    return buffer_map

def mergeTiles(tile_dir):
    '''タイル画像の統合
    args:
        tile_dir : タイル画像が格納されているディレクトリ
    '''
    # 設定ファイルの読み込み
    setting = json.load(open(os.path.join(tile_dir, "tile_setting.json"), "r"))
    # 重みづけマップの作成
    merge_coeff_map = createMergeCoeffMap(setting)
    # 統合用の空画像を作成
    merged_img = np.zeros((merge_coeff_map.shape[0], merge_coeff_map.shape[1], 3), dtype=np.float32)
    # タイル画像の読み込み
    tile_files = glob.glob(os.path.join(tile_dir, "*.png")) 
    tile_files.sort()  # ファイル名でソート
    for i, tile_file in enumerate(tile_files):
        tile = cv2.imread(tile_file).astype(np.float32)
        # マージンの除去
        tile = tile[setting["margin"]:setting["margin"] + setting["tile_height"], setting["margin"]:setting["margin"] + setting["tile_width"]]
        # タイルの位置を計算して統合
        r = i // setting["cols"]
        c = i % setting["cols"]
        y = r * setting["stride_v"]
        x = c * setting["stride_h"]
        merged_img[y:y + setting["tile_height"], x:x + setting["tile_width"]] += tile
    # 重みで割って正規化
    merged_img /= merge_coeff_map
    merged_img = merged_img.astype(np.uint8)
    # 元画像領域をクロップ
    # マージンを除いた分を考慮して切り取り開始位置を計算
    crop_offset_x = setting["offset_x"] - setting["margin"]
    crop_offset_y = setting["offset_y"] - setting["margin"]
    merged_img = merged_img[crop_offset_y:crop_offset_y + setting["src_h"], 
                            crop_offset_x:crop_offset_x + setting["src_w"]]
    return merged_img

if __name__ == "__main__":
    # 統合するタイル画像のディレクトリ
    tile_dir = "result1"
    # 画像の統合
    merged_img = mergeTiles(tile_dir)
    # 統合画像の保存
    cv2.imwrite(os.path.basename(tile_dir) + "_merged.png", merged_img)