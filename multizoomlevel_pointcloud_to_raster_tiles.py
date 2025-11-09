import os
import math
import numpy as np
import laspy
from PIL import Image

# -------------------------------
# Webメルカトル座標変換
# -------------------------------
def latlon_to_tile(lat, lon, z):
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2.0 * n
    )
    return xtile, ytile

def latlon_to_pixel(lat, lon, z, tile_size=256):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * tile_size
    y = (
        (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
        * tile_size
    )
    return x, y

# -------------------------------
# 標高 → RGB 変換（地理院方式）
# -------------------------------
def encode_height_to_rgb(h_array):
    rgb = np.zeros((h_array.shape[0], h_array.shape[1], 3), dtype=np.uint8)
    valid = h_array > 0
    h_int = (h_array[valid] * 100).astype(np.int32)
    r = (h_int >> 16) & 0xFF
    g = (h_int >> 8) & 0xFF
    b = h_int & 0xFF
    rgb[valid, 0] = r
    rgb[valid, 1] = g
    rgb[valid, 2] = b
    return rgb

# -------------------------------
# タイル保存ユーティリティ
# -------------------------------
def save_tile(h_array, output_dir, z, tx, ty):
    rgb = encode_height_to_rgb(h_array)
    out_dir = os.path.join(output_dir, str(z), str(tx))
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(os.path.join(out_dir, f"{ty}.png"))

# -------------------------------
# LAS → z_max タイル生成
# -------------------------------
def generate_zoom_tiles(las_path, output_dir, z_max=19, tile_size=256):
    las = laspy.read(las_path)
    xs, ys, zs = las.x, las.y, las.z

    tiles = {}

    for x, y, z in zip(xs, ys, zs):
        tx, ty = latlon_to_tile(y, x, z_max)
        if (tx, ty) not in tiles:
            tiles[(tx, ty)] = np.zeros((tile_size, tile_size), dtype=np.float32)
        px, py = latlon_to_pixel(y, x, z_max, tile_size)
        px = int(px - tx * tile_size)
        py = int(py - ty * tile_size)
        if 0 <= px < tile_size and 0 <= py < tile_size:
            tiles[(tx, ty)][py, px] = max(tiles[(tx, ty)][py, px], z)

    # 保存
    for (tx, ty), h_array in tiles.items():
        save_tile(h_array, output_dir, z_max, tx, ty)

    return tiles

# -------------------------------
# 粗いズーム生成 + 空タイル埋め
# -------------------------------
def generate_coarser_tiles_full_range(tiles_fine, output_dir, z_from, z_to, tile_size=256):
    # z=14のタイル範囲（基本領域）
    base_x_min, base_x_max = 14540, 14541
    base_y_min, base_y_max = 6455, 6456

    tiles = tiles_fine
    for z in range(z_from - 1, z_to - 1, -1):
        print(f"Generating zoom {z} tiles...")
        new_tiles = {}

        # 子→親結合
        for (tx_f, ty_f), h_f in tiles.items():
            tx = tx_f // 2
            ty = ty_f // 2
            if (tx, ty) not in new_tiles:
                new_tiles[(tx, ty)] = np.zeros((tile_size, tile_size), dtype=np.float32)

            h_small = h_f.reshape(tile_size // 2, 2, tile_size // 2, 2).max(axis=(1, 3))
            x_offset = (tx_f % 2) * (tile_size // 2)
            y_offset = (ty_f % 2) * (tile_size // 2)
            new_tiles[(tx, ty)][y_offset:y_offset + tile_size // 2,
                                x_offset:x_offset + tile_size // 2] = np.maximum(
                new_tiles[(tx, ty)][y_offset:y_offset + tile_size // 2,
                                    x_offset:x_offset + tile_size // 2],
                h_small
            )

        # 範囲内のすべてのタイルを生成
        zoom_factor = 2 ** (z - 14)
        x_min = base_x_min * zoom_factor
        x_max = (base_x_max + 1) * zoom_factor - 1
        y_min = base_y_min * zoom_factor
        y_max = (base_y_max + 1) * zoom_factor - 1

        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                if (tx, ty) not in new_tiles:
                    # 点群がない → 標高0で埋める
                    new_tiles[(tx, ty)] = np.zeros((tile_size, tile_size), dtype=np.float32)
                save_tile(new_tiles[(tx, ty)], output_dir, z, tx, ty)

        tiles = new_tiles

# -------------------------------
# メイン
# -------------------------------
if __name__ == "__main__":
    las_path = r"E:\DeckGL\deck.gl\examples\website\point-cloud\09LD4241_wgs84.las"
    output_dir = r"E:\DeckGL\deck.gl\examples\website\point-cloud\tiles"
    os.makedirs(output_dir, exist_ok=True)

    # 最細ズーム（z=19）
    tiles_z19 = generate_zoom_tiles(las_path, output_dir, z_max=19)

    # z=18〜14 まで生成（すべて範囲内タイルを埋める）
    generate_coarser_tiles_full_range(tiles_z19, output_dir, z_from=19, z_to=14)

    print("✅ ズームレベル14〜19の全タイルを生成しました（空領域は標高0）。")
