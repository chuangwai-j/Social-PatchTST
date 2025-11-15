#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Social-PatchTST 专用 CAT021 解析器（兼容 Python 3.7+）
输出 13 个必备字段（已添加 flight_level）
【内置去重逻辑 + 多进程并行加速】
"""
import struct
import os
import glob
import csv
import time
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count  # 导入多进程工具


# ------------------ 工具函数 ------------------
# (这部分函数与您的版本完全相同，无需修改)
def get_signed_val(data_bytes: bytes) -> int:
    if not data_bytes:
        return 0
    pad = b'\xff' if data_bytes[0] & 0x80 else b'\x00'
    return struct.unpack('>i', pad * (4 - len(data_bytes)) + data_bytes)[0]


def decode_icao_char(six_bits: int) -> str:
    if 1 <= six_bits <= 26:
        return chr(ord('A') + six_bits - 1)
    if 48 <= six_bits <= 57:
        return chr(ord('0') + six_bits - 48)
    return ' ' if six_bits == 32 else ''


def get_variable_field_length(data: bytes, start: int) -> int:
    idx = start
    while idx < len(data):
        if not (data[idx] & 1):
            idx += 1
            break
        idx += 1
    return idx - start


def get_i110_length(data: bytes, start: int) -> int:
    if start >= len(data):
        return 0
    byte1 = data[start]
    length = 1
    if not (byte1 & 1):
        return length
    length += 1
    if (byte1 >> 6) & 1 and start + length < len(data):
        rep = data[start + length]
        length += 1 + rep * 15
    return length


def get_i220_length(data: bytes, start: int) -> int:
    if start >= len(data):
        return 0
    b = data[start]
    length = 1
    if b & 0x80:
        length += 2
    if b & 0x40:
        length += 2
    if b & 0x20:
        length += 2
    if b & 0x10:
        length += 1
    return length


def get_i250_length(data: bytes, start: int) -> int:
    if start >= len(data):
        return 0
    rep = data[start]
    return 1 + rep * 8


# ------------------ 核心解析 ------------------
# (此函数与您的版本完全相同)
def parse_cat021_to_dict(hex_str: str) -> Optional[Dict[str, float]]:
    try:
        data = bytes.fromhex(hex_str)
        if data[0] != 0x15:
            return None
        length = int.from_bytes(data[1:3], 'big')
        if len(data) != length:
            return None

        # FSPEC
        fspec_bits: List[int] = []
        idx = 3
        while True:
            b = data[idx]
            fspec_bits.extend([(b >> i) & 1 for i in range(7, 0, -1)])
            idx += 1
            if not (b & 1):
                break

        uap_len = {
            1: 2, 2: 0, 3: 2, 4: 1, 5: 3, 6: 6, 7: 8, 8: 3, 9: 2, 10: 2,
            11: 3, 12: 3, 13: 4, 14: 3, 15: 4, 16: 2, 17: 0, 18: 1, 19: 2,
            20: 2, 21: 2, 22: 2, 23: 1, 24: 2, 25: 2, 26: 4, 27: 2, 28: 3,
            29: 6, 30: 1, 31: -1, 32: 2, 33: 2, 34: -1, 35: 1, 36: 1, 37: 0,
            38: 1, 39: -1, 40: 7, 41: 1, 42: 0, 48: 0, 49: 0
        }

        parsed: Dict[str, float] = {}
        start = idx
        for frn, present in enumerate(fspec_bits, 1):
            if not present:
                continue
            item_len = 0

            if frn == 6:  # I130 Position
                lat = get_signed_val(data[start:start + 3]) * (180.0 / (1 << 23))
                lon = get_signed_val(data[start + 3:start + 6]) * (180.0 / (1 << 23))
                parsed['latitude'] = lat
                parsed['longitude'] = lon
                item_len = 6

            elif frn == 11:  # I080 Target Address
                parsed['target_address'] = data[start:start + 3].hex().upper()
                item_len = 3

            elif frn == 12:  # I073 Time
                parsed['timestamp'] = int.from_bytes(data[start:start + 3], 'big') / 128.0
                item_len = 3

            elif frn == 16:  # I140 Geometric Altitude
                parsed['geometric_altitude'] = struct.unpack('>h', data[start:start + 2])[0] * 6.25
                item_len = 2

            elif frn == 21:  # I145 Flight Level
                fl = struct.unpack('>h', data[start:start + 2])[0] * 0.25
                parsed['flight_level'] = fl * 100.0
                item_len = 2

            elif frn == 24:  # I155 Vertical Rate
                raw = struct.unpack('>h', data[start:start + 2])[0]
                val = (raw & 0x7FFF) - (0x8000 if raw & 0x4000 else 0)
                parsed['vertical_rate'] = val * 6.25
                item_len = 2

            elif frn == 26:  # I160 Ground Vector
                gs_raw = int.from_bytes(data[start:start + 2], 'big') & 0x7FFF
                parsed['ground_speed'] = gs_raw * (2 ** -14) * 3600
                parsed['track_angle'] = int.from_bytes(data[start + 2:start + 4], 'big') * (360.0 / 65536)
                item_len = 4

            elif frn == 29:  # I170 Callsign
                bits = int.from_bytes(data[start:start + 6], 'big')
                tid = ''.join([decode_icao_char((bits >> (42 - i * 6)) & 0x3F) for i in range(8)]).strip()
                parsed['callsign'] = tid
                item_len = 6

            elif frn == 30:  # I020 ECAT
                parsed['aircraft_type'] = data[start]
                item_len = 1

            elif frn == 32:  # I146 Selected Altitude
                raw = int.from_bytes(data[start:start + 2], 'big') & 0x1FFF
                alt = (raw - 0x2000 if raw & 0x1000 else raw) * 25.0
                parsed['selected_altitude'] = alt
                item_len = 2

            elif frn == 23:  # I200 LNAV
                parsed['lnav_mode'] = (data[start] >> 6) & 0x01
                item_len = 1

            else:  # 跳过不关心的字段
                lt = uap_len.get(frn, 0)
                if lt > 0:
                    item_len = lt
                elif lt == 0:
                    item_len = get_variable_field_length(data, start)
                elif lt == -1:
                    if frn == 31:
                        item_len = get_i220_length(data, start)
                    elif frn == 34:
                        item_len = get_i110_length(data, start)
                    elif frn == 39:
                        item_len = get_i250_length(data, start)

            start += item_len

        required = [
            'target_address', 'timestamp', 'latitude', 'longitude',
            'geometric_altitude', 'flight_level', 'ground_speed', 'track_angle', 'vertical_rate',
            'selected_altitude', 'callsign', 'aircraft_type', 'lnav_mode'
        ]
        return parsed if all(k in parsed for k in required) else None

    except Exception:
        return None


# ------------------ 【新】工作函数 (处理单个文件) ------------------
def process_single_file(task_tuple: Tuple) -> Tuple[str, int, int, int]:
    """
    这是一个被多进程“工人”调用的函数，它只负责处理一个文件。

    返回: (文件名, 原始行数, 移除的行数, 写入的行数)
    """
    input_path, output_path, headers = task_tuple
    fname = os.path.basename(input_path)

    try:
        # 使用 .read().splitlines()
        # 通常比 .readlines() 更快
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        original_count = len(lines)
        if original_count == 0:
            return fname, 0, 0, 0  # 文件为空

        parsed_data: Dict[Tuple[str, float], Dict] = {}

        for line in lines:
            hex_str = line.strip().split(',')[-1]
            if not hex_str:
                continue

            ret = parse_cat021_to_dict(hex_str)

            if ret:
                key = (ret['target_address'], ret['timestamp'])
                parsed_data[key] = ret

        rows = list(parsed_data.values())
        deduped_count = len(rows)
        removed_count = original_count - deduped_count

        if not rows:
            return fname, original_count, removed_count, 0  # 无有效数据

        # 排序
        rows.sort(key=lambda x: (x['target_address'], x['timestamp']))

        # 写入
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        return fname, original_count, removed_count, deduped_count

    except Exception as e:
        print(f"文件 {fname} 处理失败: {e}")
        return fname, 0, 0, -1  # -1 作为错误标记


# ------------------ 【修改】主函数 (现在是“管理器”) ------------------
def process_cat_files_to_csv(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, '*.cat')))
    if not files:
        print('未找到 .cat 文件')
        return

    headers = [
        'target_address', 'callsign', 'timestamp', 'latitude', 'longitude',
        'geometric_altitude', 'flight_level', 'ground_speed', 'track_angle', 'vertical_rate',
        'selected_altitude', 'lnav_mode', 'aircraft_type'
    ]

    t0 = time.time()
    total_rows = 0
    total_removed = 0

    # 创建任务列表
    tasks = []
    for path in files:
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname.replace('.cat', '.csv'))
        tasks.append((path, out_path, headers))

    # --- 【多进程核心】 ---
    # 使用所有可用的 CPU 核心
    num_workers = cpu_count()
    print(f"--- 启动 {num_workers} 个并行“工人”处理 {len(files)} 个文件 ---")

    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 来获取结果，效率最高
        # 并用 tqdm 包装来显示文件处理进度
        results = list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc="处理文件"))

    # --- 统计结果 ---
    for res in results:
        fname, original, removed, written = res
        if written > 0:
            total_rows += written
            total_removed += removed
        elif written == -1:
            print(f"警告：文件 {fname} 处理失败。")

    print(f'\n全部完成！总耗时 {time.time() - t0:.1f}s')
    print(f'总共写入 {total_rows} 行有效数据')
    print(f'总共移除 {total_removed} 行重复数据')


# ------------------ 入口 ------------------
if __name__ == '__main__':
    # 【请确认】这是您的 .cat 文件输入目录
    INPUT_DIR = '/mnt/e/adsb'

    # 【请确认】这是您希望保存 .csv 文件的输出目录
    OUTPUT_DIR = '/mnt/d/adsb'

    process_cat_files_to_csv(INPUT_DIR, OUTPUT_DIR)