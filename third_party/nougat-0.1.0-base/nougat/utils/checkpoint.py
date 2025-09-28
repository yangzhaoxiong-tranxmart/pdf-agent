"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Optional
import requests
import os
import tqdm
import io
from pathlib import Path
import torch
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
MODEL_TAG = "0.1.0-small"


def create_robust_session():
    """
    创建一个具有重试机制的robust requests session
    """
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=5,  # 总重试次数
        backoff_factor=1,  # 重试间隔的倍数
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # 允许重试的HTTP方法
    )
    
    # 创建HTTP适配器
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置超时和连接参数
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    return session


def download_as_bytes_with_progress(url: str, name: str = None, max_retries: int = 3) -> bytes:
    """
    使用robust机制下载文件并返回字节内容，带进度条

    Args:
        url: 要下载的文件URL
        name: 文件名，用于显示进度条。如果为None，则使用URL
        max_retries: 最大重试次数

    Returns:
        bytes: 文件内容
    """
    if name is None:
        name = url.split('/')[-1] if '/' in url else url
    
    for attempt in range(max_retries):
        try:
            session = create_robust_session()
            
            # 首先发送HEAD请求获取文件大小
            head_resp = session.head(url, allow_redirects=True, timeout=30)
            total = int(head_resp.headers.get("content-length", 0))
            
            if total == 0:
                # 如果HEAD请求无法获取大小，尝试GET请求
                resp = session.get(url, stream=True, allow_redirects=True, timeout=30)
                total = int(resp.headers.get("content-length", 0))
            else:
                # 使用GET请求下载
                resp = session.get(url, stream=True, allow_redirects=True, timeout=30)
            
            resp.raise_for_status()  # 检查HTTP错误
            
            bio = io.BytesIO()
            downloaded = 0
            
            with tqdm.tqdm(
                desc=f"{name} (尝试 {attempt + 1}/{max_retries})",
                total=total,
                unit="b",
                unit_scale=True,
                unit_divisor=1024,
                initial=downloaded
            ) as bar:
                
                # 使用更小的chunk size和更robust的读取方式
                for chunk in resp.iter_content(chunk_size=32768):  # 32KB chunks
                    if chunk:  # 过滤掉keep-alive chunks
                        bio.write(chunk)
                        downloaded += len(chunk)
                        bar.update(len(chunk))
            
            # 验证下载完整性
            if total > 0 and downloaded != total:
                raise requests.exceptions.ChunkedEncodingError(
                    f"下载不完整: 期望 {total} 字节，实际下载 {downloaded} 字节"
                )
            
            print(f"✓ 成功下载 {name} ({downloaded} 字节)")
            return bio.getvalue()
            
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            
            print(f"✗ 下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                # 指数退避重试
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"所有重试都失败了，无法下载 {name}")
                raise e


def check_file_complete(url: str, filepath: Path) -> bool:
    """
    检查文件是否已经完整下载
    
    Args:
        url: 下载URL
        filepath: 文件路径
        
    Returns:
        bool: 文件是否完整
    """
    if not filepath.exists():
        return False
    
    try:
        session = create_robust_session()
        # 发送HEAD请求获取文件大小
        head_resp = session.head(url, allow_redirects=True, timeout=10)
        expected_size = int(head_resp.headers.get("content-length", 0))
        actual_size = filepath.stat().st_size
        
        if expected_size > 0 and actual_size == expected_size:
            print(f"✓ 文件 {filepath.name} 已完整下载 ({actual_size} 字节)")
            return True
        else:
            print(f"⚠ 文件 {filepath.name} 不完整: 期望 {expected_size} 字节，实际 {actual_size} 字节")
            return False
    except Exception as e:
        print(f"⚠ 无法验证文件完整性: {str(e)}")
        return False


def download_with_resume(url: str, filepath: Path, name: str = None, max_retries: int = 3) -> bool:
    """
    支持断点续传的下载函数
    
    Args:
        url: 下载URL
        filepath: 保存路径
        name: 显示名称
        max_retries: 最大重试次数
        
    Returns:
        bool: 下载是否成功
    """
    if name is None:
        name = filepath.name
    
    # 首先检查文件是否已经完整
    if check_file_complete(url, filepath):
        return True
    
    for attempt in range(max_retries):
        try:
            session = create_robust_session()
            
            # 检查是否已存在部分文件
            resume_header = {}
            initial_pos = 0
            if filepath.exists():
                initial_pos = filepath.stat().st_size
                resume_header['Range'] = f'bytes={initial_pos}-'
                print(f"发现已下载 {initial_pos} 字节，尝试断点续传...")
            
            # 发送请求
            resp = session.get(url, headers=resume_header, stream=True, allow_redirects=True, timeout=30)
            
            # 检查服务器是否支持断点续传
            if initial_pos > 0 and resp.status_code == 206:  # Partial Content
                print("✓ 服务器支持断点续传")
                total = int(resp.headers.get('content-range', '').split('/')[-1])
            elif initial_pos > 0 and resp.status_code == 200:
                print("⚠ 服务器不支持断点续传，重新下载")
                filepath.unlink()  # 删除不完整的文件
                initial_pos = 0
                total = int(resp.headers.get("content-length", 0))
            else:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
            
            # 如果文件已经完整，直接返回成功
            if initial_pos >= total and total > 0:
                print(f"✓ 文件 {name} 已经完整，跳过下载")
                return True
            
            # 打开文件进行写入
            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(filepath, mode) as f, tqdm.tqdm(
                desc=f"{name} (尝试 {attempt + 1}/{max_retries})",
                total=total,
                unit="b",
                unit_scale=True,
                unit_divisor=1024,
                initial=initial_pos
            ) as bar:
                
                for chunk in resp.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            # 验证文件完整性
            final_size = filepath.stat().st_size
            if total > 0 and final_size != total:
                raise requests.exceptions.ChunkedEncodingError(
                    f"下载不完整: 期望 {total} 字节，实际 {final_size} 字节"
                )
            
            print(f"✓ 成功下载 {name} ({final_size} 字节)")
            return True
            
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            
            print(f"✗ 下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"所有重试都失败了，无法下载 {name}")
                return False
    
    return False


def download_checkpoint(checkpoint: Path, model_tag: str = MODEL_TAG, use_resume: bool = True):
    """
    下载Nougat模型检查点，使用robust下载机制

    Args:
        checkpoint (Path): 检查点保存路径
        model_tag (str): 要下载的模型标签，默认为"0.1.0-small"
        use_resume (bool): 是否使用断点续传功能，默认为True
    """
    print(f"正在下载nougat检查点版本 {model_tag} 到路径 {checkpoint}")
    files = [
        "config.json",
        "pytorch_model.bin", 
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    for file in files:
        download_url = f"{BASE_URL}/{model_tag}/{file}"
        file_path = checkpoint / file
        
        print(f"\n检查文件: {file}")
        
        # 首先检查文件是否已经完整
        if check_file_complete(download_url, file_path):
            print(f"✓ 文件 {file} 已存在且完整，跳过下载")
            continue
        
        print(f"开始下载: {file}")
        
        if use_resume:
            # 使用断点续传下载
            success = download_with_resume(download_url, file_path, file)
            if not success:
                print(f"断点续传失败，尝试普通下载...")
                try:
                    binary_file = download_as_bytes_with_progress(download_url, file)
                    if len(binary_file) > 15:  # 完整性检查
                        file_path.write_bytes(binary_file)
                        print(f"✓ 普通下载成功: {file}")
                    else:
                        print(f"✗ 下载的文件太小，可能损坏: {file}")
                except Exception as e:
                    print(f"✗ 普通下载也失败: {file} - {str(e)}")
        else:
            # 使用普通下载
            try:
                binary_file = download_as_bytes_with_progress(download_url, file)
                if len(binary_file) > 15:  # 完整性检查
                    file_path.write_bytes(binary_file)
                    print(f"✓ 下载成功: {file}")
                else:
                    print(f"✗ 下载的文件太小，可能损坏: {file}")
            except Exception as e:
                print(f"✗ 下载失败: {file} - {str(e)}")
    
    print(f"\n检查点下载完成，保存到: {checkpoint}")


def torch_hub(model_tag: Optional[str] = MODEL_TAG) -> Path:
    old_path = Path(torch.hub.get_dir() + "/nougat")
    if model_tag is None:
        model_tag = MODEL_TAG
    hub_path = old_path.with_name(f"nougat-{model_tag}")
    if old_path.exists():
        # move to new format
        old_path.rename(old_path.with_name("nougat-0.1.0-small"))
    return hub_path


def get_checkpoint(
    checkpoint_path: Optional[os.PathLike] = None,
    model_tag: str = MODEL_TAG,
    download: bool = True,
) -> Path:
    """
    Get the path to the Nougat model checkpoint.

    This function retrieves the path to the Nougat model checkpoint. If the checkpoint does not
    exist or is empty, it can optionally download the checkpoint.

    Args:
        checkpoint_path (Optional[os.PathLike]): The path to the checkpoint. If not provided,
            it will check the "NOUGAT_CHECKPOINT" environment variable or use the default location.
            Default is None.
        model_tag (str): The model tag to download. Default is "0.1.0-small".
        download (bool): Whether to download the checkpoint if it doesn't exist or is empty.
            Default is True.

    Returns:
        Path: The path to the Nougat model checkpoint.
    """
    print('''                                                                       
                                            ........                                    
                                            ..]@`...                                    
                                            ./@@@@......                                
                                            =@@OOO@\`...                                
                                    ........//@OOOOO@@`.....                            
                                    ...]/@@@@^@OOOOOOO@\....                            
                            ......,/@@@OOOOO@^\@OOOOOOO@@@`.....                        
                            ..../@@OOOOOOOOOOO\@@OOOOOOOOO@@]...                        
                            ..,@@OOOOOOOOOOOOOOOOOOOOOOOOOOO@@`.....                    
                            ./@OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@\....                    
                        ....@@OOO@@@/@OOOOOOOOOOOOOOOOOOOOOOOOO@\...                    
                        .,/@@OOO@@@@@.@OOOOOOOOOOOOOOOOOOOOOOOOO@@..                    
            . ........,/@@OOOOOOO@@@@@OOOOOOOOOOOOOOOOOOOOOOOOOOO@@.....                
             . ..,]@@@@OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@@.. .                
            ...=@@@@OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@^...                
            ...=@@/OOOOO[`.......[OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@@...                
             ...@\..    ....        .,[[*....=OOOOOOOOOOOOOOOOOOOOO@@...                
            ....,@\.    ....        .. .......,OOOOOOOOOOOOOOOOOOOOO@...                
                ..@@@@@@@@/.                ...=OOOOOOOOOOOOOOOOOOOO@...                
                ...=@\......                ....OOOOOOOOOOOOOOOOOOOO@...                
                    .,@@]`......................OOOOOOOOOOOOOOOOOOO@@...                
                    .....[@@@@^.      ..    .../OOOOOOOOOOOOOOOOOOO@@....               
                ...........\@................./OOOOOOOOOOOOOOOOOOOO@/....               
                ......=@@@@@@@@@]]]]`........OOOOOOOOO@@@@O@@@@@@@@@@\..                
               ....../@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@^.                
                ....=@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@`.                
                ....@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@O@\...                
                ...=@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@OOO@@@@@@@@OOO@@...                
               ....=@@@@@@@@@@@@@@@@@@@@@@@@@@@OOOOOOOOOO@@@@@@@@OOO@...                
               ....@@@@@@@@@@@@@@@@@@@@@@@OOOOOOOOOOOOOOOOO@@@@@@@OO@^..                
               ....=@@@@@@@@@@@@@@@@@OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@^..                
               ....=@@@@@@@@@@@@O/,OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@@..                
               .....\@@@@@@@O/`..,OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@..                
               .......@@/[......,OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO@^                 

    ''')
    checkpoint = Path(
        checkpoint_path or os.environ.get("NOUGAT_CHECKPOINT", torch_hub(model_tag))
    )
    if checkpoint.exists() and checkpoint.is_file():
        checkpoint = checkpoint.parent
    if download and (not checkpoint.exists() or len(os.listdir(checkpoint)) < 5):
        checkpoint.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint, model_tag=model_tag or MODEL_TAG)
    return checkpoint


if __name__ == "__main__":
    get_checkpoint()
