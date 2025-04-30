#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Jupyter Notebook转换为HTML格式的脚本
使用方法：

```bash
python convert_notebook_to_html.py test.ipynb
```
注意：
这个脚本需要安装`nbconvert`包，可以通过`pip install nbconvert`安装
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def convert_notebook_to_html(notebook_path, output_dir=None, theme=None):
    """
    将Jupyter Notebook转换为HTML格式

    Args:
        notebook_path: Jupyter Notebook文件路径
        output_dir: 输出目录，默认为notebook所在目录
        theme: HTML样式主题，例如'light', 'dark', 等

    Returns:
        生成的HTML文件路径
    """
    notebook_path = Path(notebook_path).resolve()

    if not notebook_path.exists():
        print(f"错误：找不到文件 {notebook_path}")
        return None

    # Output
    if output_dir is None:
        output_dir = notebook_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件名
    output_file = output_dir / f"{notebook_path.stem}.html"

    # 构建nbconvert命令
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        str(notebook_path),
        "--output",
        str(output_file.name),
    ]

    # 添加主题设置（如果指定）
    if theme:
        cmd.extend(["--template", theme])

    try:
        # 执行转换命令
        print(f"开始转换 {notebook_path} 到 HTML 格式...")
        subprocess.run(cmd, check=True, cwd=output_dir)
        print(f"转换成功！HTML文件已保存到：{output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"转换过程中出错：{e}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="将Jupyter Notebook转换为HTML")
    parser.add_argument("notebook", help="Jupyter Notebook文件路径")
    parser.add_argument("--output-dir", "-o", help="输出目录")
    parser.add_argument("--theme", "-t", help="HTML样式主题")

    args = parser.parse_args()

    html_path = convert_notebook_to_html(args.notebook, args.output_dir, args.theme)

    if html_path:
        print(f"转换完成: {html_path}")
        return 0
    else:
        print("转换失败")
        return 1


if __name__ == "__main__":
    # 检查nbconvert是否已安装
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--version"], check=True, stdout=subprocess.PIPE
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: jupyter nbconvert 未安装或无法运行。")
        print("请使用以下命令安装: pip install nbconvert")
        sys.exit(1)

    sys.exit(main())
