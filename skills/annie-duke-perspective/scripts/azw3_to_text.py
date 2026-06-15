#!/usr/bin/env python3
"""azw3 → 纯文本转换器
支持 AZW3 / MOBI / EPUB。输出 UTF-8 纯文本，去除元数据噪音。
"""
import sys
import os
import tempfile
from pathlib import Path
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'meta'):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'meta'):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ext = TextExtractor()
    ext.feed(html)
    text = ''.join(ext.parts)
    return '\n'.join(line.strip() for line in text.split('\n') if line.strip())


def convert_azw3(src: Path, dst: Path) -> dict:
    """azw3 → text via mobi.extract + ebooklib"""
    import mobi
    from ebooklib import epub

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # 1. mobi.extract(infile) -> (tempdir, epub_path) tuple
        try:
            result = mobi.extract(str(src))
            # 兼容不同版本：可能是 str 也可能是 tuple
            if isinstance(result, tuple):
                epub_path_str = result[1] if len(result) > 1 else result[0]
            else:
                epub_path_str = result
        except Exception as e:
            return {"ok": False, "error": f"mobi.extract failed: {e}"}

        epub_path = Path(epub_path_str)
        if not epub_path.exists():
            return {"ok": False, "error": f"extract returned missing path: {epub_path_str}"}

        # 2. ebooklib 解析 epub
        try:
            book = epub.read_epub(str(epub_path))
        except Exception as e:
            return {"ok": False, "error": f"epub.read_epub failed: {e}"}

        # 3. 抽取所有 document (ITEM_DOCUMENT = 9)
        parts = []
        items = list(book.get_items_of_type(9))
        items.sort(key=lambda x: x.get_name())  # 文件名排序保证章节顺序

        for item in items:
            try:
                content = item.get_content().decode('utf-8', errors='ignore')
                text = html_to_text(content)
                # 太短的可能是版权页/封面/导航
                if text and len(text) > 100:
                    parts.append(text)
            except Exception as e:
                parts.append(f"[解析失败: {item.get_name()}: {e}]")

        full_text = '\n\n'.join(parts)
        dst.write_text(full_text, encoding='utf-8')

        return {
            "ok": True,
            "input_size": src.stat().st_size,
            "output_chars": len(full_text),
            "output_lines": full_text.count('\n') + 1,
            "epub_items_used": len(parts),
            "epub_items_total": len(items),
        }


def main():
    if len(sys.argv) < 3:
        print("Usage: azw3_to_text.py <input.azw3> <output.txt>")
        sys.exit(1)

    src = Path(sys.argv[1]).expanduser()
    dst = Path(sys.argv[2]).expanduser()

    if not src.exists():
        print(f"ERROR: input not found: {src}")
        sys.exit(1)

    print(f"Converting: {src.name} ({src.stat().st_size:,} bytes)")
    result = convert_azw3(src, dst)
    if not result.get("ok"):
        print(f"FAILED: {result.get('error')}")
        sys.exit(2)

    print(f"OK → {dst}")
    print(f"  output chars: {result['output_chars']:,}")
    print(f"  output lines: {result['output_lines']:,}")
    print(f"  epub items used: {result['epub_items_used']} / {result['epub_items_total']}")


if __name__ == "__main__":
    main()
