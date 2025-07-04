#!/usr/bin/env python3
"""
论文总结器模块
功能：
1. 下载 arxiv 论文到 pdfs 目录
2. 使用 minerU 解析为 markdown
3. 使用智谱 AI 的 glm-4-flash 模型总结论文
"""

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import openai
import requests
from loguru import logger

from vars import DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_SUMMARY_LANG


class PaperSummarizer:
    # 類級別的鎖，確保只有一個 minerU 進程在運行
    _mineru_lock = threading.Lock()

    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.mineru_dir = self.data_dir / "mineru"

        # 确保目录存在
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 OpenAI 客户端连接智谱 AI
        self.client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def download_arxiv_paper(self, pid: str) -> Optional[Path]:
        """
        下载 arXiv 论文 PDF

        Args:
            pid: 论文 ID，例如 "2301.00001"

        Returns:
            下载的 PDF 文件路径，失败返回 None
        """
        try:
            # arXiv PDF URL 格式
            pdf_url = f"https://arxiv.org/pdf/{pid}"
            pdf_path = self.pdfs_dir / f"{pid}.pdf"

            # 如果文件已存在，直接返回
            if pdf_path.exists():
                logger.trace(f"PDF file already exists: {pdf_path}")
                return pdf_path

            logger.trace(f"Downloading paper {pid} ...")
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

            logger.trace(f"Paper download complete: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.trace(f"Failed to download paper {pid}: {e}")
            return None

    def parse_pdf_with_mineru(self, pdf_path: Path) -> Optional[Path]:
        """
        使用 minerU 解析 PDF 为 Markdown (带单例锁保护)

        Args:
            pdf_path: PDF 文件路径

        Returns:
            生成的 Markdown 文件路径，失败返回 None
        """
        try:
            pdf_name = pdf_path.stem
            output_dir = self.mineru_dir

            # 检查是否已经解析过
            expected_md_path = output_dir / pdf_name / "auto" / f"{pdf_name}.md"
            if expected_md_path.exists():
                logger.trace(f"Markdown file already exists: {expected_md_path}")
                return expected_md_path

            # 获取鎖，確保只有一個 minerU 進程在運行
            logger.trace(f"Waiting for minerU lock to parse PDF: {pdf_path}")
            with self._mineru_lock:
                logger.trace(f"Acquired minerU lock, start parsing PDF: {pdf_path}")

                # 再次检查是否已经解析过（避免等待锁期间其他进程已完成解析）
                if expected_md_path.exists():
                    logger.trace(f"File generated during lock wait: {expected_md_path}")
                    return expected_md_path

                # 构建 minerU 命令
                cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-l", "en", "-d", "cuda", "--vram", "2"]

                logger.trace(f"Executing command: {' '.join(cmd)}")
                start_time = time.time()

                # 执行命令
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                elapsed_time = time.time() - start_time
                logger.trace(f"minerU execution complete, elapsed {elapsed_time:.2f} seconds")

                if result.returncode != 0:
                    logger.trace(f"minerU execution failed: {result.stderr}")
                    # minerU 解析失败时删除 PDF 文件
                    try:
                        pdf_path.unlink()
                        logger.trace(f"Parse failed, deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")
                    return None

                # 检查生成的 Markdown 文件
                if expected_md_path.exists():
                    logger.trace(f"PDF parse complete: {expected_md_path}")

                    # 解析完成后删除 PDF 文件以节省空间
                    try:
                        pdf_path.unlink()
                        logger.trace(f"Deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")

                    # 清理除了 images 和 markdown 之外的其他文件
                    self._cleanup_mineru_output(output_dir / pdf_name)

                    return expected_md_path
                else:
                    logger.trace(f"Generated Markdown file not found: {expected_md_path}")
                    # 解析失败时删除 PDF 文件
                    try:
                        pdf_path.unlink()
                        logger.trace(f"Parse failed, deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")
                    return None

        except subprocess.TimeoutExpired:
            logger.trace("minerU execution timeout")
            # 解析超时时删除 PDF 文件
            try:
                pdf_path.unlink()
                logger.trace(f"Parse timeout, deleted PDF source file: {pdf_path}")
            except Exception as e:
                logger.trace(f"Failed to delete PDF file: {e}")
            return None
        except Exception as e:
            logger.trace(f"PDF parse failed: {e}")
            # 解析异常时删除 PDF 文件
            try:
                pdf_path.unlink()
                logger.trace(f"Parse exception, deleted PDF source file: {pdf_path}")
            except Exception as e:
                logger.trace(f"Failed to delete PDF file: {e}")
            return None

    def _cleanup_mineru_output(self, output_path: Path):
        """
        清理 minerU 输出目录，只保留 images 和 markdown 文件

        Args:
            output_path: minerU 输出的论文目录路径 (如 data/mineru/2507.01679)
        """
        try:
            auto_dir = output_path / "auto"
            if not auto_dir.exists():
                return

            # 需要保留的文件模式
            paper_id = output_path.name
            keep_files = {f"{paper_id}.md"}  # 保留 markdown 文件
            keep_dirs = {"images"}  # 保留 images 目录

            items_to_remove = []

            for item in auto_dir.iterdir():
                if item.is_dir():
                    if item.name not in keep_dirs:
                        # 标记删除不需要的目录
                        items_to_remove.append(item)
                elif item.is_file():
                    if item.name not in keep_files:
                        # 标记删除不需要的文件（JSON、PDF等中间文件）
                        items_to_remove.append(item)

            # 执行删除操作
            for item in items_to_remove:
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        logger.trace(f"Deleted directory: {item}")
                    else:
                        item.unlink(missing_ok=True)
                        logger.trace(f"Deleted file: {item}")
                except Exception as e:
                    logger.trace(f"Failed to delete {item}: {e}")

        except Exception as e:
            logger.trace(f"Failed to clean minerU output: {e}")

    def extract_main_content(self, markdown_content: str) -> str:
        """
        截取论文正文，去掉 references、acknowledgements 等部分

        Args:
            markdown_content: 完整的 Markdown 内容

        Returns:
            截取后的正文内容
        """
        try:
            # 常见的结束标识符（不区分大小写）
            end_patterns = [
                r"(?i)^#+\s*references?\s*$",
                r"(?i)^#+\s*acknowledgements?\s*$",
                r"(?i)^#+\s*appendix\s*$",
                r"(?i)^#+\s*appendices\s*$",
            ]

            lines = markdown_content.split("\n")
            end_index = len(lines)

            # 找到最早出现的结束标识符
            for i, line in enumerate(lines):
                line = line.strip()
                for pattern in end_patterns:
                    if re.match(pattern, line):
                        end_index = min(end_index, i)
                        break

            # 截取正文部分
            main_content = "\n".join(lines[:end_index])

            # 如果截取后的内容太短，返回原内容
            if len(main_content.strip()) < len(markdown_content.strip()) * 0.25:
                logger.trace("Extracted main content too short, using original content")
                return markdown_content

            logger.trace(f"Paper content truncated from {len(markdown_content)} to {len(main_content)} characters")
            return main_content

        except Exception as e:
            logger.trace(f"Failed to extract main content: {e}")
            return markdown_content

    def summarize_with_llm(self, markdown_content: str) -> str:
        """
        使用智谱 AI 总结论文内容

        Args:
            markdown_content: Markdown 格式的论文内容

        Returns:
            论文总结
        """
        try:
            # 根据配置选择语言 prompt
            if LLM_SUMMARY_LANG == "en":
                # 英文 prompt
                prompt = f"""
You are a senior researcher in deep learning and computer science, skilled at analyzing and summarizing academic papers. Please carefully read the following paper content and provide a professional, accurate, and structured **English** summary.

<Summary Requirements>
1. Use academic and rigorous language, formulas are allowed
2. Use `$…$` (inline) or `$$…$$` (block) for all formulas
3. Each section should have specific content, avoid vague descriptions
4. Use appropriate subheadings, lists, and emphasis to organize content
</Summary Requirements>

<Paper Content>
{markdown_content}
</Paper Content>

Output should contain two parts, detailed summary and TL;DR, in the following format:
<Output Format>
## Detailed Paper Summary
Your detailed summary of this paper

## TL;DR
Your brief summary of this paper
</Output Format>

<Notes>
1. Ensure the summary accurately reflects the paper content, **do not add information not in the paper** except for personal insights
2. Do not guess or fabricate data, rather skip quantitative analysis than summarize incorrect data
3. Please summarize in **English**
</Notes>
"""
            else:
                # 中文 prompt (默认)
                prompt = f"""
你是一位深度学习和计算机科学领域的资深研究员，擅长分析和总结学术论文。请仔细阅读以下论文内容，并提供一个专业、准确、结构化的**中文**总结。

<总结要求>
1. 要是用学术且严谨的语言，可以适当使用公式，
2. 所有公式用 `$…$`（行内）或 `$$…$$`（独立）
3. 每个部分都要有具体内容，避免空泛的描述
4. 适当使用子标题、列表和重点标记来组织内容
<总结要求>

<论文内容>
{markdown_content}
</论文内容>

输出包含两部分，分别是详细总结和 TL;DR，按照如下格式：
<输出格式>
## 论文详细总结
你对这篇论文的详细总结

## TL;DR
你对这篇论文的简短总结
</输出格式>

<注意事项>
1. 请确保总结的时候准确反映论文内容，除了个人见解部分**不要添加论文中没有的信息**
2. 不要猜测更不要臆造数据，宁可不总结论文中的定量分析，也不要总结错误数据
3. 请一定用**中文**进行总结
</注意事项>
"""

            modelid = "glm-z1-flash"
            logger.trace(f"Calling {modelid} to generate paper summary...")

            response = self.client.chat.completions.create(
                model=modelid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=8192,
            )

            summary = response.choices[0].message.content

            # 提取 </think> 之后的内容
            # logger.trace(f"原始总结内容:\n{summary}")
            if "</think>" in summary:
                summary = summary.split("</think>", 1)[1].strip()
            else:
                summary = summary.strip()

            # 解析详细总结和TL;DR部分
            parsed_summary = self._parse_summary_sections(summary)
            logger.trace("Paper summary generation complete")
            return parsed_summary

        except Exception as e:
            logger.trace(f"LLM summary failed: {e}")
            return f"# 错误\n\n总结生成失败: {str(e)}"

    def _parse_summary_sections(self, summary: str) -> str:
        """
        解析总结内容，提取详细总结和TL;DR部分，并重新组织格式（TL;DR在前）

        Args:
            summary: 原始总结内容

        Returns:
            重新组织后的总结内容
        """
        try:
            import re

            # 新的解析逻辑：基于Markdown标题格式
            # 提取详细总结部分 (## 论文详细总结 之后的内容)
            detailed_match = re.search(r"##\s*论文详细总结\s*\n(.*?)(?=##\s*TL;DR|$)", summary, re.DOTALL)
            detailed_summary = detailed_match.group(1).strip() if detailed_match else ""

            # 提取TL;DR部分 (## TL;DR 之后的内容)
            tldr_match = re.search(r"##\s*TL;DR\s*\n(.*?)(?=##|$)", summary, re.DOTALL)
            tldr_summary = tldr_match.group(1).strip() if tldr_match else ""

            # 重新组织格式：TL;DR在前，详细总结在后
            if tldr_summary and detailed_summary:
                formatted_summary = f"""## TL;DR

{tldr_summary}

## 详细总结

{detailed_summary}"""
                return formatted_summary
            elif detailed_summary:
                # 如果只有详细总结，添加标题后返回
                return f"## 详细总结\n\n{detailed_summary}"
            elif tldr_summary:
                # 如果只有TL;DR，也返回
                return f"## TL;DR\n\n{tldr_summary}"
            else:
                # 如果都没有提取到，返回原始内容
                logger.trace("Failed to parse detailed summary and TL;DR, returning original content")
                return summary

        except Exception as e:
            logger.trace(f"Failed to parse summary sections: {e}")
            return summary

    def generate_summary(self, pid: str) -> str:
        """
        生成论文总结的主要入口函数

        Args:
            pid: 论文 ID

        Returns:
            论文总结的 Markdown 内容
        """
        try:
            # Step 0: 预先检查是否已有解析结果
            expected_md_path = self.mineru_dir / pid / "auto" / f"{pid}.md"
            if expected_md_path.exists():
                logger.trace(f"Found existing parse result: {expected_md_path}")
                # 直接读取 Markdown 内容
                with open(expected_md_path, encoding="utf-8") as f:
                    markdown_content = f.read()

                if markdown_content.strip():
                    # Step 3: 截取论文正文
                    main_content = self.extract_main_content(markdown_content)
                    # Step 4: 使用 LLM 生成总结
                    logger.info(f"Summarizing {pid}.pdf ...")
                    summary = self.summarize_with_llm(main_content)
                    return summary
                else:
                    logger.trace("Existing Markdown file content is empty, re-parsing")

            # Step 1: 下载论文 PDF
            logger.info(f"Downloading {pid}.pdf ...")
            pdf_path = self.download_arxiv_paper(pid)
            if not pdf_path:
                return "# 错误\n\n无法下载论文 PDF"

            # Step 2: 使用 minerU 解析为 Markdown
            logger.info(f"Parsing {pid}.pdf ...")
            md_path = self.parse_pdf_with_mineru(pdf_path)
            if not md_path:
                return "# 错误\n\n无法解析 PDF 为 Markdown"

            # Step 3: 读取 Markdown 内容
            with open(md_path, encoding="utf-8") as f:
                markdown_content = f.read()

            if not markdown_content.strip():
                return "# 错误\n\n解析的 Markdown 内容为空"

            # Step 4: 截取论文正文
            main_content = self.extract_main_content(markdown_content)

            # Step 5: 使用 LLM 生成总结
            logger.info(f"Summarizing {pid}.pdf ...")
            summary = self.summarize_with_llm(main_content)

            return summary

        except Exception as e:
            logger.error(f"Error occurred while generating paper summary: {e}")
            return f"# Error\n\nFailed to generate summary: {str(e)}"


# 全局实例
_summarizer = None


def get_summarizer() -> PaperSummarizer:
    """获取全局 PaperSummarizer 实例"""
    global _summarizer
    if _summarizer is None:
        _summarizer = PaperSummarizer()
    return _summarizer


def generate_paper_summary(pid: str) -> str:
    """
    生成论文总结的外部接口函数

    Args:
        pid: 论文 ID

    Returns:
        论文总结的 Markdown 内容
    """
    summarizer = get_summarizer()
    return summarizer.generate_summary(pid)


if __name__ == "__main__":
    # 测试代码
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    if len(sys.argv) > 1:
        test_pid = sys.argv[1]
        logger.trace(f"Test paper ID: {test_pid}")
        summary = generate_paper_summary(test_pid)
        logger.trace("\n" + "=" * 50)
        logger.trace("Paper summary:")
        logger.trace("=" * 50)
        logger.trace(summary)
    else:
        logger.trace("Usage: python paper_summarizer.py <paper_id>")
        logger.trace("Example: python paper_summarizer.py 2301.00001")
