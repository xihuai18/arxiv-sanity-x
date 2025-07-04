"""
批量論文總結器

功能：
1. 獲取資料庫中最新的 n 篇論文
2. 內建非單例的論文總結器進行 minerU 解析和 LLM 總結
3. 自動緩存總結到 serve.py 使用的目錄
4. 支援多線程並發處理以提高效率
5. 提供詳細的進度追蹤和錯誤處理
"""

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import requests
from loguru import logger
from tqdm import tqdm

from aslite.db import get_metas_db, get_papers_db
from vars import DATA_DIR, LLM_API_KEY, LLM_BASE_URL


def calculate_chinese_ratio(text: str) -> float:
    """
    计算文本中中文字符的占比

    Args:
        text: 输入文本

    Returns:
        float: 中文字符占比 (0.0 到 1.0)
    """
    if not text or not text.strip():
        return 0.0

    # 移除空白字符后的文本
    clean_text = re.sub(r"\s+", "", text)
    if not clean_text:
        return 0.0

    # 统计中文字符数量 (包括中文标点符号)
    chinese_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", clean_text)
    chinese_count = len(chinese_chars)

    # 计算占比
    total_chars = len(clean_text)
    ratio = chinese_count / total_chars if total_chars > 0 else 0.0

    return ratio


class BatchPaperSummarizer:
    """
    批量論文總結器類
    內建非單例的論文處理邏輯，支援真正的並發處理
    """

    def __init__(self, processor=None):
        """
        初始化論文總結器

        Args:
            processor: BatchProcessor實例，用於記錄錯誤詳情
        """
        self.data_dir = Path(DATA_DIR)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.mineru_dir = self.data_dir / "mineru"
        self.cache_dir = Path("data/summary")
        self.processor = processor  # 用於記錄錯誤詳情

        # 确保目录存在
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 OpenAI 客户端连接智谱 AI
        self.client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def _record_failure_detail(self, pid: str, reason: str, message: str, exception: Exception = None):
        """記錄失敗詳情到processor"""
        if self.processor:
            self.processor._record_failure_detail(pid, reason, message, exception)

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
                logger.trace(f"PDF 文件已存在: {pdf_path}")
                return pdf_path

            logger.trace(f"正在下载论文 {pid} ...")
            response = requests.get(pdf_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

            logger.trace(f"论文下载完成: {pdf_path}")
            return pdf_path

        except Exception as e:
            error_msg = f"下载论文失败 {pid}: {str(e)}"
            logger.error(error_msg)
            # 记录详细错误信息
            self._record_failure_detail(pid, "download_failed", error_msg, e)
            return None

    def parse_pdf_with_mineru(self, pdf_path: Path) -> Optional[Path]:
        """
        使用 minerU 解析 PDF 为 Markdown
        添加了文件級別的鎖避免同一論文的並發解析衝突

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
                logger.trace(f"Markdown 文件已存在: {expected_md_path}")
                return expected_md_path

            # 使用文件級別的鎖，避免同一論文的並發解析
            lock_file = output_dir / f"{pdf_name}.lock"

            # 嘗試創建鎖文件
            try:
                lock_file.touch(exist_ok=False)
                logger.trace(f"獲得解析鎖: {pdf_name}")
            except FileExistsError:
                # 鎖文件已存在，等待其他進程完成
                logger.trace(f"等待其他進程解析 {pdf_name}...")
                max_wait = 600  # 最多等待10分鐘
                wait_time = 0
                while lock_file.exists() and wait_time < max_wait:
                    time.sleep(5)
                    wait_time += 5
                    if expected_md_path.exists():
                        logger.trace(f"其他進程已完成解析: {expected_md_path}")
                        return expected_md_path

                if wait_time >= max_wait:
                    logger.warning(f"等待解析鎖超時: {pdf_name}")
                    # 清理可能的死鎖
                    try:
                        lock_file.unlink(missing_ok=True)
                    except Exception:
                        pass

            try:
                logger.trace(f"開始解析 PDF: {pdf_path}")

                # 构建 minerU 命令
                cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-l", "en", "-d", "cuda", "--vram", "2"]

                logger.trace(f"执行命令: {' '.join(cmd)}")
                start_time = time.time()

                # 执行命令
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                elapsed_time = time.time() - start_time
                logger.trace(f"minerU 執行完成，耗時 {elapsed_time:.2f} 秒")

                if result.returncode != 0:
                    error_msg = f"minerU 执行失败 (返回码: {result.returncode}): {result.stderr}"
                    logger.error(error_msg)
                    # 记录详细错误信息
                    self._record_failure_detail(
                        pdf_path.stem,
                        "parse_failed",
                        error_msg,
                        Exception(f"minerU返回码: {result.returncode}, 错误: {result.stderr}"),
                    )
                    # minerU 解析失败时删除 PDF 文件
                    try:
                        pdf_path.unlink()
                        logger.trace(f"解析失败，已删除 PDF 源文件: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"删除 PDF 文件失败: {e}")
                    return None

                # 检查生成的 Markdown 文件
                if expected_md_path.exists():
                    logger.trace(f"PDF 解析完成: {expected_md_path}")

                    # 解析完成后删除 PDF 文件以节省空间
                    try:
                        pdf_path.unlink()
                        logger.trace(f"已删除 PDF 源文件: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"删除 PDF 文件失败: {e}")

                    # 清理除了 images 和 markdown 之外的其他文件
                    self._cleanup_mineru_output(output_dir / pdf_name)

                    return expected_md_path
                else:
                    error_msg = f"未找到生成的 Markdown 文件: {expected_md_path}"
                    logger.error(error_msg)
                    # 记录详细错误信息
                    self._record_failure_detail(
                        pdf_path.stem, "parse_failed", error_msg, Exception("minerU未生成预期的Markdown文件")
                    )
                    # 解析失败时删除 PDF 文件
                    try:
                        pdf_path.unlink()
                        logger.trace(f"解析失败，已删除 PDF 源文件: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"删除 PDF 文件失败: {e}")
                    return None

            finally:
                # 清理鎖文件
                try:
                    lock_file.unlink(missing_ok=True)
                    logger.trace(f"釋放解析鎖: {pdf_name}")
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            logger.trace("minerU 执行超时")
            # 解析超时时删除 PDF 文件
            try:
                pdf_path.unlink()
                logger.trace(f"解析超时，已删除 PDF 源文件: {pdf_path}")
            except Exception as e:
                logger.trace(f"删除 PDF 文件失败: {e}")
            return None
        except Exception as e:
            logger.trace(f"PDF 解析失败: {e}")
            # 解析异常时删除 PDF 文件
            try:
                pdf_path.unlink()
                logger.trace(f"解析异常，已删除 PDF 源文件: {pdf_path}")
            except Exception as e:
                logger.trace(f"删除 PDF 文件失败: {e}")
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
                        logger.trace(f"已删除目录: {item}")
                    else:
                        item.unlink(missing_ok=True)
                        logger.trace(f"已删除文件: {item}")
                except Exception as e:
                    logger.trace(f"删除 {item} 失败: {e}")

        except Exception as e:
            logger.trace(f"清理 minerU 输出失败: {e}")

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
                logger.trace("截取的正文内容过短，使用原始内容")
                return markdown_content

            logger.trace(f"论文内容从 {len(markdown_content)} 字符截取到 {len(main_content)} 字符")
            return main_content

        except Exception as e:
            logger.trace(f"截取正文失败: {e}")
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
            # 构建提示词
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
            logger.trace(f"正在调用 {modelid} 生成论文总结...")

            response = self.client.chat.completions.create(
                model=modelid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=8192,
            )

            summary = response.choices[0].message.content

            # 提取 </think> 之后的内容
            if "</think>" in summary:
                summary = summary.split("</think>", 1)[1].strip()
            else:
                summary = summary.strip()

            # 解析详细总结和TL;DR部分
            parsed_summary = self._parse_summary_sections(summary)
            logger.trace("论文总结生成完成")
            return parsed_summary

        except Exception as e:
            logger.trace(f"LLM 总结失败: {e}")
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
                logger.trace("未能解析到详细总结和TL;DR，返回原始内容")
                return summary

        except Exception as e:
            logger.trace(f"解析总结部分失败: {e}")
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
                logger.trace(f"发现已存在的解析结果: {expected_md_path}")
                # 直接读取 Markdown 内容
                with open(expected_md_path, encoding="utf-8") as f:
                    markdown_content = f.read()

                if markdown_content.strip():
                    # Step 3: 截取论文正文
                    main_content = self.extract_main_content(markdown_content)
                    # Step 4: 使用 LLM 生成总结
                    logger.trace(f"summaring {pid}.pdf ...")
                    summary = self.summarize_with_llm(main_content)
                    return summary
                else:
                    logger.trace("已存在的 Markdown 文件内容为空，重新解析")

            # Step 1: 下载论文 PDF
            logger.trace(f"downloading {pid}.pdf ...")
            pdf_path = self.download_arxiv_paper(pid)
            if not pdf_path:
                return "# 错误\n\n无法下载论文 PDF"

            # Step 2: 使用 minerU 解析为 Markdown
            logger.trace(f"parsing {pid}.pdf ...")
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
            logger.trace(f"summaring {pid}.pdf ...")
            summary = self.summarize_with_llm(main_content)

            return summary

        except Exception as e:
            logger.error(f"生成论文总结时发生错误: {e}")
            return f"# 错误\n\n生成总结失败: {str(e)}"


class BatchProcessor:
    """批量處理器類"""

    def __init__(self, max_workers: int = 2):
        """
        初始化批量處理器

        Args:
            max_workers: 最大工作線程數，現在可以真正並發處理
        """
        self.max_workers = max_workers
        self.cache_dir = Path("data/summary")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 統計信息 - 增加詳細的失敗原因統計
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "cached": 0,
            "skipped": 0,
            "low_chinese": 0,
            # 詳細失敗原因統計
            "failure_reasons": {
                "download_failed": 0,  # PDF下載失敗
                "parse_failed": 0,  # minerU解析失敗
                "parse_timeout": 0,  # minerU解析超時
                "empty_content": 0,  # 解析內容為空
                "llm_failed": 0,  # LLM生成失敗
                "cache_failed": 0,  # 緩存失敗
                "other_error": 0,  # 其他未知錯誤
            },
        }
        self.stats_lock = threading.Lock()

        # 失敗詳情記錄 - 記錄每個失敗論文的具體信息
        self.failure_details = {}
        self.failure_lock = threading.Lock()

    def _record_failure_detail(self, pid: str, reason: str, message: str, exception: Exception = None):
        """
        記錄失敗詳情

        Args:
            pid: 論文ID
            reason: 失敗原因類別
            message: 詳細錯誤信息
            exception: 異常對象（可選）
        """
        with self.failure_lock:
            self.failure_details[pid] = {
                "reason": reason,
                "message": message,
                "exception_type": type(exception).__name__ if exception else None,
                "exception_str": str(exception) if exception else None,
                "timestamp": time.time(),
            }

        # 更新統計信息
        with self.stats_lock:
            if reason in self.stats["failure_reasons"]:
                self.stats["failure_reasons"][reason] += 1
            else:
                self.stats["failure_reasons"]["other_error"] += 1

    def get_latest_papers(self, n: int) -> List[Tuple[str, Dict]]:
        """
        獲取資料庫中最新的 n 篇論文

        Args:
            n: 要獲取的論文數量

        Returns:
            List[Tuple[str, Dict]]: 論文 ID 和元數據的列表，按時間倒序排列
        """
        logger.info(f"正在獲取最新的 {n} 篇論文...")

        # 獲取所有論文的元數據
        with get_metas_db() as metas_db:
            metas = {}
            # 使用進度條顯示加載進度
            with tqdm(desc="加載論文數據", unit="篇", leave=False) as pbar:
                for k, v in metas_db.items():
                    metas[k] = v
                    pbar.update(1)

        # 按時間倒序排序（最新的在前）
        logger.info("正在排序論文...")
        sorted_papers = sorted(metas.items(), key=lambda kv: kv[1]["_time"], reverse=True)

        # 取前 n 篇
        latest_papers = sorted_papers[:n]

        logger.info(f"成功獲取 {len(latest_papers)} 篇最新論文")

        return latest_papers

    def is_summary_cached(self, pid: str) -> bool:
        """
        檢查摘要是否已緩存

        Args:
            pid: 論文 ID

        Returns:
            bool: 是否已緩存
        """
        cache_file = self.cache_dir / f"{pid}.md"
        return cache_file.exists() and cache_file.stat().st_size > 0

    def cache_summary(self, pid: str, summary_content: str) -> bool:
        """
        將摘要緩存到 serve.py 使用的目錄

        Args:
            pid: 論文 ID
            summary_content: 摘要內容

        Returns:
            bool: 是否成功緩存
        """
        try:
            cache_file = self.cache_dir / f"{pid}.md"

            # 只緩存成功的摘要（非錯誤信息）
            if not summary_content.startswith("# 錯誤") and not summary_content.startswith("# 错误"):
                # Check Chinese ratio before caching
                chinese_ratio = calculate_chinese_ratio(summary_content)
                logger.trace(f"论文 {pid} 总结中文占比: {chinese_ratio:.2%}")

                if chinese_ratio >= 0.25:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(summary_content)
                    logger.debug(f"摘要已緩存到: {cache_file} (中文占比: {chinese_ratio:.2%})")
                    return True
                else:
                    logger.warning(f"摘要中文占比過低 ({chinese_ratio:.2%} < 50%)，不進行緩存: {pid}")
                    return False
            else:
                logger.warning(f"摘要生成失敗，不進行緩存: {pid}")
                return False

        except Exception as e:
            logger.error(f"緩存摘要失敗 {pid}: {e}")
            return False

    def process_single_paper(self, pid: str, paper_info: Dict, skip_cached: bool = True) -> Tuple[str, bool, str]:
        """
        處理單篇論文，每個線程使用獨立的 BatchPaperSummarizer 實例

        Args:
            pid: 論文 ID
            paper_info: 論文信息
            skip_cached: 是否跳過已緩存的論文

        Returns:
            Tuple[str, bool, str]: (論文ID, 是否成功, 摘要內容或錯誤信息)
        """
        try:
            # 檢查是否已緩存
            if skip_cached and self.is_summary_cached(pid):
                logger.trace(f"跳過已緩存的論文: {pid}")
                with self.stats_lock:
                    self.stats["cached"] += 1
                return pid, True, "已緩存"

            # 獲取論文基本信息
            title = paper_info.get("title", "未知標題")
            authors = ", ".join(a.get("name", "") for a in paper_info.get("authors", []))

            logger.trace(f"開始處理論文: {pid}")
            logger.trace(f"標題: {title}")
            logger.debug(f"作者: {authors}")

            # 為每個線程創建獨立的 BatchPaperSummarizer 實例，傳入processor用於錯誤記錄
            summarizer = BatchPaperSummarizer(processor=self)

            # 調用內建的論文總結器生成摘要
            start_time = time.time()
            summary_content = summarizer.generate_summary(pid)
            end_time = time.time()

            # 檢查是否成功生成摘要
            if summary_content.startswith("# 錯誤") or summary_content.startswith("# 错误"):
                logger.error(f"摘要生成失敗: {pid}")
                with self.stats_lock:
                    self.stats["failed"] += 1
                return pid, False, summary_content

            # 緩存摘要
            cache_success = self.cache_summary(pid, summary_content)

            if cache_success:
                logger.success(f"論文處理成功: {pid} (耗時: {end_time - start_time:.2f}秒)")
                with self.stats_lock:
                    self.stats["success"] += 1
                return pid, True, summary_content
            else:
                logger.error(f"摘要緩存失敗: {pid}")
                with self.stats_lock:
                    # 区分缓存失败的原因
                    chinese_ratio = calculate_chinese_ratio(summary_content)
                    if chinese_ratio < 0.25:
                        self.stats["low_chinese"] += 1
                    else:
                        self.stats["failed"] += 1
                return pid, False, "緩存失敗"

        except Exception as e:
            logger.error(f"處理論文時發生錯誤 {pid}: {e}")
            with self.stats_lock:
                self.stats["failed"] += 1
            return pid, False, str(e)

    def format_time_str(self, timestamp: float) -> str:
        """格式化時間戳"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    def batch_process(
        self, papers: List[Tuple[str, Dict]], skip_cached: bool = True, dry_run: bool = False, max_retries: int = 3
    ) -> Dict:
        """
        批量處理論文，支援失敗重試機制

        Args:
            papers: 論文列表
            skip_cached: 是否跳過已緩存的論文
            dry_run: 是否為乾運行模式
            max_retries: 最大重試次數

        Returns:
            Dict: 處理結果統計
        """
        # 獲取論文詳細信息
        with get_papers_db() as pdb:
            papers_data = {k: v for k, v in pdb.items()}

        self.stats["total"] = len(papers)

        if dry_run:
            logger.info("=== 乾運行模式 - 僅顯示論文信息 ===")

            # 使用進度條顯示論文信息
            with tqdm(papers, desc="檢查論文", unit="篇", leave=True) as pbar:
                for pid, meta in pbar:
                    paper_info = papers_data.get(pid, {})
                    title = paper_info.get("title", "未知標題")
                    authors = ", ".join(a.get("name", "") for a in paper_info.get("authors", []))
                    time_str = self.format_time_str(meta["_time"])
                    cached_status = "已緩存" if self.is_summary_cached(pid) else "未緩存"

                    pbar.set_postfix_str(f"{pid} ({cached_status})")

                    if logger.level("TRACE").no <= logger._core.min_level:
                        logger.trace(f"標題: {title}")
                        logger.trace(f"作者: {authors}")
                        logger.trace(f"時間: {time_str}")
                        logger.trace(f"狀態: {cached_status}")
                        logger.trace("-" * 80)

            return self.stats

        # 實際處理模式
        logger.trace(f"=== 開始批量處理 {len(papers)} 篇論文 ===")
        logger.trace(f"最大工作線程數: {self.max_workers}")
        logger.trace(f"跳過已緩存: {'是' if skip_cached else '否'}")
        logger.trace(f"最大重試次數: {max_retries}")

        start_time = time.time()

        # 初始化處理隊列和重試計數
        processing_queue = [(pid, meta, 0) for pid, meta in papers]  # (pid, meta, retry_count)

        round_num = 1
        while processing_queue:
            current_round_papers = processing_queue
            processing_queue = []

            if round_num > 1:
                logger.info(f"=== 第 {round_num} 輪處理 (重試) ===")
                logger.info(f"重試論文數量: {len(current_round_papers)}")
            else:
                logger.info(f"=== 開始第 {round_num} 輪處理 ===")

            # 創建進度條
            desc = f"第{round_num}輪處理" if round_num > 1 else "處理論文"
            pbar = tqdm(total=len(current_round_papers), desc=desc, unit="篇", leave=True, ncols=120)

            # 初始化当前轮次的计数器
            round_success = 0
            round_failed = 0

            # 使用線程池進行真正的並發處理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任務
                future_to_paper = {}
                for pid, meta, retry_count in current_round_papers:
                    paper_info = papers_data.get(pid, {})
                    future = executor.submit(self.process_single_paper, pid, paper_info, skip_cached)
                    future_to_paper[future] = (pid, meta, retry_count)

                # 處理完成的任務
                for future in as_completed(future_to_paper):
                    pid, meta, retry_count = future_to_paper[future]
                    try:
                        result_pid, success, message = future.result()

                        # 更新計數器和進度條
                        if success:
                            round_success += 1
                            if message == "已緩存":
                                pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | {result_pid} (已緩存)")
                            else:
                                pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | {result_pid}")
                        else:
                            round_failed += 1
                            pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | ✗ {result_pid}")

                        pbar.update(1)

                        if not success and message != "已緩存":
                            # 處理失敗，檢查是否需要重試
                            if retry_count < max_retries:
                                logger.warning(f"處理失敗，將重試 ({retry_count + 1}/{max_retries}): {result_pid}")
                                processing_queue.append((pid, meta, retry_count + 1))
                            else:
                                logger.error(f"處理失敗，已達最大重試次數: {result_pid} - {message}")
                                # 最終失敗統計已在 process_single_paper 中更新

                    except Exception as e:
                        logger.error(f"任務執行異常 {pid}: {e}")
                        # 更新計數器和進度條
                        round_failed += 1
                        pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | ✗ {pid} (異常)")
                        pbar.update(1)

                        # 異常也需要重試
                        if retry_count < max_retries:
                            logger.warning(f"任務異常，將重試 ({retry_count + 1}/{max_retries}): {pid}")
                            processing_queue.append((pid, meta, retry_count + 1))
                        else:
                            logger.error(f"任務異常，已達最大重試次數: {pid}")
                            with self.stats_lock:
                                self.stats["failed"] += 1

            # 關閉進度條
            pbar.close()

            round_num += 1

            # 避免無限重試
            if round_num > max_retries + 1:
                logger.warning("已達最大重試輪數，停止處理")
                break

        end_time = time.time()
        total_time = end_time - start_time

        # 顯示最終統計
        self.print_final_stats(total_time)

        return self.stats

    def print_final_stats(self, total_time: float):
        """打印最終統計信息"""
        logger.success("=" * 60)
        logger.success("處理完成！統計信息:")
        logger.success(f"總論文數量: {self.stats['total']}")
        logger.success(f"成功處理: {self.stats['success']}")
        logger.success(f"處理失敗: {self.stats['failed']}")
        logger.success(f"已緩存跳過: {self.stats['cached']}")
        logger.success(f"中文占比過低: {self.stats['low_chinese']}")
        logger.success(f"總耗時: {total_time:.2f} 秒")

        processed_count = self.stats["success"] + self.stats["failed"]
        if processed_count > 0:
            avg_time = total_time / processed_count
            logger.success(f"平均處理時間: {avg_time:.2f} 秒/篇")

        if self.stats["total"] > 0:
            success_rate = (self.stats["success"] / self.stats["total"]) * 100
            logger.success(f"成功率: {success_rate:.1f}%")

        logger.success(f"摘要已保存到: {self.cache_dir}")
        logger.success("=" * 60)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="批量處理最新論文並生成摘要")
    parser.add_argument("-n", "--num-papers", type=int, default=10, help="要處理的最新論文數量 (默認: 10)")
    parser.add_argument("-w", "--workers", type=int, default=2, help="最大工作線程數 (默認: 2，建議不超過 4)")
    parser.add_argument("--no-skip-cached", action="store_true", help="不跳過已緩存的論文，重新處理所有論文")
    parser.add_argument("--dry-run", action="store_true", help="乾運行模式，只顯示論文信息不進行處理")
    parser.add_argument("-v", "--verbose", action="store_true", help="顯示詳細日誌")
    parser.add_argument("--max-retries", type=int, default=3, help="失敗論文的最大重試次數 (默認: 3)")

    args = parser.parse_args()

    # 設置日誌級別
    logger.remove()
    if args.verbose:
        logger.add(sys.stdout, level="DEBUG", format="\n{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
    else:
        logger.add(sys.stdout, level="INFO", format="\n{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")

    # 參數驗證
    if args.workers > 8:
        logger.warning("工作線程數過多可能導致 GPU 記憶體不足或資源競爭，建議不超過 8")
    elif args.workers < 1:
        logger.error("工作線程數必須至少為 1")
        sys.exit(1)

    if args.num_papers <= 0:
        logger.error("論文數量必須大於 0")
        sys.exit(1)

    if args.max_retries < 0:
        logger.error("重試次數不能為負數")
        sys.exit(1)

    try:
        # 創建批量處理器
        processor = BatchProcessor(max_workers=args.workers)

        # 獲取最新論文
        latest_papers = processor.get_latest_papers(args.num_papers)

        if not latest_papers:
            logger.warning("沒有找到任何論文")
            return

        # 批量處理
        skip_cached = not args.no_skip_cached
        results = processor.batch_process(
            latest_papers, skip_cached=skip_cached, dry_run=args.dry_run, max_retries=args.max_retries
        )

        # 返回適當的退出碼
        if not args.dry_run and results["failed"] > 0:
            logger.warning("部分論文處理失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("用戶中斷程序")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序執行過程中發生錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
