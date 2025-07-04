"""
獲取資料庫中最新的 n 篇論文，從最新的論文開始生成 summary

調用 serve.py 的 API 接口來生成摘要
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

import requests
from loguru import logger

from aslite.db import get_metas_db, get_papers_db

# API調用配置
API_BASE_URL = "http://localhost:55555"  # serve.py默認端口
API_TIMEOUT = 600  # API請求超時時間（秒），摘要生成可能需要較長時間


def get_latest_papers(n: int) -> List[Tuple[str, Dict]]:
    """
    獲取資料庫中最新的 n 篇論文

    Args:
        n: 要獲取的論文數量

    Returns:
        List[Tuple[str, Dict]]: 論文 ID 和元數據的列表，按時間倒序排列
    """
    logger.info(f"Fetching the latest {n} papers...")

    # 獲取所有論文的元數據
    with get_metas_db() as metas_db:
        metas = {k: v for k, v in metas_db.items()}

    # 按時間倒序排序（最新的在前）
    sorted_papers = sorted(metas.items(), key=lambda kv: kv[1]["_time"], reverse=True)

    # 取前 n 篇
    latest_papers = sorted_papers[:n]

    logger.info(f"Successfully fetched {len(latest_papers)} latest papers")

    return latest_papers


def generate_paper_summary_via_api(pid: str) -> Tuple[bool, str]:
    """
    通過 API 調用生成論文摘要

    Args:
        pid: 論文 ID

    Returns:
        Tuple[bool, str]: (是否成功, 摘要內容或錯誤信息)
    """
    try:
        logger.info(f"Calling API to generate paper summary: {pid}")

        # 調用 serve.py 的 API 接口
        response = requests.post(
            f"{API_BASE_URL}/api/get_paper_summary",
            json={"pid": pid},
            timeout=API_TIMEOUT,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                summary_content = result.get("summary_content", "")
                logger.success(f"API call succeeded: {pid}")
                return True, summary_content
            else:
                error_msg = result.get("error", "未知錯誤")
                logger.error(f"API returned error: {error_msg}")
                return False, f"API error: {error_msg}"
        else:
            error_msg = f"HTTP 狀態碼: {response.status_code}"
            logger.error(f"API request failed: {error_msg}")
            return False, f"Request failed: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = f"API request timed out (>{API_TIMEOUT}s)"
        logger.error(error_msg)
        return False, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Unable to connect to API server, please make sure serve.py is running"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"API call exception: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def format_time_str(timestamp: float) -> str:
    """
    格式化時間戳為可讀字符串

    Args:
        timestamp: Unix 時間戳

    Returns:
        str: 格式化的時間字符串
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def check_api_server():
    """
    檢查 API 服務器是否可用

    Returns:
        bool: 服務器是否可用
    """
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            logger.info("API server connection is normal")
            return True
        else:
            logger.warning(f"API server abnormal response: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Unable to connect to API server: {e}")
        logger.error("Please make sure serve.py is running and listening on http://localhost:55555")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="獲取最新論文並通過 API 生成摘要")
    parser.add_argument("-n", "--num-papers", type=int, default=10, help="要處理的最新論文數量 (默認: 10)")
    parser.add_argument("--dry-run", action="store_true", help="只列出論文信息，不生成摘要")
    parser.add_argument("-v", "--verbose", action="store_true", help="顯示詳細日誌")
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:55555", help="API 服務器地址 (默認: http://localhost:55555)"
    )

    args = parser.parse_args()

    # 更新 API 基礎 URL
    global API_BASE_URL
    API_BASE_URL = args.api_url

    # 設置日誌級別
    logger.remove()
    if args.verbose:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")

    logger.info(f"Start processing the latest {args.num_papers} papers")
    logger.info(f"API server address: {API_BASE_URL}")

    # 檢查 API 服務器
    if not args.dry_run and not check_api_server():
        logger.error("API server unavailable, please start serve.py first")
        sys.exit(1)

    try:
        # 獲取最新的論文
        latest_papers = get_latest_papers(args.num_papers)

        if not latest_papers:
            logger.warning("No papers found")
            return

        # 獲取論文詳細信息
        with get_papers_db() as pdb:
            papers_data = {k: v for k, v in pdb.items()}

        # 處理每篇論文
        processed_count = 0
        failed_count = 0

        for i, (pid, meta) in enumerate(latest_papers, 1):
            # 獲取論文詳細信息
            paper_info = papers_data.get(pid, {})
            title = paper_info.get("title", "未知標題")
            authors = ", ".join(a.get("name", "") for a in paper_info.get("authors", []))
            time_str = format_time_str(meta["_time"])

            logger.info(f"\n[{i}/{len(latest_papers)}] Processing paper: {pid}")
            logger.info(f"Title: {title}")
            logger.info(f"Authors: {authors}")
            logger.info(f"Time: {time_str}")

            if args.dry_run:
                logger.info("(Dry run mode - not generating summary)")
                continue

            # 調用 API 生成摘要
            start_time = time.time()
            success, summary_or_error = generate_paper_summary_via_api(pid)
            end_time = time.time()

            if success:
                logger.success(f"Summary generated successfully: {pid} (time: {end_time - start_time:.2f}s)")
                processed_count += 1

                # 顯示摘要預覽（前200字符）
                if args.verbose:
                    preview = (
                        summary_or_error[:200].replace("\n", " ") + "..."
                        if len(summary_or_error) > 200
                        else summary_or_error
                    )
                    logger.debug(f"Summary preview: {preview}")
            else:
                logger.error(f"Summary generation failed: {pid} - {summary_or_error}")
                failed_count += 1

        # 顯示總結
        logger.info("\nProcessing complete!")
        logger.info(f"Total papers: {len(latest_papers)}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")

        if not args.dry_run:
            logger.info("Summaries have been cached to: data/summary/ directory")

    except Exception as e:
        logger.error(f"Error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
