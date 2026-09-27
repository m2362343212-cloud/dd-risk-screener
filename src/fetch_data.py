"""
第 1 步：从 AKShare 按"报告期"批量下载 A 股年报数据，每年每张表存一个 parquet 文件。

用法：
    python src/fetch_data.py --check      # 先只下载 2023 年，打印每张表的列名（第一次一定先跑这个）
    python src/fetch_data.py              # 下载 FY2014–FY2025 全部数据（中途断了可以重跑，已下载的会跳过）
"""
import argparse, pathlib, time

import akshare as ak
import requests

_orig_request = requests.Session.request
def _request_with_timeout(self, *args, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = 30
    return _orig_request(self, *args, **kwargs)
requests.Session.request = _request_with_timeout

OUT = pathlib.Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

# 四个"按报告期"的批量接口：一次调用返回当年所有披露了报告的公司
FUNCS = {
    "yjbb": ak.stock_yjbb_em,   # 业绩报表：营收、净利润、ROE、毛利率、所处行业
    "zcfz": ak.stock_zcfz_em,   # 资产负债表：货币资金、应收、存货、总资产、总负债
    "lrb":  ak.stock_lrb_em,    # 利润表：营业总收入、营业利润、销售/管理/财务费用
    "xjll": ak.stock_xjll_em,   # 现金流量表：经营/投资/筹资现金流
}


def fetch(name, fn, year, retries=3):
    """下载一张表；网络失败时等一会儿重试。"""
    for attempt in range(1, retries + 1):
        try:
            return fn(date=f"{year}1231")          # 1231 = 年报
        except Exception as e:
            print(f"  {name} {year} 第 {attempt} 次失败：{e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"{name} {year} 连续失败，稍后重跑本脚本即可（已下载的会跳过）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只下载 2023 年并打印列名")
    args = ap.parse_args()

    if args.check:
        for name, fn in FUNCS.items():
            df = fn(date="20231231")
            print(f"\n===== {name}  形状 {df.shape} =====")
            print(list(df.columns))
            print(df.head(2).T)
        return

    for year in range(2014, 2026):                 # FY2014 … FY2025
        for name, fn in FUNCS.items():
            path = OUT / f"{name}_{year}.parquet"
            if path.exists():
                continue                            # 已下载过就跳过
            df = fetch(name, fn, year)
            df["fiscal_year"] = year
            df.to_parquet(path)
            print(name, year, df.shape)
            time.sleep(2)                           # 对服务器客气一点


if __name__ == "__main__":
    main()
