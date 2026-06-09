"""
互动易情绪因子构建 Demo
=========================
目标：从"问答事件流" -> 构建可用于建模的"股票×日期"因子面板
重点演示：稀疏性问题的正确处理方式

依赖：numpy, pandas  (matplotlib 可选)
运行：python interaction_factor_demo.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)


# =====================================================================
# 第 0 步：模拟互动易原始事件流
# =====================================================================
# 关键：刻意制造"稀疏"和"不均匀"——这才是真实情况
#   - 少数热门股提问巨多，大量冷门股几乎没人问
#   - 有的股票某些时间段停牌（真·缺失，区别于"没人问"）
def simulate_events(n_stocks=50, n_days=60):
    stocks = [f"S{idx:03d}" for idx in range(n_stocks)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)  # 交易日

    # 给每只股票一个"基础热度"，呈幂律分布(少数票超热，多数票冷)
    base_heat = np.random.pareto(a=1.5, size=n_stocks) + 0.05
    base_heat = dict(zip(stocks, base_heat))

    # 模拟停牌：随机让某些(股票,日期)停牌 -> 这些日子根本不会产生事件，且属于"数据缺失"
    suspend = {}
    for s in stocks:
        if np.random.rand() < 0.3:  # 30%的股票会有停牌段
            start = np.random.randint(0, n_days - 5)
            length = np.random.randint(3, 8)
            suspend[s] = set(dates[start:start + length])
        else:
            suspend[s] = set()

    users = [f"U{idx:03d}" for idx in range(200)]
    # 指定少数"大V"用户(粉丝多、被回复率高)
    big_v = set(np.random.choice(users, size=10, replace=False))

    topics = ["业绩", "减持", "回购", "重组", "分红", "经营", "诉讼"]
    rows = []
    for d in dates:
        for s in stocks:
            if d in suspend[s]:
                continue  # 停牌不产生事件
            # 当天提问数 ~ 泊松分布，均值由基础热度决定
            n_ask = np.random.poisson(lam=base_heat[s])
            for _ in range(n_ask):
                user = np.random.choice(users)
                topic = np.random.choice(topics)
                # 情绪：减持/诉讼偏负面，回购/分红偏正面
                senti_base = {"减持": -0.5, "诉讼": -0.6, "回购": 0.5,
                              "分红": 0.4, "重组": 0.2}.get(topic, 0.0)
                senti = np.clip(np.random.normal(senti_base, 0.4), -1, 1)
                # 公司是否回复：大V提问更可能被回复
                reply_prob = 0.7 if user in big_v else 0.4
                replied = np.random.rand() < reply_prob
                reply_lag = np.random.randint(0, 3) if replied else np.nan # 回复的平均时滞
                rows.append([d, s, user, topic, senti, replied,
                             reply_lag, user in big_v])

    events = pd.DataFrame(rows, columns=[
        "date", "stock", "user", "topic", "senti_score",
        "replied", "reply_lag", "is_bigv"])
    return events, stocks, dates, suspend

if __name__=="__main__":

    events, ALL_STOCKS, ALL_DATES, SUSPEND = simulate_events()

    print("=" * 70)
    print("【第0步】原始事件流（一条 = 一次提问）")
    print("=" * 70)
    print(events.head(10))
    print(f"\n总事件数: {len(events)}  |  覆盖股票数: {events.stock.nunique()}/{len(ALL_STOCKS)}")


    # =====================================================================
    # 第 1 步：事件流 -> 聚合到 (股票, 日期) 的日频指标;把一条条"提问事件"，按"哪只股票、哪一天"分堆，然后对每一堆算出几个统计数字。
    # =====================================================================
    daily = events.groupby(["stock", "date"]).agg(
        ask_cnt      = ("user", "count"),          # 被提问数
        ask_user_cnt = ("user", "nunique"),        # 被提问用户数(去重)
        reply_cnt    = ("replied", "sum"),         # 回复数
        senti_mean   = ("senti_score", "mean"),    # 平均情绪
        neg_ratio    = ("senti_score", lambda x: (x < -0.2).mean()),  # 负面占比
        bigv_ask_cnt = ("is_bigv", "sum"),         # 大V提问数
        reply_lag    = ("reply_lag", "mean"),      # 平均回复时滞
    ).reset_index()
    daily["reply_rate"] = daily["reply_cnt"] / daily["ask_cnt"]  # 回复率

    print("\n" + "=" * 70)
    print("【第1步】聚合到(股票,日期)的日频指标（注意：这里只有'有提问'的行）")
    print("=" * 70)
    print(daily.head(10))
    print(f"\n聚合后行数: {len(daily)}  （理论满面板应有 {len(ALL_STOCKS)}×{len(ALL_DATES)} = "
          f"{len(ALL_STOCKS)*len(ALL_DATES)} 行）")
    print(f"--> 稀疏度: 仅 {len(daily)/(len(ALL_STOCKS)*len(ALL_DATES)):.1%} 的格子有数据！")


    # =====================================================================
    # 第 2 步：reindex 成完整面板 (全股票 × 全交易日)；把稀疏的 daily（缺一大半行）补成完整面板（每只股票每天都有一行，缺的填 NaN）。
    # =====================================================================
    full_index = pd.MultiIndex.from_product(
        [ALL_STOCKS, ALL_DATES], names=["stock", "date"])# 对stock列和date列算笛卡尔积，制造一个完整的stock-date名单
    panel = daily.set_index(["stock", "date"]).reindex(full_index).reset_index()

    print("\n" + "=" * 70)
    print("【第2步】补齐成完整面板后，大量格子是 NaN（这就是稀疏面板）")
    print("=" * 70)
    print(panel.head(12))
    print(f"\nask_cnt 的缺失比例: {panel['ask_cnt'].isna().mean():.1%}")


    # =====================================================================
    # 第 3 步：区分两种缺失 —— 核心！
    #   情况A: 停牌 -> 真·数据缺失 -> 保持 NaN / 后续剔除
    #   情况B: 没人问 -> 结构性缺失(真实的0) -> ask_cnt 填 0，但情绪保持 NaN
    # =====================================================================
    def is_suspended(row):
        return row["date"] in SUSPEND[row["stock"]]

    panel["suspended"] = panel.apply(is_suspended, axis=1)# 判断每一行是不是"停牌"

    # 制造has_ask 哑变量：当天是否有提问（方案1：稀疏本身是特征）
    panel["has_ask"] = (~panel["ask_cnt"].isna()).astype(int)

    # 计数类指标：没人问 = 真实的0（但停牌的不填，保持NaN）
    count_cols = ["ask_cnt", "ask_user_cnt", "reply_cnt", "bigv_ask_cnt"]
    for c in count_cols:# 计数类指标填0（但停牌的不填）
        panel.loc[~panel["suspended"], c] = panel.loc[~panel["suspended"], c].fillna(0)

    # 情绪/比率类：没人问就没有情绪，绝不填0或均值！保持 NaN
    #   (senti_mean, neg_ratio, reply_rate, reply_lag 保持 NaN)

    print("\n" + "=" * 70)
    print("【第3步】区分两种缺失：计数填0，情绪保持NaN，停牌单独标记")
    print("=" * 70)
    demo = panel[panel["stock"] == "S000"].head(15)
    print(demo[["stock", "date", "suspended", "has_ask", "ask_cnt",
                "senti_mean", "reply_rate"]])
    print("\n要点：")
    print("  - has_ask=0 且 suspended=False  -> 真实'没人问'，ask_cnt=0，情绪=NaN")
    print("  - suspended=True                -> 停牌，所有值=NaN，后续回测应剔除")


    # =====================================================================
    # 第 4 步：滚动窗口 + 指数衰减（方案4：缓解单日跳变）
    # =====================================================================
    panel = panel.sort_values(["stock", "date"])# 按 stock 排序，股票内部再按 date 排序。

    def rolling_sum(g, col, win):# 滚动求和
        return g[col].rolling(win, min_periods=1).sum()

    def ewm_score(g, col, halflife):# 指数加权平均
        # 情绪用指数加权，NaN不参与（min_periods保证至少有一条才出值）
        return g[col].ewm(halflife=halflife, min_periods=1).mean()

    N = 5
    panel["ask_5d"]       = panel.groupby("stock", group_keys=False).apply(
        lambda g: rolling_sum(g, "ask_cnt", N))
    panel["bigv_ask_5d"]  = panel.groupby("stock", group_keys=False).apply(
        lambda g: rolling_sum(g, "bigv_ask_cnt", N))
    panel["senti_ewm"]    = panel.groupby("stock", group_keys=False).apply(
        lambda g: ewm_score(g, "senti_mean", halflife=3))

    print("\n" + "=" * 70)
    print(f"【第4步】滚动{N}日累积 + 情绪指数衰减（让特征更平滑，不因单日没数据骤变）")
    print("=" * 70)
    print(panel[panel["stock"] == "S000"]
          [["stock", "date", "ask_cnt", "ask_5d", "senti_mean", "senti_ewm"]].head(12))


    # =====================================================================
    # 第 5 步：截面标准化 + 行业中性化（方案2）
    #   把绝对值 -> 当日全市场排名分位，解决量纲/可比性，没数据的自然排底部
    # =====================================================================
    # 模拟行业标签，演示行业中性化
    industry_map = {s: f"IND{idx % 5}" for idx, s in enumerate(ALL_STOCKS)}
    panel["industry"] = panel["stock"].map(industry_map)# 造行业标签

    def cross_sectional_rank(df, col):# 截面排名（核心）
        # 当日横截面 rank 分位数（NaN 不参与排名）
        return df.groupby("date")[col].rank(pct=True)

    def industry_neutralize(df, col):# 行业中性化
        # 减去同行业当日均值（行业中性化的简化版：去均值）
        grp = df.groupby(["date", "industry"])[col]
        return df[col] - grp.transform("mean")

    # 因子1：热度因子（提问量分位）—— 排除停牌
    mask = ~panel["suspended"]
    panel.loc[mask, "f_heat"] = cross_sectional_rank(panel[mask], "ask_5d")

    # 因子2：情绪因子（指数衰减情绪的截面分位）
    panel.loc[mask, "f_senti"] = cross_sectional_rank(panel[mask], "senti_ewm")

    # 因子3：大V关注因子
    panel.loc[mask, "f_bigv"] = cross_sectional_rank(panel[mask], "bigv_ask_5d")

    # 因子2再做行业中性化（演示）
    panel.loc[mask, "f_senti_neu"] = industry_neutralize(
        panel.loc[mask], "senti_ewm")

    print("\n" + "=" * 70)
    print("【第5步】截面排名标准化 + 行业中性化 -> 最终因子")
    print("=" * 70)
    print(panel[panel["date"] == ALL_DATES[30]]
          [["stock", "industry", "ask_5d", "f_heat", "senti_ewm",
            "f_senti", "f_bigv"]].head(12))


    # =====================================================================
    # 第 6 步：分池 (方案3) —— 标记哪些股票"该用"情绪因子
    # =====================================================================
    # 统计每只股票在整个区间内"有提问"的天数占比
    coverage = panel.groupby("stock")["has_ask"].mean().rename("coverage")# 算每只股票的覆盖率
    panel = panel.merge(coverage, on="stock")# 把覆盖率合并回面板
    panel["in_senti_pool"] = panel["coverage"] > 0.15  # 覆盖率>15%才纳入情绪池

    pool_stat = panel.groupby("stock").agg(# 生成股票池统计表（方便查看）
        coverage=("coverage", "first"),
        in_pool=("in_senti_pool", "first")).sort_values("coverage", ascending=False)

    print("\n" + "=" * 70)
    print("【第6步】分池：情绪因子只在'有足够覆盖'的股票上有效")
    print("=" * 70)
    print("覆盖率最高的5只股票（热门股，情绪因子可用）：")
    print(pool_stat.head(5))
    print("\n覆盖率最低的5只股票（冷门股，应剔除或改用量价/基本面因子）：")
    print(pool_stat.tail(5))
    print(f"\n情绪池内股票数: {pool_stat['in_pool'].sum()} / {len(ALL_STOCKS)}")


    # =====================================================================
    # 第 7 步：最终产出 + 缺失情况总览
    # =====================================================================
    final = panel[["stock", "date", "industry", "in_senti_pool",
                   "has_ask", "f_heat", "f_senti", "f_bigv", "f_senti_neu"]]

    print("\n" + "=" * 70)
    print("【最终因子面板】每行 = 一个(股票,日期)样本，可直接喂给模型/Qlib")
    print("=" * 70)
    print(final.head(15))

    print("\n各因子缺失率（情绪类因子天然有缺失，这是正常的）：")
    for c in ["f_heat", "f_senti", "f_bigv", "f_senti_neu"]:
        print(f"  {c:12s}: 缺失 {final[c].isna().mean():.1%}")

    # 保存
    final.to_csv("outputs/interaction_factors.csv", index=False)
    print("\n已保存 -> interaction_factors.csv")


    # =====================================================================
    # (可选) 第 8 步：可视化稀疏性
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
            "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        pivot = panel.pivot(index="stock", columns="date", values="ask_cnt")
        pivot = pivot.loc[pool_stat.index]  # 按热度排序

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot.fillna(-1).values, aspect="auto", cmap="viridis")
        ax.set_title("互动易提问数热力图（深蓝=停牌NaN, 0=没人问, 亮=热门）\n"
                     "上方=热门股(数据密), 下方=冷门股(数据稀疏)")
        ax.set_xlabel("交易日"); ax.set_ylabel("股票(按热度排序)")
        plt.colorbar(im, ax=ax, label="提问数")
        plt.tight_layout()
        plt.savefig("outputs/sparsity_heatmap.png", dpi=120)
        print("已保存稀疏性热力图 -> sparsity_heatmap.png")
    except ImportError:
        print("\n(未安装 matplotlib，跳过画图。pip install matplotlib 可生成热力图)")