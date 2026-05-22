#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
分层多Agent量化投研系统 - 最小可运行Demo
================================================================================
基于方案: R&D-Agent-Quant + AlphaCrafter + TradingAgents 融合架构

核心特性:
    - 纯Python + numpy/pandas, 无需外部LLM/Agent库
    - 自包含回测引擎 (Sharpe/MaxDrawdown/IC计算)
    - 模拟LLM因子挖掘 (可替换为真实LLM API)
    - 完整的多Agent协作 Pipeline
    - Human-in-the-Loop 关键决策点

运行方式:
    python quant_agent_demo.py

架构:
    DataAgent (MCP数据层) 
        → FactorMiningAgent (因子研发层) 
        → StrategyAgent (策略构建层) 
        → BacktestEngine (回测层) 
        → RiskControlAgent (风控层) 
        → Human-in-the-Loop (人工审核)
================================================================================
"""

import numpy as np
import pandas as pd
import random
import json
import re
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==================== 数据类定义 ====================

@dataclass
class StockData:
    """股票数据容器"""
    code: str
    prices: pd.Series      # 收盘价
    volumes: pd.Series     # 成交量
    dates: pd.DatetimeIndex
    
    def returns(self) -> pd.Series:
        return self.prices.pct_change().dropna()

@dataclass  
class ResearchReport:
    """研报数据容器"""
    title: str
    content: str
    stock_codes: List[str]
    sentiment: str      # 'bullish' | 'bearish' | 'neutral'
    publish_date: datetime

@dataclass
class Factor:
    """因子定义"""
    name: str
    description: str
    expression: str          # Python表达式字符串
    values: pd.Series        # 因子值 (time-series)
    ic: float = 0.0          # Information Coefficient
    source: str = ''         # 'llm' | 'human' | 'evolution'

@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series
    trades: List[Dict]
    factor_name: str
    
@dataclass
class RiskAssessment:
    """风险评估"""
    passed: bool
    issues: List[str]
    warnings: List[str]
    pbo_estimate: float       # Probability of Backtest Overfitting (估计)
    recommendation: str


# ==================== 模拟LLM层 ====================

class MockLLM:
    """
    模拟大语言模型 - 无需API Key即可运行
    
    在真实场景中，替换为:
        from openai import OpenAI
        client = OpenAI(api_key="YOUR_KEY")
        response = client.chat.completions.create(...)
    """
    
    # 预设的因子模板库 (模拟LLM的"知识")
    FACTOR_TEMPLATES = [
        {
            'name': 'momentum_5d',
            'description': '5日价格动量因子: 过去5日收益率',
            'expr': 'prices.pct_change(5)',
            'logic': '价格上涨趋势延续'
        },
        {
            'name': 'momentum_20d',
            'description': '20日价格动量因子: 过去20日收益率', 
            'expr': 'prices.pct_change(20)',
            'logic': '中期趋势跟踪'
        },
        {
            'name': 'mean_reversion',
            'description': '均值回归因子: 价格偏离20日均线的程度',
            'expr': '(prices - prices.rolling(20).mean()) / prices.rolling(20).std()',
            'logic': '价格过度偏离会回归均值'
        },
        {
            'name': 'volatility_ratio',
            'description': '波动率比率因子: 短期波动/长期波动',
            'expr': 'returns.rolling(5).std() / returns.rolling(20).std()',
            'logic': '波动率突变预示趋势变化'
        },
        {
            'name': 'volume_price_divergence',
            'description': '量价背离因子: 价格与成交量的相关性偏离',
            'expr': 'returns.rolling(10).corr(volumes)',
            'logic': '量价背离预示趋势反转'
        },
        {
            'name': 'rsi_proxy',
            'description': 'RSI代理因子: 基于涨跌幅的动量震荡指标',
            'expr': 'returns.rolling(14).apply(lambda x: x[x>0].sum() / abs(x).sum() * 100 if abs(x).sum()>0 else 50)',
            'logic': '超买超卖信号'
        },
        {
            'name': 'acceleration',
            'description': '价格加速度因子: 动量的变化率',
            'expr': 'prices.pct_change(5) - prices.pct_change(10)',
            'logic': '动量加速或减速信号'
        },
        {
            'name': 'volume_surge',
            'description': '成交量突增因子: 成交量相对均值的倍数',
            'expr': 'volumes / volumes.rolling(20).mean()',
            'logic': '成交量突增往往伴随重要信息'
        },
        {
            'name': 'report_sentiment_boost',
            'description': '研报情绪增强因子: 基于研报情绪的信号增强',
            'expr': 'np.where(sentiment_score > 0.5, 1, np.where(sentiment_score < -0.5, -1, 0))',
            'logic': '研报情绪极端时提供方向性信号'
        },
    ]
    
    # 研报情绪分析模板
    SENTIMENT_PATTERNS = {
        'bullish': ['买入', '增持', '推荐', '看好', '上涨', '目标价', '超预期', '业绩增长'],
        'bearish': ['卖出', '减持', '回避', '看空', '下跌', '风险', '业绩下滑', '不及预期'],
    }
    
    def __init__(self, creativity: float = 0.3):
        self.creativity = creativity  # 因子创新程度
        self.call_count = 0
        
    def _log(self, task: str, input_len: int):
        """记录LLM调用"""
        self.call_count += 1
        print(f"  [MockLLM #{self.call_count}] 🧠 {task} (input: {input_len} chars)")
        
    def generate_factor(self, stock_data: StockData, reports: List[ResearchReport], 
                       existing_factors: List[str]) -> Dict:
        """模拟LLM生成新因子创意"""
        self._log("因子创意生成", len(stock_data.code) + sum(len(r.content) for r in reports))
        time.sleep(0.1)  # 模拟推理延迟
        
        # 过滤已有因子
        available = [t for t in self.FACTOR_TEMPLATES if t['name'] not in existing_factors]
        if not available:
            available = self.FACTOR_TEMPLATES
            
        # 基于研报情绪选择更匹配的因子
        if reports:
            latest_report = reports[-1]
            if latest_report.sentiment == 'bullish':
                candidates = [f for f in available if 'momentum' in f['name'] or 'surge' in f['name']]
            elif latest_report.sentiment == 'bearish':
                candidates = [f for f in available if 'mean_reversion' in f['name'] or 'rsi' in f['name']]
            else:
                candidates = available
        else:
            candidates = available
            
        if not candidates:
            candidates = available
            
        chosen = random.choice(candidates)
        
        # 模拟LLM的"解释"
        explanation = (
            f"基于对{stock_data.code}的历史价格走势和最新研报分析，"
            f"我认为'{chosen['description']}'是一个有效的Alpha因子。"
            f"核心逻辑是: {chosen['logic']}。"
        )
        
        return {
            'name': chosen['name'],
            'description': chosen['description'],
            'expression': chosen['expr'],
            'explanation': explanation,
            'confidence': random.uniform(0.6, 0.95)
        }
    
    def analyze_report_sentiment(self, report: ResearchReport) -> Dict:
        """模拟LLM研报情绪分析"""
        self._log("研报情绪分析", len(report.content))
        time.sleep(0.05)
        
        content = report.content
        bullish_score = sum(1 for w in self.SENTIMENT_PATTERNS['bullish'] if w in content)
        bearish_score = sum(1 for w in self.SENTIMENT_PATTERNS['bearish'] if w in content)
        
        if bullish_score > bearish_score:
            sentiment, score = 'bullish', min(bullish_score * 0.2, 1.0)
        elif bearish_score > bullish_score:
            sentiment, score = 'bearish', -min(bearish_score * 0.2, 1.0)
        else:
            sentiment, score = 'neutral', 0.0
            
        return {
            'sentiment': sentiment,
            'score': score,
            'summary': f"研报《{report.title}》情绪: {sentiment}, 得分: {score:.2f}"
        }
    
    def evaluate_strategy(self, backtest_result: BacktestResult) -> Dict:
        """模拟LLM策略评估和诊断"""
        self._log("策略评估与诊断", 100)
        time.sleep(0.05)
        
        issues = []
        if backtest_result.sharpe_ratio < 0.5:
            issues.append("Sharpe比率过低，风险调整收益不佳")
        if backtest_result.max_drawdown > -0.3:
            issues.append("最大回撤过大，需加强风控")
        if backtest_result.win_rate < 0.45:
            issues.append("胜率不足45%，信号质量需提升")
            
        if not issues:
            diagnosis = "策略表现良好，建议保持当前参数"
        else:
            diagnosis = f"发现{len(issues)}个问题: " + "; ".join(issues)
            
        suggestions = []
        if 'sharpe' in diagnosis.lower() or backtest_result.sharpe_ratio < 1.0:
            suggestions.append("尝试增加因子数量或调整参数")
        if 'drawdown' in diagnosis.lower() or backtest_result.max_drawdown > -0.2:
            suggestions.append("加入止损逻辑或降低仓位")
        if not suggestions:
            suggestions.append("微调参数寻找更优解")
            
        return {
            'diagnosis': diagnosis,
            'suggestions': suggestions,
            'should_continue': len(issues) <= 1  # 最多容忍1个问题
        }


# ==================== MCP数据层 ====================

class MCPDataServer:
    """
    模拟MCP (Model Context Protocol) 数据服务器
    在真实场景中，这些会是对接Wind/同花顺/Tushare的API接口
    """
    
    def __init__(self, seed: int = SEED):
        self.seed = seed
        np.random.seed(seed)
        
    def get_stock_price(self, code: str, start_date: str, end_date: str, 
                       freq: str = 'D') -> pd.Series:
        """MCP: 获取股票价格数据"""
        print(f"  [MCP:行情数据] 📈 获取 {code} {start_date}~{end_date}")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
        
        # 生成合成但合理的股价数据 (随机游走 + 趋势 + 波动)
        n = len(dates)
        trend = np.linspace(10, 15, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 2
        noise = np.cumsum(np.random.randn(n) * 0.3)
        prices = pd.Series(trend + noise + 10, index=dates, name='close')
        prices = prices.round(2)
        return prices
    
    def get_volume(self, code: str, dates: pd.DatetimeIndex) -> pd.Series:
        """MCP: 获取成交量数据"""
        print(f"  [MCP:成交量] 📊 获取 {code} 成交量")
        base_vol = 1000000
        volumes = base_vol * (1 + np.random.randn(len(dates)) * 0.5)
        volumes = np.abs(volumes).astype(int)
        return pd.Series(volumes, index=dates, name='volume')
    
    def get_research_reports(self, code: str, n: int = 3) -> List[ResearchReport]:
        """MCP: 获取研报数据"""
        print(f"  [MCP:研报数据] 📄 获取 {code} {n}篇研报")
        
        templates = [
            ("{code}深度报告: 业绩稳健增长，维持买入评级", 
             "公司{code}发布最新财报，业绩表现超出市场预期。主营业务保持稳健增长态势，"
             "新产品线有望在未来季度贡献显著增量。考虑到行业景气度回升和公司竞争优势，"
             "我们维持买入评级，目标价上调15%。",
             'bullish'),
            ("{code}业绩点评: 短期承压但长期看好", 
             "受行业季节性因素影响，{code}短期业绩略有承压，营收增速环比放缓。"
             "但我们认为公司核心竞争力未变，随着新产品放量，下半年有望迎来业绩拐点。"
             "建议逢低布局，给予增持评级。",
             'bullish'),
            ("{code}风险提示: 关注行业竞争加剧", 
             "近期行业竞争格局发生变化，{code}面临一定的市场份额压力。"
             "虽然公司基本面依然稳健，但我们建议密切关注后续季度数据变化，"
             "暂时给予中性评级，等待更明确的信号。",
             'neutral'),
        ]
        
        reports = []
        base_date = datetime(2024, 1, 1)
        for i in range(min(n, len(templates))):
            title, content, sentiment = templates[i]
            reports.append(ResearchReport(
                title=title.format(code=code),
                content=content.format(code=code),
                stock_codes=[code],
                sentiment=sentiment,
                publish_date=base_date + timedelta(days=i*30)
            ))
        return reports


# ==================== Agent定义 ====================

class DataAgent:
    """
    数据Agent - 负责多源数据的获取和预处理
    对应架构: Data Layer (MCP数据接入层)
    """
    
    def __init__(self, mcp: MCPDataServer, llm: MockLLM):
        self.mcp = mcp
        self.llm = llm
        self.name = "DataAgent"
        
    def run(self, stock_code: str, start_date: str, end_date: str) -> Dict:
        """执行数据获取Pipeline"""
        print(f"\n{'='*50}")
        print(f"🤖 [{self.name}] 开始数据采集")
        print(f"{'='*50}")
        
        # 1. 获取行情数据
        prices = self.mcp.get_stock_price(stock_code, start_date, end_date)
        volumes = self.mcp.get_volume(stock_code, prices.index)
        
        stock_data = StockData(
            code=stock_code,
            prices=prices,
            volumes=volumes,
            dates=prices.index
        )
        
        # 2. 获取研报数据
        reports = self.mcp.get_research_reports(stock_code, n=3)
        
        # 3. LLM分析研报情绪
        sentiment_results = []
        for report in reports:
            result = self.llm.analyze_report_sentiment(report)
            sentiment_results.append(result)
            print(f"    📊 {result['summary']}")
        
        # 4. 构建情绪得分时序
        sentiment_scores = self._build_sentiment_series(
            stock_data.dates, reports, sentiment_results
        )
        
        print(f"✅ [{self.name}] 数据采集完成: {len(prices)}条价格, {len(reports)}篇研报")
        
        return {
            'stock_data': stock_data,
            'reports': reports,
            'sentiment_results': sentiment_results,
            'sentiment_scores': sentiment_scores
        }
    
    def _build_sentiment_series(self, dates: pd.DatetimeIndex, 
                                 reports: List[ResearchReport],
                                 sentiment_results: List[Dict]) -> pd.Series:
        """构建情绪得分数组 (与价格对齐)"""
        scores = pd.Series(0.0, index=dates)
        for report, result in zip(reports, sentiment_results):
            mask = dates >= report.publish_date
            scores.loc[mask] = result['score']
        return scores


class FactorMiningAgent:
    """
    因子挖掘Agent - 负责生成和评估Alpha因子
    对应架构: Factor Layer (因子研发层), 基于R&D-Agent-Quant
    """
    
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.name = "FactorMiningAgent"
        self.factor_library: List[Factor] = []
        
    def run(self, stock_data: StockData, sentiment_scores: pd.Series,
            max_factors: int = 5) -> List[Factor]:
        """执行因子挖掘Pipeline"""
        print(f"\n{'='*50}")
        print(f"🤖 [{self.name}] 开始因子挖掘 (目标: {max_factors}个)")
        print(f"{'='*50}")
        
        factors = []
        existing_names = [f.name for f in self.factor_library]
        
        for i in range(max_factors):
            print(f"\n  🔄 因子生成 #{i+1}/{max_factors}")
            
            # 1. LLM生成因子创意
            factor_idea = self.llm.generate_factor(
                stock_data, [], existing_names + [f.name for f in factors]
            )
            print(f"    💡 创意: {factor_idea['description']}")
            print(f"    📝 解释: {factor_idea['explanation'][:80]}...")
            
            # 2. 将因子表达式转换为可执行代码
            factor_values = self._execute_factor_expression(
                factor_idea['expression'],
                stock_data,
                sentiment_scores
            )
            
            if factor_values is None or factor_values.isna().all():
                print(f"    ❌ 因子执行失败，跳过")
                continue
                
            # 3. 计算IC (Information Coefficient)
            ic = self._calculate_ic(factor_values, stock_data.returns())
            
            factor = Factor(
                name=factor_idea['name'],
                description=factor_idea['description'],
                expression=factor_idea['expression'],
                values=factor_values.dropna(),
                ic=ic,
                source='llm'
            )
            
            print(f"    📊 IC: {ic:.4f} {'✅ 有效' if abs(ic) > 0.03 else '⚠️ 较弱'}")
            factors.append(factor)
            
        print(f"\n✅ [{self.name}] 因子挖掘完成: {len(factors)}个有效因子")
        self.factor_library.extend(factors)
        return factors
    
    def _execute_factor_expression(self, expr: str, stock_data: StockData,
                                    sentiment_scores: pd.Series) -> Optional[pd.Series]:
        """安全执行因子表达式"""
        try:
            prices = stock_data.prices
            volumes = stock_data.volumes
            returns = stock_data.returns()
            sentiment_score = sentiment_scores.reindex(prices.index).fillna(0)
            
            # 构建安全的局部命名空间
            local_ns = {
                'prices': prices,
                'volumes': volumes,
                'returns': returns,
                'sentiment_score': sentiment_score,
                'np': np,
                'pd': pd,
                'abs': abs,
            }
            
            result = eval(expr, {"__builtins__": {}}, local_ns)
            
            if isinstance(result, pd.Series):
                return result.reindex(prices.index)
            elif isinstance(result, np.ndarray):
                return pd.Series(result, index=prices.index)
            else:
                return pd.Series(result, index=prices.index)
                
        except Exception as e:
            print(f"    ⚠️ 表达式执行错误: {e}")
            return None
    
    def _calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算Rank Information Coefficient"""
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0
        # Spearman rank correlation
        ic = aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank())
        return ic if not np.isnan(ic) else 0.0


class StrategyAgent:
    """
    策略Agent - 基于因子构建交易信号和策略
    对应架构: Strategy Layer (策略决策层)
    """
    
    def __init__(self):
        self.name = "StrategyAgent"
        
    def run(self, stock_data: StockData, factors: List[Factor],
            top_n: int = 3) -> Dict:
        """构建交易策略"""
        print(f"\n{'='*50}")
        print(f"🤖 [{self.name}] 开始策略构建")
        print(f"{'='*50}")
        
        # 1. 因子筛选 (选择IC最高的top_n个因子)
        sorted_factors = sorted(factors, key=lambda f: abs(f.ic), reverse=True)
        selected = sorted_factors[:top_n]
        
        print(f"  📋 从{len(factors)}个因子中筛选top-{top_n}:")
        for f in selected:
            print(f"    • {f.name}: IC={f.ic:.4f} | {f.description}")
        
        # 2. 合成信号 (等权加权，考虑IC方向)
        signal = self._composite_signal(selected, stock_data)
        
        # 3. 生成交易规则
        rules = self._generate_trading_rules(selected)
        
        print(f"\n  📐 交易规则:")
        for rule in rules:
            print(f"    • {rule}")
            
        print(f"✅ [{self.name}] 策略构建完成")
        
        return {
            'selected_factors': selected,
            'composite_signal': signal,
            'trading_rules': rules
        }
    
    def _composite_signal(self, factors: List[Factor], 
                          stock_data: StockData) -> pd.Series:
        """合成多因子信号 (IC加权)"""
        signal = pd.Series(0.0, index=stock_data.prices.index)
        total_ic = sum(abs(f.ic) for f in factors)
        
        for factor in factors:
            weight = abs(factor.ic) / total_ic if total_ic > 0 else 1.0 / len(factors)
            aligned = factor.values.reindex(signal.index)
            # 标准化
            aligned = (aligned - aligned.mean()) / (aligned.std() + 1e-10)
            signal += aligned * weight * np.sign(factor.ic)
            
        return signal
    
    def _generate_trading_rules(self, factors: List[Factor]) -> List[str]:
        """生成可读的 trading rules"""
        rules = []
        for f in factors:
            direction = "正向" if f.ic > 0 else "反向"
            rules.append(f"{f.name}: {direction}信号 | IC={f.ic:.4f}")
        rules.append("综合信号 > 0.5 → 买入; 综合信号 < -0.5 → 卖出")
        rules.append("持仓最多5个交易日，到期强制平仓")
        return rules


class BacktestEngine:
    """
    回测引擎 - 执行策略回测并计算性能指标
    对应架构: Backtest Layer (回测层)
    """
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.name = "BacktestEngine"
        
    def run(self, stock_data: StockData, signal: pd.Series,
            strategy_factors: List[Factor]) -> BacktestResult:
        """执行回测"""
        print(f"\n{'='*50}")
        print(f"🤖 [{self.name}] 开始回测 (初始资金: ¥{self.initial_capital:,.0f})")
        print(f"{'='*50}")
        
        prices = stock_data.prices
        returns = stock_data.returns()
        
        # 交易参数
        threshold = 0.5       # 信号阈值
        max_holding = 5       # 最大持仓天数
        position_size = 0.95  # 仓位比例
        
        capital = self.initial_capital
        position = 0          # 0: 空仓, 1: 多头
        entry_price = 0
        entry_day = 0
        hold_days = 0
        
        equity = [capital]
        trades = []
        trade_count = 0
        win_count = 0
        
        for i in range(1, len(prices)):
            date = prices.index[i]
            price = prices.iloc[i]
            sig = signal.iloc[i] if i < len(signal) else 0
            
            # 有持仓：检查平仓条件
            if position == 1:
                hold_days += 1
                pnl_pct = (price - entry_price) / entry_price
                
                should_exit = (
                    sig < -threshold or           # 信号反转
                    hold_days >= max_holding or   # 到期平仓
                    pnl_pct <= -0.05              # 止损5%
                )
                
                if should_exit:
                    pnl = capital * position_size * pnl_pct
                    capital += pnl
                    trade_count += 1
                    if pnl > 0:
                        win_count += 1
                    trades.append({
                        'type': 'SELL',
                        'date': date,
                        'price': price,
                        'pnl': pnl,
                        'hold_days': hold_days,
                        'reason': 'signal_reverse' if sig < -threshold else 
                                  'max_hold' if hold_days >= max_holding else 'stop_loss'
                    })
                    position = 0
                    hold_days = 0
                    
            # 无持仓：检查开仓条件
            elif position == 0:
                if sig > threshold:
                    position = 1
                    entry_price = price
                    entry_day = i
                    hold_days = 0
                    trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': price,
                        'pnl': 0,
                        'hold_days': 0,
                        'reason': 'signal'
                    })
            
            # 记录权益 (含未实现盈亏)
            unrealized = 0
            if position == 1:
                unrealized = capital * position_size * ((price - entry_price) / entry_price)
            equity.append(capital + unrealized)
        
        # 强制平仓最后一天
        if position == 1:
            price = prices.iloc[-1]
            pnl = capital * position_size * ((price - entry_price) / entry_price)
            capital += pnl
            trade_count += 1
            if pnl > 0:
                win_count += 1
        
        equity_series = pd.Series(equity, index=prices.index[:len(equity)])
        
        # 计算性能指标
        total_return = (capital - self.initial_capital) / self.initial_capital
        n_years = len(prices) / 252
        annual_return = (capital / self.initial_capital) ** (1 / max(n_years, 0.01)) - 1
        
        equity_returns = equity_series.pct_change().dropna()
        sharpe = (equity_returns.mean() / (equity_returns.std() + 1e-10)) * np.sqrt(252)
        
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = drawdown.min()
        
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = win_count / trade_count if trade_count > 0 else 0
        
        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            num_trades=trade_count,
            equity_curve=equity_series,
            trades=trades,
            factor_name='+'.join([f.name for f in strategy_factors[:2]])
        )
        
        self._print_result(result)
        return result
    
    def _print_result(self, r: BacktestResult):
        print(f"\n  📊 回测结果:")
        print(f"    总收益率:    {r.total_return:>+8.2%}")
        print(f"    年化收益率:  {r.annual_return:>+8.2%}")
        print(f"    Sharpe比率:  {r.sharpe_ratio:>8.2f}")
        print(f"    最大回撤:    {r.max_drawdown:>8.2%}")
        print(f"    Calmar比率:  {r.calmar_ratio:>8.2f}")
        print(f"    胜率:        {r.win_rate:>8.1%}")
        print(f"    交易次数:    {r.num_trades:>8d}")
        print(f"✅ [{self.name}] 回测完成")


class RiskControlAgent:
    """
    风控Agent - 评估策略风险，检测过拟合
    对应架构: Risk Layer (风控层)
    """
    
    def __init__(self):
        self.name = "RiskControlAgent"
        
    def run(self, backtest_result: BacktestResult, llm: MockLLM) -> RiskAssessment:
        """执行风险评估"""
        print(f"\n{'='*50}")
        print(f"🤖 [{self.name}] 开始风险评估")
        print(f"{'='*50}")
        
        issues = []
        warnings_list = []
        
        # 1. Sharpe检查
        if backtest_result.sharpe_ratio < 0.3:
            issues.append(f"Sharpe={backtest_result.sharpe_ratio:.2f} < 0.3，策略无效")
        elif backtest_result.sharpe_ratio < 1.0:
            warnings_list.append(f"Sharpe={backtest_result.sharpe_ratio:.2f} < 1.0，风险调整后收益一般")
        else:
            print(f"  ✅ Sharpe={backtest_result.sharpe_ratio:.2f} 表现良好")
            
        # 2. 最大回撤检查
        if backtest_result.max_drawdown < -0.3:
            issues.append(f"最大回撤={backtest_result.max_drawdown:.2%}，超出-30%红线")
        elif backtest_result.max_drawdown < -0.15:
            warnings_list.append(f"最大回撤={backtest_result.max_drawdown:.2%}，需关注")
        else:
            print(f"  ✅ 最大回撤={backtest_result.max_drawdown:.2%} 在可控范围")
            
        # 3. 胜率检查
        if backtest_result.win_rate < 0.4:
            issues.append(f"胜率={backtest_result.win_rate:.1%}，信号质量过低")
        elif backtest_result.win_rate < 0.45:
            warnings_list.append(f"胜率={backtest_result.win_rate:.1%}，略低于理想水平")
            
        # 4. 交易频率检查
        if backtest_result.num_trades < 5:
            warnings_list.append(f"交易次数仅{backtest_result.num_trades}次，统计意义有限")
            
        # 5. PBO估计 (简化版)
        pbo = self._estimate_pbo(backtest_result)
        print(f"  📉 PBO(过拟合概率)估计: {pbo:.2%}")
        if pbo > 0.5:
            issues.append(f"PBO={pbo:.2%} > 50%，严重过拟合风险")
        elif pbo > 0.2:
            warnings_list.append(f"PBO={pbo:.2%} > 20%，存在过拟合可能")
        else:
            print(f"  ✅ PBO={pbo:.2%} 过拟合风险较低")
        
        # 6. LLM智能诊断
        llm_eval = llm.evaluate_strategy(backtest_result)
        print(f"  🧠 LLM诊断: {llm_eval['diagnosis']}")
        if llm_eval['suggestions']:
            print(f"  💡 建议: {'; '.join(llm_eval['suggestions'])}")
        
        passed = len(issues) == 0 and pbo < 0.5
        
        if passed:
            recommendation = "策略通过风控检查，建议人工审核后推进"
        else:
            recommendation = f"发现{len(issues)}个严重问题，{'; '.join(issues)}。建议返回调整。"
            
        print(f"\n  {'🟢 通过' if passed else '🔴 未通过'} | {recommendation}")
        
        return RiskAssessment(
            passed=passed,
            issues=issues,
            warnings=warnings_list,
            pbo_estimate=pbo,
            recommendation=recommendation
        )
    
    def _estimate_pbo(self, result: BacktestResult) -> float:
        """简化版PBO估计 (实际应使用CSCV方法)"""
        # 基于交易次数、sharpe和回撤的简单启发式估计
        if result.num_trades < 5:
            return 0.8  # 交易太少，高度可疑
        
        # 简单启发：sharpe太高或回撤太小可能过拟合
        score = 0.0
        if result.sharpe_ratio > 3.0:
            score += 0.3
        if result.win_rate > 0.8:
            score += 0.2
        if result.max_drawdown > -0.02 and result.num_trades > 10:
            score += 0.3  # 几乎无回撤但很活跃，可疑
        
        # 随机扰动
        score += random.uniform(-0.1, 0.15)
        return max(0.0, min(0.95, score))


class HumanInTheLoop:
    """
    人工审核节点 - 关键决策点的人类介入
    对应架构: Human-in-the-Loop 控制台
    """
    
    def __init__(self, auto_mode: bool = True):
        """
        auto_mode=True: 自动模拟人工决策 (用于Demo)
        auto_mode=False: 真实人工交互
        """
        self.auto_mode = auto_mode
        self.name = "HumanInTheLoop"
        
    def review_factors(self, factors: List[Factor]) -> bool:
        """审核因子列表"""
        print(f"\n{'='*50}")
        print(f"👤 [{self.name}] 人工审核: 因子列表")
        print(f"{'='*50}")
        
        for f in factors:
            status = "✅ 有效" if abs(f.ic) > 0.03 else "⚠️ 关注"
            print(f"  {status} {f.name}: IC={f.ic:.4f} | {f.description}")
            
        if self.auto_mode:
            # 自动通过 (模拟: IC有效的因子都通过)
            approved = all(abs(f.ic) > 0.01 for f in factors)
            print(f"\n  {'🟢 审核通过' if approved else '🔴 审核拒绝'} (Auto Mode)")
            return approved
        else:
            resp = input("  是否批准这些因子? (y/n): ").strip().lower()
            return resp == 'y'
    
    def review_backtest(self, result: BacktestResult, risk: RiskAssessment) -> str:
        """审核回测结果，决定下一步行动"""
        print(f"\n{'='*50}")
        print(f"👤 [{self.name}] 人工审核: 回测结果")
        print(f"{'='*50}")
        
        print(f"  策略: {result.factor_name}")
        print(f"  总收益: {result.total_return:+.2%} | Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  风控: {'通过' if risk.passed else '未通过'} | PBO: {risk.pbo_estimate:.2%}")
        print(f"  LLM建议: {risk.recommendation}")
        
        if self.auto_mode:
            # 自动决策逻辑
            if risk.passed and result.sharpe_ratio > 0.5:
                decision = 'approve'
                print(f"\n  🟢 决策: 批准上线 (Auto Mode)")
            elif not risk.passed and risk.pbo_estimate > 0.5:
                decision = 'reject'
                print(f"\n  🔴 决策: 拒绝，需重新设计 (Auto Mode)")
            else:
                decision = 'iterate'
                print(f"\n  🟡 决策: 返回迭代优化 (Auto Mode)")
            return decision
        else:
            print("\n  选项: [a]批准上线 [r]拒绝 [i]返回迭代")
            resp = input("  请选择: ").strip().lower()
            return {'a': 'approve', 'r': 'reject', 'i': 'iterate'}.get(resp, 'iterate')


# ==================== 系统编排 ====================

class QuantResearchSystem:
    """
    量化投研系统 - 编排所有Agent协作
    对应整体架构: Orchestration Layer
    """
    
    def __init__(self, auto_mode: bool = True):
        print("=" * 60)
        print("🚀 分层多Agent量化投研系统启动")
        print("=" * 60)
        
        # 初始化组件
        self.llm = MockLLM(creativity=0.3)
        self.mcp = MCPDataServer(seed=SEED)
        
        # 初始化Agent
        self.data_agent = DataAgent(self.mcp, self.llm)
        self.factor_agent = FactorMiningAgent(self.llm)
        self.strategy_agent = StrategyAgent()
        self.backtest_engine = BacktestEngine(initial_capital=1_000_000)
        self.risk_agent = RiskControlAgent()
        self.human = HumanInTheLoop(auto_mode=auto_mode)
        
        # 运行统计
        self.iteration_count = 0
        self.max_iterations = 5
        
    def run(self, stock_code: str, start_date: str, end_date: str,
            max_factors: int = 5) -> Dict:
        """
        执行完整的量化投研Pipeline
        
        Returns:
            包含所有中间结果和最终决策的字典
        """
        print(f"\n{'#'*60}")
        print(f"# 标的: {stock_code} | 区间: {start_date} ~ {end_date}")
        print(f"# 最大迭代: {self.max_iterations} | 模式: {'自动' if self.human.auto_mode else '人工'}")
        print(f"{'#'*60}")
        
        # ============ Step 1: 数据采集 ============
        data_result = self.data_agent.run(stock_code, start_date, end_date)
        stock_data = data_result['stock_data']
        sentiment_scores = data_result['sentiment_scores']
        
        # ============ Step 2: 因子挖掘 (循环迭代) ============
        final_decision = 'iterate'
        all_factors = []
        backtest_result = None
        risk_result = None
        strategy_result = None
        
        while final_decision == 'iterate' and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n{'='*60}")
            print(f"🔄 第 {self.iteration_count}/{self.max_iterations} 轮迭代")
            print(f"{'='*60}")
            
            # 2a. 因子挖掘
            new_factors = self.factor_agent.run(
                stock_data, sentiment_scores, max_factors=max_factors
            )
            all_factors.extend(new_factors)
            
            if not new_factors:
                print("  ❌ 未生成有效因子，终止")
                break
            
            # 2b. 人工审核因子 (HITL)
            if not self.human.review_factors(new_factors):
                print("  ⛔ 因子未通过人工审核，重新生成")
                continue
            
            # ============ Step 3: 策略构建 ============
            strategy_result = self.strategy_agent.run(stock_data, all_factors)
            
            # ============ Step 4: 回测 ============
            backtest_result = self.backtest_engine.run(
                stock_data,
                strategy_result['composite_signal'],
                strategy_result['selected_factors']
            )
            
            # ============ Step 5: 风控检查 ============
            risk_result = self.risk_agent.run(backtest_result, self.llm)
            
            # ============ Step 6: 人工审核回测结果 (HITL) ============
            final_decision = self.human.review_backtest(backtest_result, risk_result)
            
            if final_decision == 'iterate':
                print(f"\n  ↩️ 返回因子挖掘，尝试新因子...")
                # 下一轮会生成不同的因子 ( MockLLM有随机性 )
        
        # ============ 输出最终结果 ============
        return self._compile_results(
            data_result, all_factors, strategy_result,
            backtest_result, risk_result, final_decision
        )
    
    def _compile_results(self, data_result, factors, strategy, 
                         backtest, risk, decision) -> Dict:
        """编译最终报告"""
        print(f"\n{'='*60}")
        print(f"📋 最终报告")
        print(f"{'='*60}")
        
        if decision == 'approve':
            print(f"  🎉 策略已批准! 准备上线")
        elif decision == 'reject':
            print(f"  ❌ 策略被拒绝，需要重新设计")
        else:
            print(f"  ⏸️ 达到最大迭代次数，策略待优化")
        
        print(f"\n  迭代轮次: {self.iteration_count}")
        print(f"  挖掘因子: {len(factors)}个")
        if backtest:
            print(f"  最终收益: {backtest.total_return:+.2%}")
            print(f"  最终Sharpe: {backtest.sharpe_ratio:.2f}")
        
        return {
            'decision': decision,
            'iterations': self.iteration_count,
            'stock_data': data_result['stock_data'],
            'factors': factors,
            'strategy': strategy,
            'backtest': backtest,
            'risk': risk,
            'llm_calls': self.llm.call_count
        }


# ==================== 可视化 ====================

def plot_results(results: Dict):
    """绘制回测结果图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️ matplotlib未安装，跳过图表绘制")
        print("   安装: pip install matplotlib")
        return
    
    backtest = results['backtest']
    if backtest is None:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=100)
    
    # 1. 权益曲线
    ax1 = axes[0]
    equity = backtest.equity_curve
    ax1.plot(equity.index, equity / equity.iloc[0] * 100, 'b-', linewidth=1)
    ax1.set_title('Equity Curve (权益曲线)', fontsize=12)
    ax1.set_ylabel('Normalized Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = axes[1]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title(f'Drawdown (回撤) | Max: {backtest.max_drawdown:.2%}', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True, alpha=0.3)
    
    # 3. 交易标记
    ax3 = axes[2]
    prices = results['stock_data'].prices
    ax3.plot(prices.index, prices, 'k-', linewidth=0.8, alpha=0.7)
    
    for t in backtest.trades:
        if t['type'] == 'BUY':
            ax3.axvline(t['date'], color='green', alpha=0.3, linestyle='--')
        elif t['type'] == 'SELL':
            ax3.axvline(t['date'], color='red', alpha=0.3, linestyle='--')
    ax3.set_title('Trades (交易标记: 绿=买入 红=卖出)', fontsize=12)
    ax3.set_ylabel('Price')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/quant_agent_demo_result.png', dpi=150)
    plt.close()
    print(f"\n  📊 图表已保存: quant_agent_demo_result.png")


def print_banner():
    """打印系统横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║     分层多Agent量化投研系统 - 最小可运行Demo                  ║
    ║     Layered Multi-Agent Quantitative Research System          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  架构: Data → Factor → Strategy → Backtest → Risk → HITL   ║
    ║  核心: R&D-Agent-Quant + AlphaCrafter + TradingAgents        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


# ==================== 主入口 ====================

def main():
    """主函数 - 运行完整Demo"""
    print_banner()
    
    # 初始化系统
    system = QuantResearchSystem(auto_mode=True)
    
    # 运行投研Pipeline
    results = system.run(
        stock_code='600519.SH',      # 模拟标的
        start_date='2023-01-01',      # 数据起始
        end_date='2024-06-01',        # 数据结束
        max_factors=3                  # 每轮挖掘3个因子
    )
    
    # 绘制结果
    plot_results(results)
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"🏁 Demo运行完成")
    print(f"{'='*60}")
    print(f"  LLM调用次数: {results['llm_calls']}")
    print(f"  总迭代轮次: {results['iterations']}")
    print(f"  最终决策: {results['decision']}")
    print(f"\n  💡 要接入真实LLM，修改MockLLM类中的调用逻辑")
    print(f"  💡 要接入真实数据，替换MCPDataServer中的方法")
    print(f"  💡 要人工交互，设置 QuantResearchSystem(auto_mode=False)")
    
    return results


if __name__ == '__main__':
    results = main()
