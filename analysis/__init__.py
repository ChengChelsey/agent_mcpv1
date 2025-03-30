# analysis/__init__.py
"""
分析模块，包含单序列和多序列时序数据分析功能
"""
from analysis.single_series import analyze_single_series
from analysis.multi_series import analyze_multi_series
from analysis.data_alignment import align_series