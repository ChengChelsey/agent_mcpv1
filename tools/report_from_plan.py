# tools/report_from_plan.py
import os
import pickle
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 导入任务计划类
from tools.task_plan import SeriesInfo, TaskConfig, TaskPlan

# 导入检测器
from detectors import (
    ZScoreDetector,
    STLDetector,
    CUSUMDetector,
    ResidualComparisonDetector,
    TrendDriftCUSUMDetector,
    ChangeRateDetector,
    TrendSlopeDetector
)

# 导入可视化和报告生成模块
from output.visualization import generate_summary_echarts_html
from output.report_generator import generate_report_single_series

# 导入工具函数
from utils.ts_cache import ensure_cache_file, load_series_from_cache

def run_task_and_generate_report(task_plan_dict: dict) -> dict:
    """
    根据TaskPlan执行异常检测任务并生成可视化报告
    
    参数:
        task_plan_dict: TaskPlan字典定义
        
    返回:
        dict: 包含执行结果路径和摘要
    """
    # 解析任务计划
    plan = TaskPlan.from_dict(task_plan_dict)
    result_paths = []
    
    # 遍历执行每个任务
    for task in plan.tasks:
        # 创建任务输出目录
        ts = int(time.time())
        task_dir = os.path.join(plan.output_dir, f"{task.task_id}_{ts}")
        os.makedirs(task_dir, exist_ok=True)
        
        # 处理单序列异常检测任务
        if task.task_type == "single":
            # 加载数据
            s = task.series[0]
            series = load_series_from_cache(s.ip, s.field, s.start, s.end)
            
            # 执行检测
            results = [
                ZScoreDetector().detect(series),
                STLDetector().detect(series),
                CUSUMDetector().detect(series)
            ]
            
            # 筛选启用的方法（如果指定）
            if task.enabled_methods:
                results = [r for r in results if r.method in task.enabled_methods]
            
            # 生成图表和报告
            chart_path = os.path.join(task_dir, "chart.html")
            html_path = os.path.join(task_dir, "final_report.html")
            
            # 汇总可视化
            chart_path, tooltip_map = generate_summary_echarts_html(
                series, results, chart_path, 
                title=f"{s.ip} {s.field} 异常检测汇总图"
            )
            
            # 生成报告
            generate_report_single_series(
                plan.user_query, chart_path, results, tooltip_map, html_path
            )
            
            result_paths.append(html_path)
            
        # 处理多序列对比检测任务
        elif task.task_type == "pair":
            # 加载两个序列数据
            s1, s2 = task.series
            series1 = load_series_from_cache(s1.ip, s1.field, s1.start, s1.end)
            series2 = load_series_from_cache(s2.ip, s2.field, s2.start, s2.end)
            
            # 执行对比检测
            results = [
                ResidualComparisonDetector().detect(series1, series2),
                TrendDriftCUSUMDetector().detect(series1, series2),
                ChangeRateDetector().detect(series1, series2),
                TrendSlopeDetector().detect(series1, series2)
            ]
            
            # 筛选启用的方法（如果指定）
            if task.enabled_methods:
                results = [r for r in results if r.method in task.enabled_methods]
            
            # 生成图表和报告
            chart_path = os.path.join(task_dir, "chart.html")
            html_path = os.path.join(task_dir, "final_report.html")
            
            # 汇总可视化 - 为多序列对比实现特殊版本
            chart_path, tooltip_map = generate_summary_echarts_html(
                series1, results, chart_path,
                title=f"{s1.ip} vs {s2.ip} 对比异常检测汇总图"
            )
            
            # 生成报告
            generate_report_single_series(
                plan.user_query, chart_path, results, tooltip_map, html_path
            )
            
            result_paths.append(html_path)
            
        # 暂不支持的任务类型
        else:
            print(f"[跳过] 暂不支持的 task_type: {task.task_type}")
    
    # 返回汇总结果
    return {
        "user_query": plan.user_query,
        "summary": f"已完成 {len(result_paths)} 个任务的图表与报告生成。",
        "report_paths": result_paths
    }