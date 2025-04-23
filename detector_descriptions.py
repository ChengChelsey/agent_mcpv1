"""
ADTK 异常检测方法的结构化描述
每个检测方法包含：原理描述、适用场景、局限性、参数说明和默认参数
"""

DETECTOR_DESCRIPTIONS = {
    "IQR异常检测": {
        "category": "univariate",
        "method": "InterQuartileRangeAD",
        "description": "基于四分位数范围的异常检测，通过计算序列的四分位距（IQR）来识别离群值。",
        "principle": "计算数据的第一四分位数（Q1）和第三四分位数（Q3），定义四分位距IQR=Q3-Q1，认为落在[Q1-c*IQR, Q3+c*IQR]范围之外的点为异常点。",
        "suitable_features": [
            "适用于大多数数据分布，尤其是近似正态分布数据",
            "对于有明显离群值的数据效果好",
            "不受极端值影响",
            "适合处理数据范围变化的情况"
        ],
        "limitations": [
            "不考虑时间序列的时间依赖性",
            "对于多峰分布效果较差",
            "无法检测趋势性异常",
            "对季节性变化敏感度低"
        ],
        "parameters": {
            "c": {
                "description": "异常判定倍数，值越大异常判定越宽松",
                "type": "float",
                "default": 3.0,
                "range": [1.0, 10.0]
            }
        },
        "default_params": {"c": 3.0},
        "output_type": "point"  # 输出异常点
    },
    
    "广义ESD检测": {
        "category": "univariate",
        "method": "GeneralizedESDTestAD",
        "description": "基于广义极端学生化残差（Generalized ESD）统计检验的异常检测方法。",
        "principle": "该方法通过迭代地寻找和移除数据中的离群值，对每次移除，计算极端学生化残差统计量并与临界值比较，以决定该点是否为离群值。",
        "suitable_features": [
            "适用于近似正态分布的数据",
            "对有少量离群值的数据效果好",
            "适合需要统计显著性的场景",
            "有明确的异常点数量上限"
        ],
        "limitations": [
            "要求数据近似正态分布",
            "不适合有强烈季节性的数据",
            "计算开销较大",
            "需要指定最大异常数量"
        ],
        "parameters": {
            "alpha": {
                "description": "显著性水平，值越小判定越严格",
                "type": "float",
                "default": 0.05,
                "range": [0.001, 0.1]
            }
        },
        "default_params": {"alpha": 0.05},
        "output_type": "point"  # 输出异常点
    },
    
    "分位数异常检测": {
        "category": "univariate",
        "method": "QuantileAD",
        "description": "基于分位数的异常检测方法，定义超出指定分位数范围的点为异常。",
        "principle": "计算数据的低分位点和高分位点，将低于低分位点或高于高分位点的值标记为异常。",
        "suitable_features": [
            "适用于任意分布数据",
            "对非正态分布数据也有效",
            "处理噪声数据时稳健",
            "能适应不同数据范围"
        ],
        "limitations": [
            "静态阈值可能不适合动态变化的数据",
            "无法检测趋势变化",
            "对极端值过于敏感",
            "不考虑时间序列的时序特性"
        ],
        "parameters": {
            "low": {
                "description": "低分位阈值，例如0.05表示5%分位点",
                "type": "float",
                "default": 0.05,
                "range": [0.001, 0.2]
            },
            "high": {
                "description": "高分位阈值，例如0.95表示95%分位点",
                "type": "float",
                "default": 0.95,
                "range": [0.8, 0.999]
            }
        },
        "default_params": {"low": 0.05, "high": 0.95},
        "output_type": "point"  # 输出异常点
    },
    
    "阈值异常检测": {
        "category": "univariate",
        "method": "ThresholdAD",
        "description": "基于固定阈值的异常检测，简单直观地将超出指定阈值范围的点标记为异常。",
        "principle": "将低于指定下限或高于指定上限的数据点标记为异常，适用于有明确阈值限制的监控场景。",
        "suitable_features": [
            "适用于有明确阈值边界的数据",
            "适合监控明确上下限的指标",
            "简单直观，计算开销小",
            "对定义明确的异常有高检出率"
        ],
        "limitations": [
            "需要预先知道合理阈值",
            "不适应数据分布变化",
            "过于简单可能导致误报",
            "无法检测复杂模式异常"
        ],
        "parameters": {
            "low": {
                "description": "下限阈值，低于此值的点被标记为异常",
                "type": "float",
                "default": None,
                "range": [None, None]
            },
            "high": {
                "description": "上限阈值，高于此值的点被标记为异常",
                "type": "float",
                "default": None,
                "range": [None, None]
            }
        },
        "default_params": {},  # 默认不设阈值，需要用户指定
        "output_type": "point"  # 输出异常点
    },
    
    "持续性异常检测": {
        "category": "univariate",
        "method": "PersistAD",
        "description": "检测持续异常状态，识别在一段时间内未发生足够变化的序列。",
        "principle": "当时间序列在指定窗口内变化很小（低于预期变异度）时，判定为持续性异常，适合检测系统卡死等情况。",
        "suitable_features": [
            "适用于应该保持相对稳定的数据",
            "能检测出持续偏离正常水平的异常",
            "适合检测系统状态变化",
            "对渐变异常敏感"
        ],
        "limitations": [
            "对短期波动敏感度低",
            "可能错过短暂的异常尖峰",
            "窗口大小选择影响检测效果",
            "不适用于有明显趋势的数据"
        ],
        "parameters": {
            "window": {
                "description": "滑动窗口大小",
                "type": "int",
                "default": 1,
                "range": [1, 100]
            },
            "c": {
                "description": "变异度阈值倍数",
                "type": "float",
                "default": 3.0,
                "range": [1.0, 10.0]
            }
        },
        "default_params": {"window": 1, "c": 3.0},
        "output_type": "point"  # 输出异常点
    },
    
    "水平位移检测": {
        "category": "univariate",
        "method": "LevelShiftAD",
        "description": "检测数据水平的突变，识别时间序列中的明显跳变。",
        "principle": "通过比较相邻窗口的平均值差异，检测时间序列中的明显水平位移，适合识别系统配置变更等引起的阶跃变化。",
        "suitable_features": [
            "适用于存在明显水平跳变的数据",
            "能检测出均值突变",
            "适合系统配置变更、环境变化等场景",
            "对阶跃变化敏感"
        ],
        "limitations": [
            "无法检测渐变异常",
            "对噪声敏感",
            "可能将正常的季节性变化误判为异常",
            "窗口大小选择影响检测效果"
        ],
        "parameters": {
            "window": {
                "description": "用于计算水平的滑动窗口大小",
                "type": "int",
                "default": 5,
                "range": [2, 100]
            },
            "c": {
                "description": "显著性阈值倍数",
                "type": "float",
                "default": 6.0,
                "range": [1.0, 20.0]
            }
        },
        "default_params": {"window": 5, "c": 6.0},
        "output_type": "point"  # 输出异常点
    },
    
    "波动性变化检测": {
        "category": "univariate",
        "method": "VolatilityShiftAD",
        "description": "检测数据波动性的变化，识别时间序列中的波动幅度突变。",
        "principle": "通过比较相邻窗口的标准差差异，检测时间序列中的波动性突变，适合识别系统不稳定性增加等情况。",
        "suitable_features": [
            "适用于应保持稳定波动幅度的数据",
            "能检测出波动性突变",
            "适合检测系统不稳定性增加",
            "对方差变化敏感"
        ],
        "limitations": [
            "需要足够的历史数据建立基线",
            "对初始波动性假设敏感",
            "可能忽略均值变化",
            "窗口大小选择影响检测效果"
        ],
        "parameters": {
            "window": {
                "description": "用于计算波动性的滑动窗口大小",
                "type": "int",
                "default": 10,
                "range": [5, 100]
            },
            "c": {
                "description": "显著性阈值倍数",
                "type": "float",
                "default": 6.0,
                "range": [1.0, 20.0]
            }
        },
        "default_params": {"window": 10, "c": 6.0},
        "output_type": "point"  # 输出异常点
    },
    
    "季节性异常检测": {
        "category": "univariate",
        "method": "SeasonalAD",
        "description": "检测季节性数据的异常，识别违背季节性模式的数据点。",
        "principle": "首先分解时间序列获取季节性成分，然后检测实际值与预期季节性模式的偏差，适合周期性波动的时序数据。",
        "suitable_features": [
            "适用于有明显周期性模式的数据",
            "能检测出违背季节性模式的异常",
            "适合日常、周度或月度循环的数据",
            "需要数据至少包含2-3个完整周期"
        ],
        "limitations": [
            "需要准确指定季节性周期",
            "数据不足时效果差",
            "对非季节性异常检测能力弱",
            "计算开销大"
        ],
        "parameters": {
            "freq": {
                "description": "季节性周期长度，如每天24小时则设为24",
                "type": "int",
                "default": None,  # 自动推断
                "range": [2, 1000]
            },
            "c": {
                "description": "异常判定阈值倍数",
                "type": "float",
                "default": 3.0,
                "range": [1.0, 10.0]
            }
        },
        "default_params": {"freq": None, "c": 3.0},
        "output_type": "point"  # 输出异常点
    },
    
    "自回归异常检测": {
        "category": "univariate",
        "method": "AutoregressionAD",
        "description": "基于自回归模型的异常检测，通过预测与实际值的偏差识别异常。",
        "principle": "使用自回归模型基于历史数据预测当前值，将实际值与预测值的显著偏差标记为异常，适合有时间相关性的数据。",
        "suitable_features": [
            "适用于有时间相关性的数据",
            "能检测出违背历史模式的异常",
            "适合有短期依赖关系的数据",
            "对短期预测偏差敏感"
        ],
        "limitations": [
            "需要足够的训练数据",
            "对参数选择敏感",
            "计算开销大",
            "不适合有长期依赖性的数据"
        ],
        "parameters": {
            "n_steps": {
                "description": "预测使用的历史步长，即AR模型的阶数",
                "type": "int",
                "default": 1,
                "range": [1, 50]
            },
            "step_size": {
                "description": "步长间隔",
                "type": "int",
                "default": 1,
                "range": [1, 20]
            },
            "c": {
                "description": "异常判定阈值倍数",
                "type": "float",
                "default": 3.0,
                "range": [1.0, 10.0]
            }
        },
        "default_params": {"n_steps": 1, "step_size": 1, "c": 3.0},
        "output_type": "point"  # 输出异常点
    }
}