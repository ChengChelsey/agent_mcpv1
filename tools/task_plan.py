# agent/tools/task_plan.py
from typing import List, Optional

class SeriesInfo:
    def __init__(self, label: str, ip: str, field: str, start: str, end: str):
        self.label = label                # 用户层面的标签（如“昨天”、“这周一”）
        self.ip = ip                      # 数据源IP
        self.field = field                # 指标名称（如 'cpu_rate'）
        self.start = start                # 开始时间（'2025-03-27 00:00:00'）
        self.end = end                    # 结束时间

class TaskConfig:
    def __init__(
        self,
        task_id: str,
        task_type: str,                  # "single", "pair", "multivariate"
        field: str,
        series: List[SeriesInfo],
        enabled_methods: Optional[List[str]] = None,
        target_report_path: Optional[str] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.field = field
        self.series = series
        self.enabled_methods = enabled_methods or []
        self.target_report_path = target_report_path

class TaskPlan:
    def __init__(
        self,
        user_query: str,
        tasks: List[TaskConfig],
        output_dir: str = "output/plots"
    ):
        self.user_query = user_query
        self.tasks = tasks
        self.output_dir = output_dir

    @staticmethod
    def from_dict(plan_dict: dict) -> "TaskPlan":
        return TaskPlan(
            user_query=plan_dict["user_query"],
            output_dir=plan_dict.get("output_dir", "output/plots"),
            tasks=[
                TaskConfig(
                    task_id=task["task_id"],
                    task_type=task["task_type"],
                    field=task["field"],
                    series=[
                        SeriesInfo(
                            label=series.get("label", ""),
                            ip=series["ip"],
                            field=series.get("field", task["field"]),
                            start=series["start"],
                            end=series["end"]
                        ) for series in task["series"]
                    ],
                    enabled_methods=task.get("enabled_methods"),
                    target_report_path=task.get("target_report_path")
                )
                for task in plan_dict["tasks"]
            ]
        )
