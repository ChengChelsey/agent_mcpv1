import json
import config
try:
    import pymysql
except ImportError:
    pymysql = None

def save_analysis_record(record):
    """
    保存分析记录到 MySQL 数据库。
    record: dict 包含 ip, field, start_time, end_time, methods, anomalies, composite_score, classification, report 等
    """
    ip = record.get("ip")
    field = record.get("field")
    start = record.get("start_time")
    end = record.get("end_time")
    methods = record.get("methods")
    anomalies = record.get("anomaly_times")
    composite_score = record.get("composite_score")
    classification = record.get("classification")
    report = record.get("report")
    methods_json = json.dumps(methods, ensure_ascii=False)
    anomalies_json = json.dumps(anomalies, ensure_ascii=False)
    insert_sql = (
        "INSERT INTO anomaly_analysis_records "
        "(ip, field, start_time, end_time, methods, anomalies, composite_score, classification, report) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    values = (ip, field, start, end, methods_json, anomalies_json, composite_score, classification, report)
    try:
        if pymysql is None:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute(insert_sql, values)
                connection.commit()
        else:
            conn = pymysql.connect(host=config.DB_CONFIG['HOST'],
                                   port=config.DB_CONFIG['PORT'],
                                   user=config.DB_CONFIG['USER'],
                                   password=config.DB_CONFIG['PASSWORD'],
                                   database=config.DB_CONFIG['NAME'],
                                   charset='utf8mb4')
            cursor = conn.cursor()
            cursor.execute(insert_sql, values)
            conn.commit()
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"保存记录失败: {e}")
