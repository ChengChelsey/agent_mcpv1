# detectors/ttest_detector.py
import statistics, math

def detect_ttest(series):
    """
    对单序列做某种T检验,演示
    """
    if not series:
        return {"method":"TTest","anomalies":[],"scores":[],"description":"无数据"}
    vals= [v for (_,v) in series]
    meanv= statistics.mean(vals)
    stdev= statistics.pstdev(vals) if len(vals)>1 else 0
    n= len(vals)
    if stdev==0 or n<2:
        return {"method":"TTest","anomalies":[],"scores":[],"description":"无法t检验"}
    t_stat= meanv/(stdev/ math.sqrt(n))
    # 仅演示fake p
    p= 2* math.exp(-abs(t_stat))
    if p<0.05:
        anoms= [ts for (ts,_) in series]  # 全异常(纯演示)
        desc= f"T检验:t={t_stat:.2f},p={p:.3f}=>显著"
    else:
        anoms=[]
        desc= f"T检验:t={t_stat:.2f},p={p:.3f}=>不显著"
    return {
      "method":"TTest",
      "anomalies": anoms,
      "scores":[],
      "description": desc
    }

def detect_ttest_2samples(series1, series2):
    """
    对比两序列 => t检验
    """
    if not series1 or not series2:
        return {"method":"TTest2","anomalies":[],"scores":[],"description":"无数据"}
    v1= [v for (_,v) in series1]
    v2= [v for (_,v) in series2]
    mean1= statistics.mean(v1)
    mean2= statistics.mean(v2)
    var1= statistics.pvariance(v1)
    var2= statistics.pvariance(v2)
    n1= len(v1)
    n2= len(v2)
    # pool var
    if n1+n2<3:
        return {"method":"TTest2","anomalies":[],"scores":[],"description":"样本过少"}
    sp= math.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2)) if (n1+n2>2) else 0
    if sp==0:
        return {"method":"TTest2","anomalies":[],"scores":[],"description":"方差=0?"}
    t_stat= (mean1- mean2)/(sp* math.sqrt(1/n1+1/n2))
    p= 2* math.exp(-abs(t_stat))
    desc= f"双样本T检验:t={t_stat:.2f},p={p:.3f}"
    anomalies=[]
    if p<0.05:
        # 全异常
        all_ts= set(ts for (ts,_) in series1).union(ts for (ts,_) in series2)
        anomalies= sorted(all_ts)
        desc+= "=>显著差异"
    else:
        desc+= "=>无显著差异"
    return {
      "method":"TTest2",
      "anomalies": anomalies,
      "scores":[],
      "description": desc
    }
