# main.py
import sys
import agent

def main():
    last_results = {}  # 用于存储最近一次的分析结果
    
    while True:
        user_query = input("请输入查询 (输入'退出'结束): ")
        if user_query.lower() in ["退出", "exit", "quit", "q"]:
            print("谢谢使用，再见！")
            break
            
        # 如果查询中包含"上次"或"之前"等词，可以传递上下文
        if "上次" in user_query or "之前" in user_query or "刚才" in user_query:
            # 可以在这里添加上下文处理逻辑
            pass
            
        result = agent.chat(user_query)
        last_results = result  # 存储结果供后续使用
        
        print("\n是否有其他问题？")

if __name__ == '__main__':
    main()
