# main.py
import sys
import agent

def main():
    last_results = {} 
    
    while True:
        user_query = input("请输入查询 (输入'退出'结束): ")
        if user_query.lower() in ["退出", "exit", "quit", "q"]:
            print("谢谢使用，再见！")
            break

        if "上次" in user_query or "之前" in user_query or "刚才" in user_query:
            pass
            
        result = agent.chat(user_query)
        last_results = result

        print("\n是否有其他问题？")

if __name__ == '__main__':
    main()
