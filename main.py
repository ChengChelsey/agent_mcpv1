# main.py
import sys
import agent

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("请输入查询: ")
    agent.chat(query)

if __name__ == '__main__':
    main()
