import os

def dump_project_code(project_root, output_file="project_code_dump.txt"):
    with open(output_file, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(project_root):
            # 跳过 __pycache__
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, project_root)
                    out.write(f"# === {rel_path} ===\n")
                    with open(file_path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                    out.write("\n\n")

# 使用示例：设定你的项目目录路径
project_path = "/home/cnic/aiagent1"  # 修改成你的实际路径
dump_project_code(project_path)
