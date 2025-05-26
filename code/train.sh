#!/bin/bash

# 保存原工作路径
ORIGINAL_DIR="$(pwd)"

# 获取脚本所在的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 切换到脚本所在目录
cd "$SCRIPT_DIR" || exit
# 切换到上级目录
cd "../" || exit

run_python_script() {
    local script_name="$1"
    # 检查并运行指定的 Python 脚本
    if [ -f "$script_name" ]; then
        echo "正在运行 $script_name..."
        python "$script_name"
        if [ $? -ne 0 ]; then
            echo "运行 $script_name 失败"
            exit 1
        fi
    else
        echo "错误: 找不到 $script_name"
        exit 1
    fi
}

run_python_script "code/get_dataset.py"
run_python_script "code/augment.py"
run_python_script "code/train.py"
echo "所有脚本执行完成"

# 切换回原工作路径
cd "$ORIGINAL_DIR" || exit