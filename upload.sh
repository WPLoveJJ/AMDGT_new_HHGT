#!/bin/bash
# 确保脚本出错时停止
set -e

echo "📦 [1/3] 正在添加所有文件..."
git add .

# 获取备注信息
commit_msg="$1"
if [ -z "$commit_msg" ]; then
    commit_msg="Auto update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "📝 [2/3] 正在提交... 备注: $commit_msg"
# 只有当有变化时才提交，避免产生空提交报错
if ! git diff-index --quiet HEAD --; then
    git commit -m "$commit_msg"
else
    echo "⚠️  没有文件发生变化，跳过提交步骤。"
fi

echo "🚀 [3/3] 正在推送到远程服务器..."
git push

echo "✅ 全部完成！"
