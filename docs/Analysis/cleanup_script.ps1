# PowerShell清理脚本
# 用于删除IoT项目中的重复目录

# 删除重复的目录
Write-Host "开始删除重复目录..."

# 设置基础路径
$basePath = "docs\Analysis"

# 定义要删除的目录列表
$directoriesToDelete = @(
    "00-Index",
    "01-Industry_Architecture",
    "02-Enterprise_Architecture",
    "03-Conceptual_Architecture",
    "04-Algorithms",
    "05-Technology_Stack",
    "06-Business_Specifications",
    "07-Performance",
    "09-Integration",
    "10-Standards",
    "11-IoT-Architecture"
)

# 循环删除每个目录
foreach ($dir in $directoriesToDelete) {
    $fullPath = Join-Path -Path $basePath -ChildPath $dir
    if (Test-Path -Path $fullPath) {
        Write-Host "删除目录: $fullPath"
        Remove-Item -Path $fullPath -Recurse -Force
    } else {
        Write-Host "目录不存在: $fullPath"
    }
}

Write-Host "目录清理完成!"

# 创建项目完成标记文件
$completionFile = Join-Path -Path $basePath -ChildPath "PROJECT_CLEANUP_COMPLETED"
Set-Content -Path $completionFile -Value "项目清理完成于: $(Get-Date)"

Write-Host "清理任务全部完成!从$($directoriesToDelete.Count)个目录中清理了重复内容。" 