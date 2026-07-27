# ProcessSqlData-TS

从 PostgreSQL 提取设备时序数据并执行清洗、统计和建模的历史研究原型。

> 严重警告：不要直接运行 `ProcessSqlData Final.py`。脚本包含内联数据库连接参数，会递归删除多个相对路径目录，并采用不稳定的分页方式。它是 Python 2 时代的一体化脚本，不适合作为当前生产任务。

## 原始处理流程

单个脚本按顺序执行以下阶段：

1. 从 PostgreSQL 分页导出 CSV；
2. 按设备分组；
3. 解析设备特征；
4. 合并并按时间排序；
5. 将各特征拆分为独立时间序列；
6. 删除异常值并绘图；
7. 判断零方差和白噪声序列；
8. 生成时间戳并插值；
9. 执行 ARIMA 建模；
10. 生成直方图、均值、方差和线性趋势结果。

执行过程中会在脚本上级目录创建或重建多组 `Data*` 目录。

## 历史技术栈

- Python 2
- PostgreSQL / `psycopg2`
- NumPy、pandas、SciPy
- Matplotlib
- scikit-learn
- statsmodels 旧版 ARIMA API
- `gatspy`
- Parallel Python（`pp`）

项目没有依赖清单或版本锁。当前源码在 Python 3.12 下会因 Python 2 的 `print` 语法而无法解析，并且还使用了 `unicode`、`DataFrame.append` 和已废弃的 statsmodels API。

## 已确认的高风险问题

- 数据库连接信息直接写在源码中。相关凭据应视为已暴露并立即轮换，不能只删除当前文件；
- 脚本启动后会多次 `shutil.rmtree` 删除相对路径，工作目录不正确时可能误删数据；
- `CurOffset` 在第一次查询前先增加 `10000`，导致首批 10,000 行被跳过；
- 分页查询没有 `ORDER BY`，数据库返回顺序变化时可能重复或遗漏记录；
- 所有阶段在导入文件时立即执行，没有主入口保护、检查点或事务边界；
- `ProcessFile` 被重复定义十次，阶段间依赖通过目录和全局状态隐式传递；
- ARIMA 阶段差分后保留空值，并错误地读取 Python `list.shape`，非零方差序列无法稳定完成；
- 绘图和趋势阶段使用错误的列位置/时间类型，按当前脚本会分别越界或解析失败；
- 裸异常处理和缺少数据契约，使错误结果难以被发现。

## 安全审阅方式

在完成重构前，只建议进行静态阅读。不要：

- 连接任何生产或仍在使用的数据库；
- 在含重要文件的目录中执行脚本；
- 把真实凭据改成新的值后继续提交；
- 把历史输出当成完整、稳定或可复现的数据集。

## 已确定的迁移策略

1. **外部兼容优先**：保留原启动命令以及全部输入、输出、中间目录、文件名、CSV 编码和列结构。
2. **双模式运行**：`legacy` 保留旧数据行为；`corrected` 修复分页、排序和已确认的逻辑错误。
3. **本地配置**：真实数据库配置只写入 Git 忽略的 `config.local.toml`，不使用 Windows 环境变量。
4. **数据库只读**：连接后强制只读事务和查询超时，无法确认只读状态时立即关闭连接。
5. **路径固定**：所有旧输出路径根据入口脚本位置计算，不再依赖当前 PowerShell 目录。
6. **安全发布**：结果先写入 `.runtime` 下的隔离目录，完整成功后再替换正式输出，并保留回滚副本。
7. **整体切换**：可以逐阶段开发，但所有阶段通过端到端兼容验证前不替换正式入口。

## 当前迁移基础

新的 Python 3.12 基础代码位于 `src/process_sql_data/`，目前已经实现：

- 严格 TOML 配置加载与占位值检查；
- 强制只读数据库会话；
- 14 个历史输出目录的固定契约；
- 安全的 staging、发布和回滚路径；
- CSV 编码、表头、行顺序、精确值和浮点容差比较；
- `legacy` offset 分页和 `corrected` 主键 keyset 分页计划；
- 分设备、特征解析、排序、异常值、时间戳、插值和统计等纯数据变换；
- 零方差/白噪声判定、无界面时序图与直方图输出；
- 使用新版 statsmodels API 的有界 ARIMA 阶数选择；
- 只在 `.runtime` staging 中运行的阶段 2–14 文件级编排；
- 具有输入指纹校验的显式断点续跑状态；
- 不需要安装第三方测试框架的标准库测试入口。

复制配置模板后再填写本地值：

```powershell
Copy-Item config.example.toml config.local.toml
```

`config.local.toml` 已被 `.gitignore` 排除。不要把真实配置复制到其他已跟踪文件。

运行测试：

```powershell
python test/run_tests.py
```

安全迁移工具不会运行旧流水线：

```powershell
python migration_tools.py show-layout
python migration_tools.py validate-config
python migration_tools.py compare --expected <legacy-output> --actual <new-output>
```

## 验证状态

基础测试共 70 项，覆盖配置、路径、输出比较、只读数据库会话、双模式分页、原子发布、失败回滚、断点状态、纯数据变换、时序判定、绘图和 ARIMA 适配，当前全部通过。合成数据已经在 `legacy` 和 `corrected` 两种模式下完成阶段 2–14 的端到端验证，且只写入隔离 staging。ARIMA 已通过合成数据和新版 API 冒烟验证，但旧版与新版实现的精确数值兼容仍需使用脱敏历史样本做黄金对比。旧脚本仍在第 131 行存在 Python 2 `print` 语法错误；出于安全考虑，尚未连接数据库，也没有运行旧流水线或改写任何正式 `Data*` 输出目录。

## 许可证

仓库目前没有顶层许可证。
