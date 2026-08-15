# CRP-VLA Task 11–12 最终总结（2026-08-16）

## 结论

Task 11 **正式通过**：20k Snap-1 在严格独立的 1,200 个 ordinary-LIBERO 配对案例上满足相对 Base-10 的 `-3%` 非劣效界限。

Task 12 **已完成但训练探针不晋升**：机制审计显示差异主要集中在 Snap-1 自己访问到的闭环状态，并会累积为机器人关节与末端轨迹分歧；一次固定的 500-step preservation probe 虽改善离线目标，却未改善目标 Snap-1 闭环成功率。因此保留原 20k Snap-1 作为已确认基线，不追加调参或训练。

## VERIFIED：Task 11 正式确认

| 指标 | 结果 |
|---|---:|
| 完整配对 | 1,200 / 1,200 |
| Base-10 | 969 / 1,200（80.75%） |
| 20k Snap-1 | 959 / 1,200（79.9167%） |
| Snap-1 − Base-10 | −0.8333 个百分点 |
| 配对 bootstrap 95% CI | [−2.9167%, +1.25%] |
| 非劣效界限 | −3% |
| 决策 | `FORMAL_CONFIRMATION1200_PASS` |

完整性审计覆盖 2,400/2,400 arm results、40/40 shards、47,276 个 replan archives、94,552 个 capsule members，以及 4,895/4,895 个最终校验项。

## VERIFIED：Task 12 机制定位

- Phase A 覆盖全部 1,200 对已归档轨迹。
- Phase B 在 7,772 个归档状态上完成 15,544 次同状态策略查询，并得到 7,772/7,772 次精确 self-replay。
- 在 Snap-visited 状态中，失败案例相对保留成功案例的动作差异显著更大；对应的 Base-visited 对比置信区间跨零。
- 独立分支审计恢复 3,842 个 Snap-visited 状态，完成 7,684/7,684 条隔离分支。
- harmful-minus-preserved 的关节、末端位置和末端姿态差异在 h1/h3/h5/h10 均为正；物体位置差异区间跨零。

这些结果支持“局部学生访问状态敏感性会累积为机器人轨迹分歧”的机制描述，但不是一般化因果结论，也不是语言 grounding 方法声明。

## VERIFIED：固定训练探针

唯一获准的探针从已确认的 20k checkpoint 继续 500 steps；参数在训练前冻结，未做中间 checkpoint 选择或超参 sweep。

| 指标 | 训练前 | 训练后 |
|---|---:|---:|
| held-out Snap-visited Base-target MSE | 0.01165795 | 0.01040619 |
| 相对改善 | — | 10.7% |
| held-out Base-visited anchor drift MSE | 约 0 | 0.00035369 |

模型 SHA-256：`b29bdff8b1bd0125e15805a074d1bc1366c905ff026bc6d22f14231cc158dd01`。

## VERIFIED：冻结 dev40 闭环比较

| Arm | 原 20k | 探针 | 变化 |
|---|---:|---:|---:|
| Snap-10 | 30/40 | 30/40 | 0 |
| Snap-2 | 32/40 | 34/40 | +2 |
| Snap-1（目标） | 32/40 | 30/40 | −2 |

Snap-1 的配对成功率差为 `-0.05`，case 与 task-cluster bootstrap 95% CI 均为 `[-0.15, +0.05]`，exact McNemar `p=0.625`；翻转计数为 1 个 probe-only success、3 个 original-only success。120/120 条探针轨迹均重新计算内容哈希，路径和哈希全部唯一。

## 决策与证据边界

- `VERIFIED`：原 20k Snap-1 通过 Task 11 非劣效确认，继续作为项目基线。
- `VERIFIED`：该固定 probe 改善离线拟合，但没有改善目标一阶闭环端点。
- `INFERRED`：单纯的同状态 action-prefix preservation loss 与真实闭环分布修复之间仍有缺口。
- `UNVERIFIED`：该机制是否能跨 checkpoint、训练种子、机器人平台或 LIBERO-CF 泛化。
- 决策：`TASK12_EXPLORATORY_PROBE_NOT_PROMOTED`；不在 dev40 上调参，不重用 Task 11 做选择，不训练 LIBERO-CF。

## 下一步建议

下一项研究任务应重新预注册，而不是延续本 probe 做局部调参。建议设计一个全新的、与 dev40 和 Task 11 均不重叠的闭环开发集，并把“student-visited 状态分布下的闭环保持”直接纳入选择标准；在该协议冻结前不启动新训练。正式确认仍应保留一个从未用于训练、诊断或选择的独立集合。
