# Unity 手势模型交付说明

## 本次模型

- 模型编号：`ok-20260806-134328`
- 原始标签：`none`、`ok`
- 背景标签：`none`
- 目标手势：`ok`
- SHA-256：`20db31e5c2cf148703124440d8b946ce11ac799305541a1d59af5c1e4e940695`

## 当前交付状态

- 模型替换所需文件已齐全，可交付 Unity 开发接入。
- 未参与训练的验收图片/视频尚未提供。
- 目标 Unity 设备上的逐类验收记录尚未完成。
- 后两项属于正式验收前的待补充项，不影响开发先完成模型替换。

## 如何替换

1. 备份 Unity 工程中现有模型。
2. 用 `unity/ai/mediapipe/gesture_recognizer.bytes` 替换工程中的 `ai/mediapipe/gesture_recognizer.bytes`。
3. 同步更新 `label_mapping.json` 对应的业务动作配置。
4. 在目标设备上完成逐类验收。

`.bytes` 与 `models/gesture_recognizer.task` 的二进制内容完全相同，只是扩展名不同。如果项目不通过 Unity `TextAsset` 加载，也可使用原始 `.task` 文件。

## Unity 代码要求

- 使用当前模型的自定义分类头结果，并按原始字符串读取标签。
- 标签区分大小写；不要将 `ok` 改成 `OK` 后再匹配。
- `none` 不触发任何业务动作。
- 未知标签应记录日志并忽略，不能静默映射成 `none`。
- 保留 MediaPipe 返回的原始 `score`，不进行伪造或缩放。
- 模型元数据中的全局 score 阈值为 `0.5`。如 Unity 额外增加业务阈值或多帧防抖，请作为独立配置记录。
- 不要假设 MediaPipe 官方手势枚举是本模型的业务标签。

## 验收项

- `checksums.sha256` 校验通过。
- Unity 能加载模型，不出现模型格式或资源路径错误。
- `ok` 能以原始字符串返回并触发对应动作。
- `none` 不触发业务动作。
- 验收左右手、远近距离、明暗光照、复杂背景及连续切换手势。
- 记录误触发、漏识别和已知易混淆场景。

## 文件说明

- `unity/ai/mediapipe/gesture_recognizer.bytes`：Unity 直接替换文件。
- `models/gesture_recognizer.task`：MediaPipe 原始模型。
- `model_manifest.json`：模型标签、版本、参数和完整性信息。
- `label_mapping.json`：模型标签到 Unity 业务动作的映射。
- `checksums.sha256`：交付文件校验值。

> 注：原训练目录没有保留可靠的测试集 loss/accuracy 记录，因此清单中明确标记为未保存，不伪造评估数值。
