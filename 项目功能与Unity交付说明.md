# Gesture Lab 项目功能与 Unity 交付说明

## 1. 项目定位

Gesture Lab 用于在本机完成手势样本采集、模型训练、实时验收和 Unity 交付。

每次训练的模型彼此独立。系统不强制加入 MediaPipe 官方 7 个手势；操作者只需创建当前模型真正需要的类别。例如：

```text
模型 A：none + ok
模型 B：none + one + two + five
模型 C：none + Open_Palm + Victory + ok
```

三个模型有各自独立的类别空间和置信度，互不继承。当前数据目录中的全部标签构成下一次训练的模型；增删标签后再次训练会生成新的模型文件，已经导出的旧模型不会改变。

## 2. 标签规则

### 2.1 必需标签

系统只固定一个标签：

```text
none
```

`none` 是 MediaPipe Model Maker 必需的背景类，不可删除。它只包含不属于当前模型全部目标类别的其他动作，不应包含没有手的空画面。

小写 `none` 是有意设计。MediaPipe 标准组合图会把精确的 `None` 解释为切换到 canned 官方分类器的信号；使用小写 `none` 可以让当前独立分类头的背景结果保持权威，并允许本地读取真实背景分数。

### 2.2 用户创建标签

除 `none` 外不预置任何类别。标签由操作者自行输入，支持字母、数字、下划线和短横线，长度 1～40 个字符。

标签大小写会保留并成为模型原始返回值。例如输入 `Open_Palm`，Unity 收到的也是 `Open_Palm`；输入 `open_palm`，返回值则是 `open_palm`。仅 `none` 会统一转成小写。

若希望模型识别官方手势，需要手动创建对应标签并采集样本；未加入的官方手势不会被当前独立模型识别。

## 3. 当前页面功能

本地服务地址：

```text
http://127.0.0.1:8765/
```

页面提供三个单页工作面板：

### 3.1 采集数据

- 创建和删除当前模型的目标标签。
- 选择标签后单张或连续采集摄像头图片。
- 查看标签样本数、采集进度和最近图片。
- 删除错误样本。
- `none` 自动创建并禁止删除，其他标签均可管理。

### 3.2 训练模型

- 严格使用 `mediapipe-model-maker==0.2.1.4`。
- Apple Silicon 通过 Linux AMD64 Docker 镜像训练。
- 配置模型名、epochs、batch size、learning rate 和 dropout。
- 实时查看训练进度与日志并支持取消。
- 当前所有标签至少 10 张、且存在目标类别时允许训练。
- 每次训练输出独立 `.task` 和模型清单。

10 张只是流程最低门槛。正式训练建议每类 100～300 张，覆盖不同人员、左右手、距离、角度、光线和背景。

### 3.3 实时测试

- 自动读取每个 `.task` 的实际标签。
- 含 `none` 和至少一个目标类别的模型标记为“独立分类模型”。
- 显示当前模型最终判定、真实 score 和 Top 5。
- 绘制手部关键点。
- 单独展示官方预训练模型基线，但不参与当前模型判定或 Top 5。

Top 5 中每条 score 都来自同一个模型分类头，可以直接比较，不会混合不同分类器的分数。

## 4. 实际训练范围

项目不会重新训练完整 MediaPipe 视觉管线。Model Maker 继续复用官方：

- 手掌检测器（palm detector）
- 手部关键点模型（hand landmarker）
- 手势嵌入器（gesture embedder）

每次重新训练的是最后的手势分类头。某个官方手势只有在被加入当前数据集时，才需要为它采集样本并参与该模型的分类头训练。

## 5. 固定训练环境

依赖版本严格锁定：

```text
mediapipe-model-maker==0.2.1.4
```

Apple Silicon 训练镜像：

```text
gesture-trainer:0.2.1.4-independent
```

训练脚本会读取实际安装版本，不是精确的 `0.2.1.4` 就拒绝训练。

## 6. 训练产物

每次成功训练生成独立目录：

```text
exported_model/<模型名-时间>/
├── gesture_recognizer.task
└── model_manifest.json
```

模型清单记录：

- 架构标识 `independent_gesture_classifier`。
- 本次模型的完整标签顺序。
- 必需背景标签和目标标签快照。
- Model Maker 精确版本。
- 训练参数、数据切分和随机种子。
- 测试集 loss 与 accuracy。
- `.task` 文件 SHA-256。
- 复用的官方模型组件。
- `unity_top5_required: false`。

## 7. Unity 交付内容

根据 2026-08-07 与 Unity 开发确认的结果，每个模型建立独立的最小 ZIP 交付包。ZIP 根目录只包含两项：

```text
model_manifest.json
models/
└── gesture_recognizer.task
```

### 7.1 交付项

1. `model_manifest.json`：模型清单，记录模型相对路径、标签、版本、训练参数、文件大小和 SHA-256。如旧训练产物未保存可靠的测试指标，loss/accuracy 必须写为 `null` 并注明未保存，不得猜测。
2. `models/gesture_recognizer.task`：当次正式 MediaPipe 手势模型。

### 7.2 不再放入 ZIP 的内容

- `gesture_recognizer.bytes` 副本。
- `label_mapping.json`。
- `README_Unity.md`。
- `checksums.sha256`。
- 验收图片、视频和验收报告。
- 训练权重、checkpoint、epoch 模型和日志。

上述内容如仍有协作需求，在 Unity 工程、验收流程或其他文档中单独管理，不再作为模型 ZIP 交付物。不同模型的 `.task` 和 `model_manifest.json` 不可混用。

### 7.3 当前 `ok-20260806-134328` 交付状态

- 已生成 `deliverables/unity_gesture_ok_20260806.zip`。
- ZIP 根目录只有 `model_manifest.json` 和 `models/` 两项。
- `models/` 中只有 `gesture_recognizer.task`。
- 原训练目录未保存可靠的测试集 loss/accuracy，交付清单中已以 `null` 和说明文字如实记录。

## 8. Unity 接入要求

### 8.1 模型替换

开发从交付包的 `models/gesture_recognizer.task` 取得模型。模型在 Unity 工程中的最终路径、扩展名和资源打包方式由 Unity 开发维护，不在此 ZIP 中额外提供 `.bytes` 副本。

### 8.2 必须实现

Unity 必须：

- 使用当前模型自定义分类头的结果作为业务判定。
- 消费原始字符串标签，不能只依赖官方手势枚举。
- 使用该模型自己的 `model_manifest.json` 核对标签和模型完整性；业务标签映射由 Unity 工程自身管理。
- 将小写 `none` 作为无业务动作。
- 保留 MediaPipe 返回的真实 score，不伪造、缩放或跨模型替换。
- 对未知标签记录日志，不能静默映射为 `none`。

### 8.3 Top 5 边界

本次不强制 Unity 迁移 Top 5。最低交付要求只有：

- 当前模型最终原始标签。
- 该标签对应的真实 score。

如果以后 Unity 需要 Top 5，所有候选必须来自当前模型的同一个分类头，不能把 canned 官方分类器候选与独立模型候选混排。

## 9. Unity 验收清单

- 模型 SHA-256 与清单一致。
- Unity 加载的标签集合与清单完全一致。
- 每个目标标签都能以原始字符串返回。
- `none` 不触发业务动作。
- 阈值处理前的 score 与 MediaPipe 原始值一致。
- 未加入当前模型的手势不会被误认为“官方自动支持”。
- 模型替换时，Unity 工程内的标签映射配置与清单标签保持一致。
- 左右手、远近、复杂背景和连续切换手势均通过测试。
- 未接入 Top 5 不影响本次交付验收。

## 10. 当前数据状态

当前本地训练数据：

- `none`：52 张。
- `ok`：20 张。

两类都已达到页面的最低训练门槛，因此当前可以训练一个独立的 `none + ok` 模型。若要训练其他模型，可按需要新增或删除目标标签；已导出的旧模型仍会保留在 `exported_model` 中。
