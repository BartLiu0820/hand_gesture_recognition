# 本地 MediaPipe 手势识别 Demo

一个完全本地运行的摄像头手势识别项目，包含：

- 浏览器可视化工作台：采集、训练、日志和实时测试
- 实时摄像头识别、手部骨架、统一分类结果和真实 Top 5
- 多帧投票防抖
- 手势图片采集工具
- 基于 MediaPipe Model Maker 的独立分类头训练、`.task` 与模型清单导出
- 通过 `--model` 随时替换训练结果

## 五分钟快速开始

### 开始前准备

首次运行需要：

- Git，用于克隆项目
- 64 位 Python 3.9 或 3.10（推荐 3.10）
- 可用的摄像头，以及系统授予终端/浏览器的摄像头权限
- 首次安装 Python 依赖和下载官方模型时可访问互联网
- Chrome、Edge、Safari 等现代浏览器（使用可视化工作台时）

> 只运行、采集和测试手势时不需要 TensorFlow，也不需要 Docker。训练自定义模型才需要完整训练环境。

### macOS / Linux

```bash
git clone https://github.com/BartLiu0820/hand_gesture_recognition.git
cd hand_gesture_recognition

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python scripts/download_default_model.py
python web_app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/BartLiu0820/hand_gesture_recognition.git
cd hand_gesture_recognition

py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python scripts/download_default_model.py
python web_app.py
```

浏览器打开 <http://127.0.0.1:8765>，允许摄像头权限即可开始体验。结束服务时在终端按 `Ctrl+C`。

如果项目已经在本地，跳过 `git clone` 和 `cd` 两行。若只想使用桌面摄像头窗口，将最后一条命令换成：

```bash
python app.py
```

按 `Q` 或 `Esc` 退出，按 `R` 清空多帧识别历史。

## 应该安装哪套依赖

| 目标 | 安装文件 | 额外工具 |
| --- | --- | --- |
| 运行官方/已有模型 | `requirements-demo.txt` | 无 |
| 采集手势图片 | `requirements-demo.txt` | 摄像头 |
| 使用浏览器工作台 | `requirements-demo.txt` | 现代浏览器 |
| 在 Linux/Windows x86_64 训练 | `requirements.txt` | 建议使用 Python 3.10 |
| 在 Apple Silicon Mac 训练 | `requirements-demo.txt` | Docker Desktop，训练在 `linux/amd64` 容器中进行 |

`requirements-demo.txt` 是日常使用的轻量环境，包含 MediaPipe、OpenCV、NumPy 和 Flask。`requirements.txt` 会进一步安装 TensorFlow 和 MediaPipe Model Maker，下载量、安装时间及磁盘占用都会明显增加。

## 环境与兼容性

- 64 位 Python 3.9 或 3.10（推荐 3.10）
- macOS / Windows / Linux
- 摄像头

训练环境严格固定：

```text
mediapipe-model-maker==0.2.1.4
```

`train.py` 启动时也会校验已安装版本；不匹配时会拒绝训练，防止产物因环境漂移而不可复现。

平台注意事项：

- macOS 首次启动会弹出摄像头授权；如果曾拒绝，请到“系统设置 → 隐私与安全性 → 摄像头”重新开启。
- Windows 如果禁止执行虚拟环境激活脚本，可以先执行 `Set-ExecutionPolicy -Scope Process Bypass`，只对当前 PowerShell 会话生效。
- Linux 需要图形桌面和可用的 `/dev/video*` 摄像头设备。若 OpenCV 报缺少 `libGL.so.1`，在 Debian/Ubuntu 上安装 `libgl1` 和 `libglib2.0-0`。
- Web 服务只监听本机 `127.0.0.1:8765`。若端口已被占用，请先结束占用该端口的进程。

## 可视化工作台（推荐）

启动本地 Web 服务：

```bash
source .venv/bin/activate
python web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

页面支持：

- 保留 `none` 背景类，按每个模型的需要自行创建手势标签
- 查看和删除数据集样本
- 检查并准备固定版本训练环境
- 配置参数、启动训练、查看实时日志和进度
- 使用每次训练生成的独立模型进行最终判定；官方模型作为独立基线
- 展示同一模型分类头内的真实置信度 Top 5

数据和模型均保存在当前项目目录，页面不会把摄像头图片上传到外部服务。

> Model Maker 会安装 TensorFlow 等较大的训练依赖。只想运行 Demo 时，可以先使用轻量的 `requirements-demo.txt`；要训练时再使用 `requirements.txt`。

## 1. 详细安装

仅运行摄像头 Demo：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

需要训练自定义模型（Linux / Windows x86_64）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import importlib.metadata as m; print(m.version('mediapipe-model-maker'))"
```

### Apple Silicon 训练说明

`mediapipe-model-maker==0.2.1.4` 依赖 TensorFlow Text 2.13～2.15，而这些版本没有原生 macOS ARM64 或 Linux ARM64 wheel。因此 M 系列 Mac 可以原生运行和采集，但完整训练环境应使用项目提供的 Linux x86_64 容器（Docker Desktop 会自动模拟）；Model Maker 版本仍严格为 `0.2.1.4`：

```bash
docker build --platform linux/amd64 -f Dockerfile.train -t gesture-trainer:0.2.1.4-independent .
docker run --rm --platform linux/amd64 \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/exported_model:/workspace/exported_model" \
  gesture-trainer:0.2.1.4-independent \
  --data data/gestures --export-dir exported_model --epochs 20
```

若机器没有 Docker，可在 Linux Python 3.10 环境中直接安装 `requirements.txt`。`constraints-train.txt` 固定了 TensorFlow、MediaPipe、NumPy 等易发生漂移的间接依赖。

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 2. 运行内置手势 Demo

先下载 Google 官方预训练模型：

```bash
python scripts/download_default_model.py
```

再启动摄像头：

```bash
python app.py
```

按 `Q` 或 `Esc` 退出，按 `R` 清空多帧识别历史。首次启动时 macOS 会请求摄像头权限。

常用参数：

```bash
python app.py --camera 1 --num-hands 1 --min-gesture-confidence 0.65
python app.py --model /绝对路径/gesture_recognizer.task
```

内置模型支持 `Closed_Fist`、`Open_Palm`、`Pointing_Up`、`Thumb_Down`、`Thumb_Up`、`Victory` 和 `ILoveYou`。

## 3. 采集自己的训练数据

数据目录必须是 `<数据目录>/<标签>/<图片>`。每次训练只要求小写 `none` 和至少 1 个由你定义的目标手势：

```text
data/gestures/
├── none/
├── ok/
└── five/
```

逐类采集；按空格保存图片：

```bash
python collect_data.py --label none
python collect_data.py --label ok
python collect_data.py --label five
```

也可以每 0.3 秒自动采一张，采满 200 张停止：

```bash
python collect_data.py --label ok --interval 0.3 --limit 200
```

建议每类至少准备 100～300 张，覆盖不同人员、左右手、距离、角度和光照。`none` 只放不属于当前模型全部目标类别的其他动作，不放空画面。需要官方手势时手动创建对应标签；不需要时无需添加。

当前数据目录中的全部标签会共同组成下一次训练的模型。之后增删标签并再次训练，会生成标签集合不同的新模型；已导出的旧模型保持不变。

## 4. 训练并替换模型

```bash
python train.py --data data/gestures --epochs 20 --batch-size 8
```

默认产物：

```text
exported_model/gesture_recognizer.task
exported_model/model_manifest.json
```

直接加载训练结果：

```bash
python app.py --model exported_model/gesture_recognizer.task
```

也可以在训练完成后自动复制到 Demo 默认位置：

```bash
python train.py \
  --data data/gestures \
  --export-dir exported_model \
  --install-model models/gesture_recognizer.task
```

以后替换到 Android、iOS、Web 或其他使用 MediaPipe Gesture Recognizer Tasks API 的地方时，部署的也是这个 `.task` 文件。

Unity 交付 ZIP 采用最小结构，根目录只包含两项：

```text
model_manifest.json
models/
└── gesture_recognizer.task
```

`.bytes` 副本、标签映射、接入说明和验收材料不再放入模型 ZIP，由 Unity 工程或协作流程单独管理。

完整的当前功能、标签规则和 Unity 交付清单见 [项目功能与 Unity 交付说明](项目功能与Unity交付说明.md)。

## 5. 目录与本地产物

```text
app.py                    桌面摄像头识别入口
web_app.py                浏览器可视化工作台入口
collect_data.py           命令行图片采集工具
train.py                  自定义模型训练入口
gesture_demo/             识别结果防抖等公共逻辑
scripts/                  模型下载等辅助脚本
tests/                    无需摄像头即可运行的单元测试
models/                   Demo 默认模型目录
data/gestures/            本地训练图片
exported_model/           本地训练输出
deliverables/             Unity 交付包和说明
```

`data/`、`exported_model/`、`.venv/` 以及 `models/*.task` 已被 Git 忽略。它们可能包含个人摄像头图片、大体积依赖或本地模型，不会在正常提交中上传。新环境需要执行 `python scripts/download_default_model.py` 获取官方 Demo 模型。

## 6. 本地检查

无需摄像头和模型即可运行的单元测试：

```bash
python -m unittest discover -s tests -v
```

检查所有 Python 文件语法：

```bash
python -m compileall app.py collect_data.py train.py gesture_demo tests
```

## 常见问题

### 提示 `Model not found`

执行：

```bash
python scripts/download_default_model.py
```

也可以使用 `python app.py --model /绝对路径/gesture_recognizer.task` 指定已有模型。

### 无法打开摄像头

先检查系统摄像头权限，再尝试其他设备编号：

```bash
python app.py --camera 1
```

同时关闭 Zoom、微信会议等可能独占摄像头的程序。

### Apple Silicon 无法安装训练依赖

这是 TensorFlow Text 相关 wheel 的平台限制，不是项目代码错误。请保留原生轻量环境用于运行和采集，并按上面的 Docker 命令在 `linux/amd64` 容器中训练。
