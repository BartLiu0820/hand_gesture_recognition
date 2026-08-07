# Change Log

## CR-20260807-unity-minimal-delivery

- 日期：2026-08-07
- 用户请求：根据 Unity 开发确认结果，交付 ZIP 只保留 `model_manifest.json` 和 `models/` 两项。
- 路由层级：L2（功能/交付规则变更）
- 最早修正点：`03-feature-independent-gesture-training.md`
- 影响文件：功能规格、README、Unity 交付说明、模型清单和当前 ZIP。
- PM 确认状态：confirmed
- 执行状态：verified
- 备注：删除 ZIP 中的 `.bytes` 副本、标签映射、接入说明、校验文件和验收材料；Unity 工程自身管理这些接入配置。

## CR-20260806-flexible-model-labels

- 日期：2026-08-06
- 用户请求：取消官方固定类别，由操作者自行添加每个模型需要的标签；每次模型训练独立。
- 路由层级：L2（功能层变更）
- 最早修正点：`03-feature-independent-gesture-training.md`
- 影响文件：训练校验、标签 API、数据目录、模型识别、H5 页面、测试、README、Unity 交付文档和 Docker 镜像。
- PM 确认状态：confirmed
- 执行状态：verified
- 数据影响：移除 7 个由上一变更创建且始终为空的官方类别目录；保留 `none` 52 张、`ok` 20 张及全部模型。
- 备注：仅 `none` 固定且不可删除；当前数据目录中的标签构成下一次模型，已导出模型彼此独立。

## CR-20260806-unified-gesture-classifier

- Date: 2026-08-06
- Level: L2（功能层变更）
- Status: implemented
- Confirmed by: user
- Earliest correction layer: `03-feature-unified-gesture-training.md`
- Reason: 将原“自定义优先、官方兜底”的双分类器逻辑改为官方类别与自定义类别共同训练的统一分类头；Unity 不强制迁移 Top 5。
- Propagated files:
  - `train.py`
  - `web_app.py`
  - `collect_data.py`
  - `templates/index.html`
  - `static/app.js`
  - `README.md`
  - `tests/test_web_app.py`
  - `项目功能与Unity交付说明.md`
- Data impact: 保留 `none` 52 张和 `ok` 20 张；新增 7 个官方类别目录，现有图片未被删除或自动重新分类。
- Verification: 14 unit tests, Python/JavaScript syntax checks, Docker version/build check, dataset/model/status API smoke tests, 1280×720 single-page visual QA.
- Superseded by: `CR-20260806-flexible-model-labels`
