# pytest 命名与断言规范

## 测试文件命名

- 测试文件以 `test_` 开头，如 `test_aeb_braking.py`。
- 文件名应反映被测模块，使用小写与下划线。

## 测试函数命名

- 函数名以 `test_` 开头，描述场景与预期，如 `test_aeb_triggers_when_ttc_at_threshold`。
- 边界用例命名应包含阈值信息，如 `test_aeb_does_not_trigger_when_ttc_above_1_5`。
- 异常用例命名包含 `invalid` 或 `raises`，如 `test_aeb_rejects_negative_ttc`。

## 断言规范

- 使用 `assert` 进行布尔判定，避免 `assert True` 占位。
- 触发/不触发制动：`assert result is True` 或 `assert result is False`。
- 异常输入：`pytest.raises(ValueError)` 或 `pytest.raises(TypeError)` 包裹调用。
- 每个测试函数只验证一个主要行为，保持单一职责。

## 参数化建议

对阈值两侧用例可使用 `@pytest.mark.parametrize`：

```python
@pytest.mark.parametrize("ttc,expected", [(1.49, True), (1.50, True), (1.51, False)])
def test_aeb_ttc_boundary(ttc, expected):
    ...
```

## 可读性

- 在 docstring 或注释中说明所引用的需求条款（如 TTC <= 1.5）。
- 魔法数字应使用命名常量，如 `TTC_THRESHOLD = 1.5`。
