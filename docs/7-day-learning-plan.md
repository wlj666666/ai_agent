# 7 天学习计划（每天 1–2 小时）

适用对象：具备 Python 基础（变量、函数、类、异常处理、看得懂 `pytest` 断言）但**不需要**提前了解 LLM/Agent/RAG 相关概念的人。目标是 7 天后能独立运行、修改并讲解这个项目的每一处核心代码。

每天结构固定：**学习目标 → 需读文件 → 动手练习 → 当天可回答的面试问题 → 验收点**。所有命令均在 PowerShell 下、项目根目录（已完成 README 第 8 节安装）执行。

## 第 1 天：场景理解 + 模拟 AEB 组件 + pytest 基础

**学习目标**：理解项目要解决的问题；理解 `should_trigger_aeb` 的判定逻辑与输入校验；能读懂 pytest 的基本断言写法。

**需读文件**：

- `docs/superpowers/specs/2026-07-15-drivetest-agent-design.md`（第 1–3 节）
- `knowledge/aeb-input-constraints.md`、`knowledge/boundary-exception-testing.md`
- `src/drivetest_agent/domain/aeb.py`
- `tests/domain/test_aeb.py`

**动手练习**：

```powershell
pytest tests/domain -v
```

1. 逐条读 `test_aeb.py` 里的测试名，猜测每个测试在验证什么，再对照断言确认。
2. 临时把 `aeb.py` 里的 `TTC_THRESHOLD_SECONDS = 1.5` 改成 `1.4`，重新跑 `pytest tests/domain -v`，观察哪些测试变红（RED），理解「阈值两侧测试」的作用；改完记得改回 `1.5`。
3. 自己新增一个测试函数 `test_aeb_triggers_when_ttc_is_exactly_zero`，验证 `should_trigger_aeb(ttc=0.0, relative_speed=1.0, sensor_valid=True)` 返回 `True`，跑通后删除或保留都可以。

**当天可回答的面试问题**：

- 这个项目要解决什么问题？（人工写回归测试容易漏边界条件）
- 模拟 AEB 组件的触发条件是什么？为什么传感器无效时是抛异常而不是返回 `False`？（无效意味着「不知道」，不是「已知不触发」，用异常区分两种语义）

**验收点**：`pytest tests/domain -v` 全部通过；能不看代码口头说出触发条件的三个必要条件。

## 第 2 天：知识库、切分与检索

**学习目标**：理解 Markdown 知识库如何被切分成带来源的片段；理解 TF-IDF + 余弦相似度检索和低置信度判断。

**需读文件**：

- `knowledge/` 下全部 4 份文档
- `src/drivetest_agent/retrieval/chunking.py`
- `src/drivetest_agent/retrieval/retriever.py`
- `tests/retrieval/test_chunking.py`、`tests/retrieval/test_retriever.py`

**动手练习**：

```powershell
pytest tests/retrieval -v
```

在项目根目录打开一个 Python 交互式会话，实际调用一次检索器（体感相关度分数）：

```powershell
python
```

```python
from pathlib import Path
from drivetest_agent.retrieval.retriever import KnowledgeRetriever

retriever = KnowledgeRetriever(Path("knowledge"))
for ref in retriever.search("TTC 小于等于 1.5 秒 边界 阈值两侧"):
    print(ref.source, round(ref.relevance_score, 3), ref.low_confidence)

for ref in retriever.search("车机天气小组件"):
    print(ref.source, round(ref.relevance_score, 3), ref.low_confidence)
```

对比两次查询的分数差异，理解「低置信度」是怎么触发的。

**当天可回答的面试问题**：

- 为什么用 TF-IDF 而不是向量数据库/Embedding？（一周范围内足够、无需额外付费 API、字符 n-gram 对中文友好、可解释性强）
- 系统如何判断「信息不足」？（Top-K 结果里最高相关度低于阈值）

**验收点**：`pytest tests/retrieval -v` 全部通过；能解释 `low_confidence_threshold` 的作用和默认值。

## 第 3 天：模型网关、Schema 与 Prompt

**学习目标**：理解结构化输出（Pydantic Schema）为什么比直接解析自由文本更可靠；理解 Prompt 的分段结构和格式重试策略。

**需读文件**：

- `src/drivetest_agent/domain/models.py`（重点：`TestPlan`、`TestCasePlan`、`AgentReport`）
- `src/drivetest_agent/llm/protocol.py`、`llm/exceptions.py`
- `src/drivetest_agent/llm/prompts.py`
- `src/drivetest_agent/llm/openai_client.py`
- `src/drivetest_agent/llm/fake_client.py`
- `tests/llm/test_openai_client.py`、`tests/llm/test_prompts.py`、`tests/llm/test_fake_client.py`

**动手练习**：

```powershell
pytest tests/llm -v
```

```python
from drivetest_agent.domain.models import TestCasePlan, TestPlan

plan = TestPlan(
    test_cases=[TestCasePlan(name="test_x", description="d", expected_outcome="o")],
    pytest_code="def test_x():\n    assert True\n",
)
print(plan.model_dump_json(indent=2))
```

尝试把 `test_cases` 传成空列表，观察 Pydantic 校验报错（`min_length=1`），体会「结构化校验」如何在源头拦住不合法数据，而不是等到后面用到时才崩溃。

**当天可回答的面试问题**：

- 为什么要求模型输出 JSON/Pydantic Schema，而不是直接解析自然语言？（可校验、可重试、下游代码不用猜文本结构）
- 模型输出格式不合法时怎么处理？（重试一次，带上格式提示；仍失败则抛 `LLMFormatError` 终止，不无限重试）

**验收点**：`pytest tests/llm -v` 全部通过；能画出/口述一次 `generate()` 调用的重试流程。

## 第 4 天：受限 pytest 工具

**学习目标**：理解 AST 级别的导入/调用黑白名单如何工作；理解为什么这不是一个「安全沙箱」。

**需读文件**：

- `src/drivetest_agent/tools/pytest_runner.py`
- `tests/tools/test_pytest_runner.py`
- README 第 13 节「安全边界」

**动手练习**：

```powershell
pytest tests/tools -v
```

```python
from drivetest_agent.tools.pytest_runner import run_pytest

ok = run_pytest("def test_ok():\n    assert 1 + 1 == 2\n")
print(ok.exit_code, ok.passed, ok.failed)

blocked = run_pytest("import os\n\ndef test_bad():\n    os.system('echo hi')\n")
print(blocked.exit_code, blocked.error_summary)
```

观察第二次调用是如何在**不执行任何子进程**的情况下就被 AST 校验拦截并返回 `error_summary`。

**当天可回答的面试问题**：

- 受限执行器具体做了哪些防护？（导入白名单、危险调用/属性黑名单、执行超时、隔离临时目录、输出截断、执行后清理）
- 为什么它不是强安全沙箱？（没有容器/系统调用隔离/资源配额，黑名单是启发式的，不是形式化证明）

**验收点**：`pytest tests/tools -v` 全部通过；能现场说出至少 3 个被禁止的导入模块和 3 个被禁止的调用。

## 第 5 天：Agent 闭环与 Streamlit 页面

**学习目标**：理解单 Agent 有限状态机的完整流转；理解 Streamlit 页面如何调用业务层而不直接写业务逻辑。

**需读文件**：

- `src/drivetest_agent/agent/orchestrator.py`
- `src/drivetest_agent/reporting/report_builder.py`
- `src/drivetest_agent/ui/service.py`、`ui/page.py`、`ui/render.py`、`ui/view_model.py`、`ui/examples.py`、`ui/paths.py`
- `tests/agent/test_orchestrator.py`、`tests/reporting/test_report_builder.py`、`tests/ui/*.py`

**动手练习**：

```powershell
pytest tests/agent tests/reporting tests/ui -v
streamlit run app.py
```

不配置 `OPENAI_API_KEY` 直接点击「运行」，观察安全的配置错误提示；再选择一个固定案例，尝试点击「填充该案例到需求输入框」并观察文本框内容变化（即使不点「运行」也能验证界面交互）。

**当天可回答的面试问题**：

- Agent 的四种最终状态是什么？分别在什么条件下出现？（`success`/`failed`/`insufficient_info`/`error`）
- 为什么最多修正一次，而不是无限重试？（控制成本和延迟，避免不收敛；这是设计规格里的明确边界）

**验收点**：`pytest tests/agent tests/reporting tests/ui -v` 全部通过；`streamlit run app.py` 能正常启动并展示未配置密钥时的安全提示。

## 第 6 天：Fake LLM 测试、CI 与配置校验

**学习目标**：理解如何用 Fake LLM 让集成测试完全脱离真实网络；理解 CI 流水线做了什么；理解 `.env` 加载与 `RETRIEVAL_MIN_RELEVANCE` 校验的边界。

**需读文件**：

- `src/drivetest_agent/llm/fake_client.py`（第 2 次读，重点关注它如何被用在 `tests/agent/test_orchestrator.py`）
- `src/drivetest_agent/config.py`
- `tests/test_config.py`、`tests/ui/test_service.py` 中 `TestDotenvLoading` 和 `TestBuildAgent` 相关用例
- `.github/workflows/ci.yml`

**动手练习**：

```powershell
pytest -q
ruff check .
```

```python
from drivetest_agent.config import parse_retrieval_min_relevance, ConfigError

print(parse_retrieval_min_relevance("0.3"))
try:
    parse_retrieval_min_relevance("abc")
except ConfigError as exc:
    print("捕获到预期的配置错误：", exc)
```

打开 `.github/workflows/ci.yml`，找出「哪一步」保证了 CI 不需要任何密钥（提示：全程没有出现任何 `OPENAI_API_KEY` 或真实网络调用，测试本身通过 Fake LLM/打桩验证逻辑）。

**当天可回答的面试问题**：

- Fake LLM 是怎么工作的？为什么这样设计能测出真实的 Agent 逻辑？（队列式返回固定 `LLMGeneration`/异常，记录收到的 Prompt，可以精确构造「首次失败、修正后成功」等场景）
- `RETRIEVAL_MIN_RELEVANCE` 配置错误时系统怎么处理？为什么不会触发 API 调用？（在构建检索器/模型客户端之前完成解析和校验，校验失败直接抛 `ConfigError`）

**验收点**：`pytest -q` 与 `ruff check .` 本地全部通过；能解释 CI 工作流的每一步。

## 第 7 天：README、演示脚本、面试问答与完整彩排

**学习目标**：把前 6 天的技术细节整理成能在面试中流畅讲述的叙事；确认 5 分钟演示可以在时间内完成。

**需读文件**：

- `README.md`（完整重读一遍，尤其是第 3、4、13、14、15 节）
- `docs/demo-script.md`
- `docs/interview-qa.md`

**动手练习**：

1. 掐表完整走一遍 `docs/demo-script.md` 里的三个案例演示，确认能在 5 分钟内完成（包含口述部分）。
2. 故意制造一次「失败场景」（比如临时改错 `OPENAI_API_KEY`），练习演示脚本里的备用方案，确认能从容切换到 Fake LLM 测试演示。
3. 盖住答案，口头回答 `docs/interview-qa.md` 里的每一个问题；对答不上来的问题回到对应源码文件重新过一遍。

**当天可回答的面试问题**：

- 「介绍一下这个项目」——用 30 秒版本（问题、方案、你的关键工程判断）+ 2 分钟版本（架构、取舍、边界）两种长度都练一遍。

**验收点**：能在 5 分钟内不看脚本、流畅讲完三个演示案例；`docs/interview-qa.md` 中的问题都能不翻书回答出要点。

## 进度不足时的降级顺序

如果某一天时间不够，按以下顺序降级，**不要**删除标黑的部分：

1. 界面样式可以简化。
2. 检索算法可以退化为简单的关键词重叠计数（TF-IDF 是加分项，不是底线）。
3. 三个演示案例可以先只准备两个（**保留案例三「信息不足」**，它最能体现工程判断）。
4. **不得删除**：Fake LLM 集成测试、pytest 执行超时、修正一次即停止的规则、失败报告——这些是工程可信度的核心，删掉之后项目会变成「看起来能跑的 Demo」而不是「可验证的工程闭环」。
