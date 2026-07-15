# 面试问答要点

以下答案追求「诚实、可讲」，不追求把项目包装得比实际更强。每个答案后括号里的文件是对应代码位置，回答时可以直接指过去。

## 1. 这个项目的价值是什么？

问题不是「AI 能不能写测试」，而是「在一个有边界、可验证的范围内，构建一个真正闭环、可解释的工程系统」。价值体现在三点：

- **闭环**：从需求到报告的每一步都有结构化的中间产物（检索结果、测试计划、执行结果），不是一个黑盒。
- **可验证**：通过 Fake LLM 把「模型可能不稳定」这个变量隔离掉，让核心逻辑（状态机、重试规则、报告构建）可以用确定性测试验证。
- **诚实的边界**：低置信度时返回「信息不足」而不是编造测试；执行器明确不是安全沙箱；这些都是主动设计出来的边界，而不是没做到。

（`README.md` 第 1–2 节、`src/drivetest_agent/agent/orchestrator.py`）

## 2. 这个项目的 RAG 是怎么做的？

严格说这是一个「轻量 RAG」：需求文本作为查询，在本地 4 份 Markdown 规范文档切分出的片段里做检索，取 Top-3 相关片段连同来源和相关度一起塞进 Prompt，让模型「有据可依」地生成测试，而不是纯靠模型的先验知识编测试。检索不涉及生成式的「检索后再生成查询」之类的高级 RAG 技巧，属于最基础的一次检索 + 一次生成。

（`src/drivetest_agent/retrieval/retriever.py`、`src/drivetest_agent/llm/prompts.py`）

## 3. 为什么用 TF-IDF，不用 Embedding？

三个理由：

1. **范围匹配**：一周、每天 1–2 小时的项目范围里，引入向量数据库和 Embedding API 会显著增加学习和调试成本，收益（在只有 4 份短文档的知识库上）并不明显。
2. **不依赖付费 API**：TF-IDF 是纯本地计算（`scikit-learn`），CI 和面试演示都不需要额外的 Embedding API Key。
3. **中文友好且可解释**：用字符 2–4 gram 的 TF-IDF（而不是按词切分），不需要额外分词器就能处理中文，而且相关度分数是可解释的余弦相似度，不是黑盒向量距离。

**代价也很清楚**：TF-IDF 是词法匹配，理解不了同义改写或语义相似（比如「制动」和「刹车」在字面上不重叠时可能检索不到），知识库规模变大后效果会明显下降，这时候才是引入 Embedding + 向量库的合理时机。

（`src/drivetest_agent/retrieval/retriever.py`，`analyzer="char", ngram_range=(2, 4)`）

## 4. Prompt 是怎么设计的？为什么要结构化输出？

Prompt 用固定的分段标记（`=== REQUIREMENT ===`、`=== KNOWLEDGE ===`、`=== PREVIOUS_TEST ===`、`=== PYTEST_ERROR ===`）把不同来源的内容隔开，降低模型把「用户需求」和「执行错误日志」混为一体的风险，并在结尾附带明确的 JSON Schema 说明（`test_cases` + `pytest_code`）。

要求结构化输出（Pydantic `TestPlan`）而不是自由文本，是因为：下游代码（执行器、报告构建）需要确定性的字段去处理，如果靠正则或字符串解析自由文本，任何模型措辞上的微小变化都可能让解析崩溃；用 Schema 校验，不合法就是不合法，可以明确地重试或报错，而不是「看起来解析成功但字段是错的」。

（`src/drivetest_agent/llm/prompts.py`、`src/drivetest_agent/domain/models.py::TestPlan`）

## 5. Agent 和 Tool Calling 在这里是怎么体现的？

这里没有用 OpenAI 原生的 `function calling`/`tool_calls` 接口，而是用一个显式的 Python 状态机模拟「Agent 决定调用工具」的效果：模型只负责生成结构化的测试计划（含 pytest 代码字符串），真正「调用工具」的动作——执行 pytest——是由编排器代码显式触发的（`test_runner(test_code)`），不是模型自己发起的函数调用请求。

这样做的好处是执行工具这一步完全在我们的控制之下（能加超时、能加黑白名单校验、能保证只执行我们期望的那一个动作），代价是如果任务需要模型动态决定「要不要调用工具、调用哪个工具」，这种硬编码的调用点就不够灵活了——那种场景才是原生 tool calling 更合适的地方。

（`src/drivetest_agent/agent/orchestrator.py::DriveTestAgent.run`）

## 6. 为什么选择单 Agent 而不是多 Agent？

当前任务的步骤是固定的（检索 → 生成 → 执行 → 最多修正一次 → 报告），不需要动态规划下一步该做什么，也不需要多个持续存在、各自维护上下文的角色。多 Agent 在这种场景下主要增加的是通信开销（角色间传消息本身要消耗 token 和延迟）和调试难度（问题出在哪个 Agent、哪一轮对话里更难定位），却没有换来实际的能力提升。

（详见 README 第 3 节；`src/drivetest_agent/agent/orchestrator.py`）

## 7. 上下文是怎么裁剪/管理的？

几条规则：

- 只传入 Top-K（默认 3）个最相关的知识片段，而不是整份文档。
- 每个片段保留「来源 + 相关度 + 片段内容」，不夹带无关的文档全文。
- 修正 Prompt 里的执行错误摘要会被截断到最多 1500 字符，只保留头部标记和**尾部**内容（`_truncate_error_summary`）——因为 pytest 输出里最有信息量的部分通常在末尾（最后的 traceback/断言差异），截断策略特意保留尾部而不是头部。
- 修正请求会复用之前的测试计划上下文，但不会重复传入无关的历史内容。

（`src/drivetest_agent/agent/orchestrator.py::_truncate_error_summary`、`src/drivetest_agent/llm/prompts.py::_format_knowledge`）

## 8. 「一次重试」具体指哪几处？为什么是一次？

项目里有两种不同性质的「重试一次」，容易被面试官追问混在一起，需要分清楚：

1. **格式重试**（`OpenAICompatibleClient.generate`）：模型返回的内容不是合法 JSON 或不满足 `TestPlan` Schema 时，带上格式提示再请求一次；仍失败则抛 `LLMFormatError` 终止。这是「同一个任务，格式不对，纠正一次」。
2. **修正重试**（`DriveTestAgent.run`）：生成的 pytest 代码第一次执行没有全部通过时，把截断后的失败摘要带给模型，请求一次代码修正，再执行一次；仍失败则停止并返回 `failed`。这是「同一个任务，结果不对，修正一次」。

两者都限定为「一次」，是为了控制成本、延迟，并避免不收敛的无限循环——这是设计规格里明确写出的边界，不是漏加了重试次数的 bug。

（`src/drivetest_agent/llm/openai_client.py`、`src/drivetest_agent/agent/orchestrator.py`）

## 9. Fake LLM 是什么？为什么需要它？

`FakeLLMClient` 是一个实现了和真实客户端相同接口（`generate(prompt) -> LLMGeneration`）的内存对象：调用前往队列里塞入固定的 `LLMGeneration` 或异常，调用时按顺序弹出，并记录收到的每一个 Prompt。它让集成测试可以**精确构造**「首次失败、修正后成功」「两次都失败」「格式非法重试后仍失败」这些场景，而不需要真的调用模型（模型的随机性会让这些场景很难稳定复现）。CI 里的所有 Agent 集成测试都基于 Fake LLM，所以 CI 完全不需要真实 API Key。

（`src/drivetest_agent/llm/fake_client.py`、`tests/agent/test_orchestrator.py`）

## 10. 怎么评测这个系统？有哪些指标？

要诚实地分两层说：

- **单次运行的可解释指标**：首次通过率、修正后通过率、修正次数、累计 token、累计耗时、最终状态（`reporting/report_builder.py`）。这些是运行时反馈给用户的信息，不是离线评测。
- **代码正确性的评测**：目前主要靠自动化测试套件（`pytest -q`）本身作为证据——覆盖检索、Schema 校验、Prompt 构建、客户端重试、执行器黑白名单、Agent 全部状态转移路径。
- **没有做的**：一个独立的、跨多个 Prompt/模型/检索策略的离线评测基准集（比如准备 50 条需求和「标准答案」测试用例，批量跑分对比）。这是设计规格里明确列出的后续演进项，当前版本没有实现，如果被问到会直接说「还没做，属于下一步」。

（README 第 12 节）

## 11. 代码执行有什么风险？怎么控制的？

风险很直接：这个系统会执行 LLM 生成的 Python 代码，理论上模型可能生成恶意或危险的代码。控制手段是一个基于 AST 的**受限执行器**，不是沙箱：

- 导入白名单（只允许 `pytest`、`math`、`drivetest_agent`）。
- 危险调用黑名单（`eval`/`exec`/`compile`/`open`/`__import__`/`getattr`/`vars`/`globals`/`locals`）。
- 危险属性/名称黑名单（`__builtins__`/`__globals__`/`__class__`/`__subclasses__` 等常见沙箱逃逸路径）。
- 执行超时（默认 5 秒）、输出截断、隔离临时目录、执行后清理。

**必须主动说清楚**：这是启发式的黑名单防御，不是形式化验证的沙箱，也没有容器/系统调用级别隔离。所以这套东西只适合执行「模型在受控 Prompt 和知识库约束下生成的代码」这种相对可信的场景，绝对不能直接套用到「执行任意公网用户提交的代码」。

（`src/drivetest_agent/tools/pytest_runner.py`、README 第 13 节）

## 12. CI/CD 是怎么配置的？

`.github/workflows/ci.yml` 在 `ubuntu-latest`（以及附加的 `windows-latest`）+ Python 3.11 上：checkout 代码、`setup-python`、`pip install -e ".[dev]"` 安装项目和开发依赖、`ruff check .` 做静态检查、`pytest -q` 跑全部测试。整个流程不设置任何密钥，也不访问任何真实/付费 API——因为所有涉及模型调用的测试都用 Fake LLM 或对客户端对象打桩，天然满足「无密钥可复现」的要求。

（`.github/workflows/ci.yml`）

## 13. 怎么理解 MCP、Skill、Harness Engineering 这些概念，跟本项目的关系？

- **MCP（Model Context Protocol）**：一种让不同 Agent/客户端以标准方式发现和调用「工具」的协议。本项目目前没有 MCP Server，是设计上有意排除的首版范围；后续演进方向之一是把受限 pytest 执行器包装成一个 MCP Tool，让其他 Agent 客户端也能复用，而不必各自重新实现执行器和黑白名单。
- **Skill**：可以理解为「给 Agent 的一份可复用操作手册」——把某类任务的最佳实践、步骤和注意事项写成结构化文档，让 Agent 在需要时读取并照做，而不是每次都靠通用推理重新摸索。本项目里的 Prompt（含 Schema 提示）和知识库文档在功能上起到类似的作用，但没有做成正式的、跨项目复用的 Skill 格式。
- **Harness Engineering**：指围绕一个 AI Agent 搭建的「外部脚手架」——工具接口、执行环境的限制、测试与验证手段、错误恢复策略——这些往往比 Prompt 本身对最终可靠性的影响更大。本项目其实就是一个小型的 Harness Engineering 实践：受限执行器 + 有限状态机 + Fake LLM 测试套件，共同构成了让一个不完全可控的 LLM 输出变得「可验证、可控制」的外部脚手架。

## 14. 什么时候真的需要多 Agent？

当出现以下任一情况时，值得认真考虑拆分为多 Agent（而不是默认这样做）：

- 任务里有明显不同的职责边界（比如「规划测试策略」和「执行并复核」需要不同的上下文、不同的工具集，混在一个 Prompt 里会让单个 Agent 的职责过载）。
- 需要长期记忆或跨会话状态（单 Agent 当前是一次性运行，不持久化）。
- 需要人工审批节点（比如生成的测试计划要先经人审核才能执行，这天然是一个需要「暂停等待外部输入」的状态，适合用 LangGraph 这类显式状态图来建模）。
- 需要多个任务并行、独立扩缩容或独立可观测性。

如果这些条件都不满足，拆多 Agent 只是增加复杂度和成本，没有实际收益——这也是本项目现在坚持单 Agent 的理由。

## 15. 自动驾驶/AEB 相关的业务边界是什么？

需要非常明确地划清楚：`domain.aeb.should_trigger_aeb` 是一个**教学/演示用的简化确定性函数**——TTC ≤ 1.5 秒、相对速度 > 0、传感器有效这三个条件的组合判断，不接入真实车辆数据、不接入 CAN 总线、不模拟传感器噪声/融合/滤波，也不代表真实量产 AEB 算法的复杂度（真实系统通常涉及多传感器融合、置信度估计、执行器响应延迟、功能安全等级要求等）。它存在的目的只是提供一个面试官能在几秒内理解的业务语境和边界条件（阈值、等号方向、异常输入），用来承载「测试生成」这个核心能力的演示，而不是一个自动驾驶算法作品。

（`src/drivetest_agent/domain/aeb.py`、`knowledge/aeb-input-constraints.md`）

## 16. 安全性怎么样？有没有想过失败案例？

**不夸大**：这个项目的「安全边界」体现在两个具体机制上——（1）未配置 API Key 时不会尝试请求，直接展示明确错误；（2）执行器对生成代码做 AST 黑白名单和超时限制。除此之外，**没有**做认证鉴权、没有做多租户隔离、没有做强安全沙箱、没有做输入内容审核（比如防止 Prompt 注入攻击这个话题，本项目没有专门的防御机制，需要如实说明这是一个已知的未覆盖风险）。

想过的失败案例包括（并且都有对应测试）：

- 模型服务不可用/网络错误 → `LLMServiceError`，终止并展示安全的中文提示，不泄露异常细节到界面（日志里会记录完整堆栈用于排查）。
- 模型响应结构异常（缺 `choices`/`message`/`content`）→ `LLMResponseError`，不重试（这类错误重试也不会自愈，直接报错更诚实）。
- 模型输出格式非法 → 重试一次，仍失败 → `LLMFormatError`，终止。
- 生成的代码包含被禁止的导入/调用 → 执行器在真正跑子进程之前就用 AST 校验拦截，返回明确的 `error_summary`。
- pytest 执行超时 → 标记 `timed_out=True`，视为一次失败结果，进入修正流程而不是卡死。
- 测试执行器本身抛异常（比如临时目录权限问题）→ Agent 捕获后返回 `error` 状态，不让未处理异常冒泡到界面。
- 检索不到相关规范 → 直接返回 `insufficient_info`，不编造测试。

这些失败路径全部在 `tests/agent/test_orchestrator.py`、`tests/llm/test_openai_client.py`、`tests/tools/test_pytest_runner.py`、`tests/ui/test_service.py` 里有对应的确定性测试，可以在追问时直接翻开对应测试类逐条讲。
