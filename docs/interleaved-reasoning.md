# 边听 / 边说 / 边读时也做 Reasoning：做法整理

> 整理时间：2026-09。承接「边听边想」文献地图，把当前把 **reasoning 叠进流式交互** 的做法收成一张对照表。
> 核心问题不是「要不要 CoT」，而是：**在信息不完整、时钟还在走的时候，把思考放进哪一段空窗、用什么载体、何时更新、怎么训练。**

## 1. 为什么要「也做」reasoning

标准大模型推理是 **batch thinking**：输入全部到齐 → 生成完整 CoT → 再给答案/开口说话。

对实时语音这会直接变成用户可见延迟：

```
听完整句  →  想完一整条 CoT  →  才开始说
                 ↑
           用户在等的这段
```

人在对话里不是这样：听的时候已经在准备，说的时候还可以继续想。近期工作把这段重叠做成三类时间窗：

| 时间窗 | 英文 | 空窗从哪来 | 典型约束 |
|--------|------|------------|----------|
| 边听边想 | Thinking while listening | 用户还在说，这段墙钟时间可拿来算 | 证据不完整；短问句窗口太短 |
| 边说边想 | Thinking while speaking | GPU 出 token 远快于音频播放 | 推理链太长会卡顿/口吃 |
| 边读边想 | Thinking while reading | 文本/上下文正在流式到达 | 必须因果，不能偷看未来 token |

三者可以叠：先在听的窗口里预热状态，endpoint 后再用播放窗口把剩余 CoT 藏进去。综述 [A Survey of Audio Reasoning](https://arxiv.org/abs/2605.21008) 把前两类写成 Audio-to-Speech 实时推理的主分类。

## 2. 五个设计轴（比「是不是边听边想」更有用）

### 2.1 何时开始想（trigger）

| 策略 | 做法 | 代表 |
|------|------|------|
| 固定节拍 | 每 N 秒 / 每 chunk 必想一段 | SHANKS（约 4s） |
| 语义饱和 | 前缀信息够了才触发一次或多次 | Shih QC \(\zeta(p)\)；LTS Dynamic Semantic Trigger |
| 可学习控制 | 在 `wait / think / answer` 上做策略 | wait-think-answer + DAPO |
| 听完立刻停 | 思考只摊在 listening window，endpoint 切断 | Chronological Thinking |
| 说的时候才想 | 听的阶段仍等完整句 | STITCH、Mini-Omni-Reasoner、MPS |

负例：**WordShift**（固定提前 N 个词开 CoT）在 Shih 文里系统性差于 QC。

### 2.2 思考载体（representation）

| 载体 | 可解释 | 可修订 | 延迟 | 代表 |
|------|--------|--------|------|------|
| 自由文本 CoT（不合成语音） | 高 | 难（token 已写下） | 占解码步 | Shih、SHANKS、STITCH |
| 结构化节点（entity / intent / action / knowledge / logic） | 中 | 按节点更新 | 可截断 | Chronological Thinking |
| 短状态外化（多次小更新） | 中 | 后续 think 可覆盖 | 把长 CoT 摊薄 | wait-think-answer |
| 隐式 latent / 隐状态回灌 | 低 | 连续修订更容易 | 几乎 0 | FLAIR |
| 双模块：想的脑 vs 说的脑 | 中 | 想的模块可持续写 | 开口可很早 | MPS、LTS Thinker/Speaker |

### 2.3 更新粒度

- **一次**：语义拐点后写一条 CoT（Shih）。
- **每 chunk**：音频块到了就想（SHANKS）。
- **多步控制**：同一句话上反复 wait/think（wait-think-answer）。
- **token 级交织**：每个回复 token 前紧跟支撑它的 silent reasoning（Mini-Omni-Reasoner）。
- **播放驱动的块交织**：说一块、想下一块（STITCH）。

### 2.4 系统形态

- **端到端 Speech / Omni LLM**：Moshi 三流（用户音频 / 系统音频 / text monologue）；Qwen2.5-Omni Thinker–Talker。
- **级联 agent**：ASR 流 + 语义触发 + 后台 Thinker + 前台 Speaker（LTS-VoiceAgent）。
- **双脑并行**：Formulation Brain 持续产出 think segments，Articulation Brain 边说边吃（MPS）。
- **全双工 SDLM + 专家蒸馏**：非因果 Global-aware Expert 只在训练时提供后验（FLAIR）。
- **文本流并行 KV**：输入编码与 CoT 生成解耦（StreamingThinker）。

### 2.5 怎么训练

| 手段 | 用在哪 |
|------|--------|
| SFT 左移 / 交织轨迹 | 把 CoT 塞进 ASR 空位、chunk 交替、token 交错 |
| DPO | Shih：Correctness-DPO（改主意）+ Length-DPO（压 endpoint 后延迟） |
| GRPO 族 / DAPO | wait-think-answer：正确性、动作合法、更新时机、延迟、推理质量、链路一致 |
| ELBO + 非因果专家 | FLAIR：不需要显式 CoT 标注 |
| 规则奖励的交错推理 RL | Apple Interleaved Reasoning：中间步正确就给奖，压 TTFT |

## 3. 方法卡片

### A. 边听边想

#### 1. Can Speech LLMs Think while Listening?（Shih et al., ICLR 2026）

- **arXiv:** [2510.07497](https://arxiv.org/abs/2510.07497)
- **底座：** Moshi 多流；text monologue 上交织 streaming ASR 与 CoT；`<switch_cot>` / `<switch_asr>`。
- **触发：** Question Completeness \(\zeta(p)\)，用 streaming ASR 分布相对完整句的 KL 当「问题说完了没」的进度条，阈值约 0.95。
- **训练：** SFT 把 CoT 左移到拐点；Correctness-DPO + Length-DPO。
- **结果要点：** 文本 CoT 相对无 CoT 口语推理约 **2.4×**；Length-DPO 可把 endpoint 后延迟压约 **70%**。QC 优于 WordShift。
- **局限：** 多为 TTS 题；QC 非严格单调；开太早要靠 DPO 纠偏。

#### 2. SHANKS: Simultaneous Hearing and Thinking（Chiang et al., ACL 2026）

- **arXiv:** [2510.06917](https://arxiv.org/abs/2510.06917) · [项目页](https://d223302.github.io/SHANKS/)
- **做法：** 固定时长音频块（文中约 4s）→ 一段 **unspoken** `<think>` CoT，基于已听内容不断改写内部状态。
- **推理之外的用处：** (1) 用户讲题讲错时 **打断**（比不思考的打断准约 **37.1%**）；(2) 用户没说完就 **提前 tool call**（约 **56.9%** 的调用发生在 turn 结束前）。
- **定位：** 最「直给」的边听边想：不问何时开始，块到了就想。适合需要中途动作的 agent。

#### 3. Learning When to Think While Listening（Song et al., 2605.27190）

- **arXiv:** [2605.27190](https://arxiv.org/abs/2605.27190)
- **做法：** 把流式推理做成 **wait / think / answer** 控制器；决策前只能 wait 或 think，endpoint 后再 final-think + answer。可见的短状态会累积进后续决策。
- **底座：** Qwen2.5-Omni-7B；数据约 7.5 万条对齐轨迹，SFT 后再 **DAPO**。
- **奖励（六项）：** 答案正确、动作合法、更新时机、延迟同步、思考质量、链路一致。只优化最终答案会学成「一直 wait，把思考全堆到用户可见延迟」。
- **结果：** SRQA 加权准确率 67.6% → 70.3%，endpoint 后 final-think 从 10.44 → 8.99 token。真人录音 Real Audio Bench 上 SFT 准度最好，六奖励 DAPO 是唯一把 final-think 压到低于 base 的学得变体。
- **定位：** 回答的是 **何时外化中间思考**，不是再发明一种 CoT 格式。

#### 4. Chronological Thinking（Full-duplex SDLM, 2510.05150）

- **arXiv:** [2510.05150](https://arxiv.org/abs/2510.05150)
- **做法：** 全双工听的阶段不再预测 silence token，改成按 ACT-R 风格的 **类型化思考节点**（entity, intent, action, knowledge, logic），贴在话语的语义片段上。
- **延迟合同：** 思考摊在 listening window；用户停口立刻停想、开口，**不额外加延迟**。
- **定位：** 轻量会话认知，不是解 GSM8K 的长 CoT。适合任务型对话 / 开放 QA，不适合「短问题深推理」。

#### 5. FLAIR / The Silent Thought（2603.17837, ICML 2026）

- **arXiv:** [2603.17837](https://arxiv.org/abs/2603.17837)
- **做法：** 听的时候把上一步 **latent embedding 递归喂回**，隐状态本身累积推理；训练用非因果 Global-aware Expert + **ELBO**，不需要写出来的 CoT。
- **卖点：** 前向本身就在更新，**推理延迟约 0**；后续词否定前面时，连续状态比离散文本更好改。
- **代价：** 思考不可读，难做 tool 触发 / 打断解释。

#### 6. LTS-VoiceAgent（2601.19952）

- **arXiv:** [2601.19952](https://arxiv.org/abs/2601.19952)
- **做法：** 级联 Listen–Think–Speak。Dynamic Semantic Trigger 滤掉 "hum..." 这类无信息片段；Dual-Role Orchestrator：后台 Thinker 维护可修订状态，前台 Speaker 投机生成。
- **评测：** VERA、Spoken-MQA、BigBenchAudio，以及带口误/停顿的 Pause-and-Repair。
- **定位：** 工业级联管线里把「何时想」和「怎么增量想」拆开，而不是改一个端到端 SLM。

### B. 边说边想

#### 7. STITCH（ICLR 2026）

- **arXiv:** [2507.15375](https://arxiv.org/abs/2507.15375) · [项目页](https://d223302.github.io/STITCH/)
- **做法：** 未出声 reasoning chunk ↔ 口语 response chunk 交替。播放第 N 块音频时，GPU 空闲去生成下一段 unspoken CoT，再条件生成第 N+1 块语音。
- **结果：** 延迟对齐「根本不写 CoT」的基线，数学推理约 **+15%**；非推理集不明显掉点。
- **风险：** chunk 大小绑硬件；推理比播放慢就会卡；也可能把太多推理「说出来」拉长总时长。

#### 8. Mini-Omni-Reasoner（2508.15827）

- **arXiv:** [2508.15827](https://arxiv.org/abs/2508.15827)
- **做法：** **token 级** thinking-in-speaking：silent reasoning 与 spoken token 按固定局部比例交织（文中示例约 2:8），每个回复 token 前必须有支撑它的思考。Thinker–Talker：只有 response token 进 Talker。
- **数据：** SPOKEN-MATH-PROBLEMS-3M。
- **结果：** Spoken-MQA 算术 **+19.1%**、上下文理解 **+6.4%**，输出更短，解码延迟按设计为 0。

#### 9. Mind-Paced Speaking, MPS（2510.09592）

- **arXiv:** [2510.09592](https://arxiv.org/abs/2510.09592)
- **做法：** Formulation Brain 持续产 think segments，Articulation Brain 用历史+当前 think 生成当前话；开口不必等完整 CoT。
- **结果：** 零延迟配置下 Spoken-MQA **92.8%**，URO-Bench **82.5**；声称可接近「先想完再说」同时大幅降延迟。
- **定位：** 真正的双模块并行，不是单序列里切模式。

#### 10. ReEmpathy

- 同一套「说一块、想一块」，但想的是 **共情质量自评**，用来改下一句，而不是算术推导。说明边说边想不限于 math CoT。

### C. 文本侧对照（同一套「也做」逻辑）

#### 11. StreamingThinker（2510.17238, ICLR 2026）

- **arXiv:** [2510.17238](https://arxiv.org/abs/2510.17238)
- **做法：** 边读边想。streaming attention mask（只能看已到输入）+ 输入/推理独立位置编码 + **并行 KV cache**（编码与 CoT 解耦）。读完后还可加深推理。
- **结果：** 相对 batch thinking，等待输入再开想的 token 等待约 **-80%**，出最终答案的时间级延迟约 **-60%**，准确率基本持平。
- **和交错文本的差别：** 单 cache 里交替仍是串行；并行 cache 才是真并发。

#### 12. Apple Interleaved Reasoning（2025）

- [技术说明](https://machinelearning.apple.com/research/interleaved-reasoning)
- RL（PPO / GRPO / REINFORCE++）训练 **边想边答** 多跳问答；中间步规则奖励。TTFT 平均降 **80%+**，Pass@1 最高约 **+19.3%**。说明「交错」不一定要语音。

#### 13. When to Think, When to Speak（SxS, 2605.03314）

- **arXiv:** [2605.03314](https://arxiv.org/abs/2605.03314)
- 单流自回归里把 **何时对外披露** 当成可学习决策：先 SFT 对齐「有依据才说」，再 RL 把推理质量捞回来。避免为了 TTFT 输出无信息 filler。

## 4. 对照总表

| 方法 | 时间窗 | 触发 | 载体 | 训练 | 延迟怎么藏 | 额外能力 |
|------|--------|------|------|------|------------|----------|
| Shih et al. | 听 | QC 一次 | 文本 CoT | SFT + DPO | 左移到用户说话期间 | — |
| SHANKS | 听 | 固定 chunk | unspoken CoT | SFT 类 | 听的时候写状态 | 打断、提前 tool |
| wait-think-answer | 听 | 学得 wait/think | 短状态链 | SFT + DAPO | 把长 think 摊到听的阶段 | 时机可检视 |
| Chronological Thinking | 听 | 连续/可截断 | 类型节点 | 替换 silence | 思考不得超出听窗 | 全双工对话 |
| FLAIR | 听 | 每步 latent | 隐状态 | ELBO + 专家 | 无额外解码 | 易修订、不可读 |
| LTS-VoiceAgent | 听(+说) | 语义触发 | 后台状态表 | 级联编排 | Thinker/Speaker 并行 | 口误修复 |
| STITCH | 说 | 播放节拍 | CoT chunk | 交织 SFT | 播放空闲算下一段 | — |
| Mini-Omni-Reasoner | 说 | token 比例 | silent tokens | 局部对齐数据 | 和语音同序列 | — |
| MPS | 说 | 双脑持续 | think segments | 双模块 | 先开口再补想 | 接近完整 CoT 准度 |
| ReEmpathy | 说 | chunk | 反思 token | 交织 | 同 STITCH | 情感校准 |
| StreamingThinker | 读 | 输入单元 | 流式 CoT | mask + 并行 KV | 读的时候就开始想 | 读完可加深 |
| Apple Interleaved | 答 | RL 学交错 | 中间答案 | PPO/GRPO | 先吐部分答案 | 多跳 QA |
| SxS | 答 | 学得披露策略 | 私有推理+公开前缀 | SFT + RL | 有依据才外泄 | 防 filler |

## 5. 延迟与正确性：各自会在哪死

**边听边想**

- 优点：endpoint 后可以几乎立刻开口。
- 死法 1：用户话很短、题很难（「三七二十一之后再乘…」还没说完窗口已经不够）。
- 死法 2：最后几个词改条件，前面已经写死一条错误 CoT（离散文本尤其痛；FLAIR 的 latent 略好改）。
- 死法 3：每 chunk 都想 → 算力浪费 + 被未完成句带偏。

**边说边想**

- 优点：不依赖用户话有多长；听的阶段仍可用完整句。
- 死法：推理吞吐量 < 播放速度 → 停顿；或把推理说漏嘴，总时长反而更长。
- 听的阶段默认仍是「听完再开」，**不能**单独解决「用户还在说就要打断 / 调工具」。

**选型很短的启发式**

| 目标 | 更顺的做法 |
|------|------------|
| 实时对话、要打断、要提前 function call | SHANKS；或 LTS 级联 |
| 压 endpoint 后静默、题是一次说完的推理题 | Shih QC + Length-DPO；或 wait-think-answer |
| 全双工、不能加延迟、思考宜浅 | Chronological Thinking；要零开销则 FLAIR |
| 数学口播、允许先听完 | STITCH / Mini-Omni-Reasoner / MPS |
| 文本流、长文档边读边想 | StreamingThinker |
| 已有级联 ASR–LLM–TTS，不想换底座 | LTS-VoiceAgent |

更完整的系统几乎一定是 **听窗预热 + 说窗把剩余 CoT 藏起来 + 错误假设可回滚**。综述也把这写成下一步：需要一个 runtime scheduler，根据「问题是否够完整」和「播放 buffer 还剩多少」切策略。

## 6. 和「普通 reasoning」的关系（避免混成一篇 R1 综述）

下面这些 **不是**「边听边想」，但经常被一起叫进来：

- **听完再想：** 标准 CoT / Audio-CoT / Audio-Reasoner / Step-Audio-R1。准，但延迟在用户侧。
- **Test-time compute：** o1 / R1 / GRPO / DAPO（作为算法，DAPO 也被 wait-think-answer 拿来优化 **时机**，不只优化最终答案）。
- **Agent 工具：** SHANKS 已经表明 unspoken CoT 可以驱动提前 tool call；Stream RAG、AuTAgent 是同一「先动再等完整信息」谱系。

本文只收 **把思考和流式时间轴重叠** 的做法。

## 7. 还没被好好做的点

1. **过早承诺的恢复：** 离散 CoT 写错之后怎么廉价作废，而不是硬着头皮说下去。
2. **混合调度：** 同一会话里听窗 / 说窗 / 外部文本 backend 怎么切。
3. **评测协议：** TTS 合成题 vs 真人语音；SRQA / Spoken-MQA / Real Audio Bench / Pause-and-Repair 之间几乎不能直接比数字。
4. **延迟定义不统一：** post-endpoint token 数、墙钟 TTFT、用户说话时长是否算进计算预算。
5. **Serving：** 多数论文用 full-prefix 重放模拟「cache 里持续听」；Qwen Omni 等官方 serving 往往没有 controller 式 KV 接口。
6. **可解释 vs 零延迟：** FLAIR 和 SHANKS 几乎是两个极端。

## 8. 文献入口

- 音频推理综述（含实时两类）：[2605.21008](https://arxiv.org/abs/2605.21008)
- 边听边想项目动画：<https://d223302.github.io/SHANKS/>
- 边说边想项目动画：<https://d223302.github.io/STITCH/>
- StreamingThinker 代码：<https://github.com/EIT-NLP/StreamingLLM>
