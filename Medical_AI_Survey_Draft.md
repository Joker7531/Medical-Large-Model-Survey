# 医疗人工智能与医疗大模型研究综述（2022–2026）

> **撰写日期**：2026-05-12
> **文献范围**：纳入 2026-04-30 以前公开发表或可信预印本
> **检索口径**：以 *Nature / Nature Medicine / NEJM AI / Nature Communications / JAMA Network / npj Digital Medicine* 以及 NeurIPS 等顶会顶刊为主，辅以方向性技术报告（Med-Gemini、Med-PaLM、LLaVA-Med 等）

---

## 摘要

过去四年，医疗人工智能（Medical AI）经历了从"任务专用深度学习"到"通用基础模型"的范式转变。以 GPT-3.5 / 4、PaLM 2、LLaMA 与 Gemini 系列为代表的大语言模型（LLM）与多模态大模型（LMM）将医学问答、影像报告生成、临床决策支持与患者沟通统一在生成式框架之下，并不断刷新 USMLE、MultiMedQA、MultiMedBench 等基准的纪录。然而，与此同时多项随机对照试验（RCT）与系统综述揭示了一个反直觉的现实：**模型在考试题上接近满分，并不意味着在真实临床工作流中具有相同的可靠性**[^1][^2]。本文以六个主题——技术演进、医疗大语言模型、多模态医疗大模型、评估体系、临床应用与人机协作、安全与伦理治理——对该领域进行系统梳理，并基于已识别的研究空白讨论未来方向。综述强调三条主线：(1) 数据多样性、领域微调与推理增强（如检索增强、自博弈、智能体）共同决定了医疗大模型的能力上限；(2) 评估体系正从纯文本基准走向覆盖真实 EHR、患者交互与成本-收益分析的整体性评估（MedHELM、TRIPOD-LLM）；(3) 临床落地的核心瓶颈不再是"模型够不够强"，而是"人机交互够不够安全"。

---

## 1. 引言

医疗 AI 的发展可大致划分为三个阶段[^3][^4]：

1. **任务专用监督学习时代（约 2015–2020）**——以 Gulshan 等糖尿病视网膜病变筛查、Esteva 等皮肤癌分类、McKinney 等乳腺癌筛查为代表。这一阶段已在 FDA 批准 500+ AI/ML 医疗器械中体现成熟，但模型几乎全部针对单一任务、单一模态[^3]。
2. **基础模型/多模态融合时代（2020–2023）**——Transformer、对比学习（CLIP/ConVIRT）、Perceiver、Gato 与 AlphaFold 共同奠定了通用模型的技术基础；Acosta 等[^5]和 Moor 等[^6]分别提出"多模态生物医学 AI"与"通用医疗 AI（GMAI）"概念。
3. **生成式 AI 与智能体时代（2023–2026）**——ChatGPT 引爆公众关注，Med-PaLM、Med-PaLM 2、Med-Gemini 与 GPT-4 系列推动 LLM 进入临床问答、对话、报告生成与转诊全流程；DeepSeek-R1、o-series 等推理模型与检索/工具调用智能体进一步扩展了能力边界[^7]。

Chen 等于 2026 年发表的 LLM 辅助系统综述[^1]统计显示：自 2022 年 1 月至 2025 年 9 月，临床医学领域累计发表关于 LLM 的研究约 4,609 篇，平均 **3.2 篇/天**；然而其中只有 **19 篇为前瞻性 RCT**，约 25% 的研究样本量低于 30；闭源模型占比高达 **87.7%**。这一"数量爆发、质量稀缺、生态封闭"的图景，是本综述展开讨论的总体背景。

本综述以截至 2026 年 4 月底的顶刊文献为主体，目标是回答以下问题：

- **Q1（技术）**：医疗 LLM 与多模态大模型的核心架构、训练范式与代表性模型有哪些里程碑？
- **Q2（评估）**：从 MultiMedQA 到 MedHELM，医疗 AI 评估体系如何演进？真实临床证据与基准成绩之间的落差有多大？
- **Q3（应用）**：在诊断推理、转诊、药房安全、心理治疗等真实任务中，LLM 的实证证据是否支持其临床部署？
- **Q4（安全）**：幻觉、过度自信、数据投毒、偏见与可重复性问题在医疗场景下如何放大？
- **Q5（未来）**：监管、人机协作、长上下文与智能体的下一步关键问题是什么？

---

## 2. 从专科 AI 到通用医疗基础模型：技术演进

### 2.1 范式转变：GMAI 框架

Moor 等在 *Nature* 提出的 GMAI 框架[^6]总结了通用医疗 AI 的三大核心能力：(i) 动态任务指定——通过自然语言或少样本即可执行未见任务；(ii) 灵活多模态输入输出——任意组合影像、文本、EHR、基因组与音视频；(iii) 医学领域知识表示——通过知识图谱与检索增强提供事实接地。GMAI 同时被认为是一柄"双刃剑"：模型能力的广泛适用性意味着其任何失效模式都会通过下游放大，因而"验证 GMAI 远比验证专用模型更困难"[^6]。

Acosta 等[^5]同期发表的多模态综述将上述能力对应到六类应用：精准组学健康、数字化临床试验、居家医院/远程监护、大流行病监测、数字孪生、虚拟健康助手；并明确指出**多模态特有的偏差放大**——当多源数据共同链接时，对各模态都更愿意提供数据的人群会被叠加放大代表性。

### 2.2 技术演进里程碑（2017–2026）

| 时间 | 里程碑 | 意义 |
|---|---|---|
| 2017 | Transformer[^A] | 跨模态统一架构基础 |
| 2018–2019 | BERT、BioBERT、ClinicalBERT[^A] | 生物医学双向编码器 |
| 2020 | GPT-3、AlphaFold、CLIP[^A] | LLM/蛋白/对比学习三大基石 |
| 2022 | ChatGPT、Med-PaLM 前身（Flan-PaLM 67.6% MedQA）[^8] | LLM 通过 USMLE 通过线 |
| 2022 | GatorTron 89B EHR 模型[^9] | 临床域内规模定律首次验证 |
| 2022 | BioGPT 347M[^10] | 生物医学生成式预训练 |
| 2023 | Med-PaLM[^8]、LLaVA-Med[^11]、Med-Flamingo[^12] | 医学 LLM 与多模态 VLM 同步突破 |
| 2024 | Med-PaLM 2 MedQA 86.5%[^13]、Med-PaLM M[^14]、BiomedGPT[^15]、Med-Gemini[^7][^16] | 多模态 + 长上下文 + 不确定性引导搜索 |
| 2024–2025 | AMIE 对话诊断[^17]、Med-Gemini-2D/3D/Polygenic[^16] | 真实工作流与基因组扩展 |
| 2025 | MetaMedQA[^2]、MedHELM[^18] | 评估体系系统升级 |
| 2026 | 心脏专科 LLM RCT[^19]、PreA 转诊 RCT[^20]、AMIE 心理治疗扩展[^21] | 真实临床 RCT 涌现 |

[^A]: 时间线背景节点引自 Acosta 等[^5]、Moor 等[^6]、Thirunavukarasu 等[^22]、Teo 等[^7]的综述。

### 2.3 训练范式

(a) **领域内全参从头预训练**：GatorTron 用 820 亿临床笔记词 + 60 亿 PubMed + 25 亿 Wiki，从头训练至 89B 参数；在 NLI、MQA 等复杂任务上相对 BioBERT 提升约 9.5–9.6%[^9]。BioGPT 在 1500 万 PubMed 摘要上从头训练 GPT-2 Medium，PubMedQA 准确率达 78.2%[^10]。

(b) **指令微调 + 提示策略**：Med-PaLM 系列的关键不在于继续预训练，而在于**指令提示微调**——仅更新软提示参数（约 65 个示例），即可将 Flan-PaLM 的科学共识一致率从 61.9% 提升至 92.6%，接近临床医生的 92.9%[^8]。Med-PaLM 2 进一步引入**集成精炼（Ensemble Refinement）**和**检索链（Chain of Retrieval）**：前者通过多路推理 + 综合输出，将 MedQA 提升至 86.5%；后者将初始答案拆分为可验证子声明，逐项调用搜索完成接地[^13]。

(c) **自博弈与模拟学习**：AMIE[^17] 在 PaLM 2 基础上引入自博弈框架——内循环用 in-context 批评反馈，外循环用迭代微调；模拟覆盖 5,230 种疾病，使其在 OSCE 风格 159 例对话研究中 30/32 个专家维度与 25/26 个患者维度上显著优于全科医生。

(d) **课程学习与轻量化**：LLaVA-Med 用两阶段课程——先 600K 图文对齐生物医学概念，再 60K 自指令对话微调，仅用 8 张 A100、<15 小时即完成训练[^11]。BiomedGPT 用 OFA 框架将 VQ-GAN 量化的视觉 token、文本 BPE 与位置 token 统一到 59,457 维词表，183M 参数在 25 项任务中 16 项达 SOTA，相对 562B Med-PaLM M 缩小 3,000+ 倍[^15]。

(e) **不确定性引导的智能体能力**：Med-Gemini[^7]提出基于 Shannon 熵的**不确定性引导搜索**：当模型置信度低于阈值时自动触发网络检索；MedQA(USMLE) 达到 91.1%（重新标注后上升至 91.8–92.9%，说明评测瓶颈已转向数据质量）；并依靠 1M+ token 上下文支持完整 EHR 与手术视频理解。

(f) **生成式 AI 的扩展架构**：Teo 等[^7]将这一阶段的方法学谱系总结为"合成数据系统 (VAE/GAN) → 扩散模型 → 规则型 AI → LLM → 多模态基础模型 → 推理/智能体 → 模型蒸馏"。其中**领域专用小模型可超越通用大模型**已被多次验证：Foresight 2 在英国 NHS EHR 时间线预测上显著优于 GPT-4[^7]。

---

## 3. 医疗大语言模型：从生物医学 NLP 到专家级问答

### 3.1 代表性模型横向对比

| 模型 | 年份 | 主干 | 训练数据 | 关键贡献 | 代表性结果 |
|---|---|---|---|---|---|
| **BioGPT**[^10] | 2022 | GPT-2 Medium 347M | 1500 万 PubMed 摘要 | 首个生物医学生成式 PT | PubMedQA 78.2%（+6.0% vs BioLinkBERT-large） |
| **GatorTron**[^9] | 2022 | Megatron-LM BERT | 820 亿临床笔记词 | 89B 临床语言模型 | NLI 90.20%（+9.6%），MQA 93.10%（+9.5%） |
| **Med-PaLM**[^8] | 2023 | Flan-PaLM 540B | MultiMedQA + 指令提示微调 | 多维医生人工评估框架 | MedQA 67.6%；危害比例 5.9% ≈ 医生 5.7% |
| **Med-PaLM 2**[^13] | 2025 | PaLM 2 | MedQA / MedMCQA / HSQA 等 | ER + Chain of Retrieval | MedQA 86.5%；专科会诊 65% 偏好优于 GP |
| **Med-Gemini-L**[^7] | 2024 | Gemini 1.0 Ultra | 多模态混合 + 自训练 | 不确定性引导搜索 | MedQA 91.1%；多模态平均较 GPT-4 提升 44.5% |
| **AMIE**[^17] | 2025 | PaLM 2 | 真实对话 + 自博弈 | 对话诊断专用 | OSCE 30/32 专科维度优于 GP |
| **心脏 AMIE**[^19] | 2026 | Gemini 2.0 Flash + 工具 | 9 例开发，无领域微调 | 首个心脏专科 LLM RCT | 显著错误率 13.1% vs 24.3%（P=0.033） |
| **PreA**[^20] | 2026 | GPT-4o mini + co-design | 共同设计而非微调 | 患者直面转诊 RCT（n=2,069） | 问诊时长 −28.7%；协调度 +113.1% |

### 3.2 评估框架的同步演进

Med-PaLM[^8]首次构建 **MultiMedQA**（整合 MedQA、MedMCQA、PubMedQA、MMLU 临床子集、LiveQA、MedicationQA + HealthSearchQA 3,173 题），并以"科学共识一致性 / 不当内容 / 遗漏 / 危害程度 / 偏见"5 维度由临床医生人工评估。Med-PaLM 2[^13]将评估升级为**配对排名**——9 个临床轴中医生评审者在 8 个轴上更偏好 AMIE / Med-PaLM 2 答案而非真实医生答案（均 P<0.001）。

### 3.3 推理与元认知

LLM 在 MedQA 上可接近满分，但 Griot 等[^2]以 *Nature Communications* 提出的 **MetaMedQA**（1,373 题，含 "以上均不正确"与"我不知道"选项）显示，12 个主流模型中 9 个的 "Unknown Recall" 为 **0%**，GPT-4o 也仅 3.7%——**几乎所有模型都"不知道自己不知道"**，并表现出系统性过度自信。Kim 等[^23]提出 **mARC-QA**（基于 Einstellung 思维定势）也得出相同结论：人类医生平均 66%，最优模型 DeepSeek-R1 仅 52%，部分医疗专用模型甚至接近 0%。

> *综合判读*：执照考试与"对抗性推理"两类基准之间的性能差距，是医疗 LLM 当前最重要的方法论问题之一。综述写作建议将其单列为"考试 ≠ 临床"小节，且与 Goh 等 RCT、Bean 等真实用户研究并列引用。

---

## 4. 多模态医疗大模型：影像、视频、组学与长上下文

### 4.1 模型生态与对比

| 模型 | 模态覆盖 | 规模 | 突出点 | 局限 |
|---|---|---|---|---|
| **LLaVA-Med**[^11] | 文本 + 影像（CXR/CT/MRI/病理） | 7B / 13B | GPT-4 自指令、两阶段课程 | 输入仅 224×224；幻觉显著 |
| **Med-Flamingo**[^12] | 文本 + 影像 | 8.3B（OpenFlamingo + LLaMA-7B） | 4,721 本教科书构造 MTB（80 万图，5.84 亿 tokens），少样本 ICL | PathVQA 上弱；仅 PoC |
| **Med-PaLM M**[^14] | 文本 + CXR + 病理 + 皮肤 + 钼靶 + 基因组 | 12B / 84B / 562B | MultiMedBench 14 任务统一权重 | 未开源；缺少前瞻验证 |
| **BiomedGPT**[^15] | 文本 + 2D 图像 + 表格 | 33M / 93M / 182M | 完全开源，轻量化 | 零样本 VQA <60% |
| **Med-Gemini-2D/3D/Polygenic**[^16] | 2D 影像 + 3D CT + 基因组 | 未披露 | 3D CT 报告生成、PRS 零样本 | 3D 仍有幻觉，CT-US1 等私有 |
| **MAIRA-2**[^24] | CXR + 历史片 + 临床文本 | ~7B (Vicuna + Rad-DINO) | "Grounded Radiology Report"（句级 BBox） | 仅 CXR；51% 训练数据私有 |

### 4.2 关键技术路线

- **早期/联合融合**：以 PaLM-E 为代表，将图像 ViT token 与文本 token 在同一嵌入空间交错（Med-PaLM M）[^14]。
- **门控交叉注意力**：以 Flamingo / OpenFlamingo 为骨干，配合 Perceiver Resampler 把视觉特征压缩到固定槽位（Med-Flamingo）[^12]。
- **离散 token 化与统一序列建模**：BiomedGPT 用 VQ-GAN 把图像离散为视觉 token，与文本/位置/边界框 token 共享词表[^15]；MAIRA-2 用 100×100 网格坐标 token 编码空间定位[^24]。
- **报告生成的可追溯接地**：MAIRA-2 配套提出 **RadFact** 评估框架，使用 Llama3-70B 进行句级逻辑蕴含推断，给出 *逻辑精确率 / 召回率* 与 *空间精确率 / 召回率*；在 MIMIC-CXR 上 BLEU-4 23.1、RadGraph-F1 34.6 为 SOTA，专家评审显示 91% 句子可直接接受[^24]。
- **超长上下文**：Med-Gemini-M 1.5 支持 1M+ tokens，无需 RAG 切片即可解析完整 EHR 或手术视频[^7]。

### 4.3 多模态评估基准

MultiMedBench[^14]（>100 万样本，覆盖 MedQA / MedMCQA / PubMedQA / MIMIC-III RRS / VQA-RAD / Slake-VQA / Path-VQA / MIMIC-CXR / PAD-UFES-20 / VinDr-Mammo / CBIS-DDSM / PrecisionFDA V2 等 14 项任务）目前是医疗多模态最广覆盖的基准。**PMC-15M**（1,500 万生物医学图文对）则成为 BiomedCLIP、LLaVA-Med 等模型的数据基础[^11]。

> *综合判读*：多模态医疗大模型的瓶颈已从模型架构转向**数据规模、模态平衡与放射科级评估指标**：放射影像在公开数据中占主导，病理、3D CT、心脏超声等模态严重稀缺；而临床显著性加权评估（如 RadFact 区分错误临床严重程度）仍是开放问题[^24]。

---

## 5. 临床部署与人机协作：来自 RCT 的证据

医疗 AI 综述长期被批评为"过度依赖回顾性基准、缺乏 RCT 证据"。Chen 等[^1]系统综述中真实 RCT 仅 19 篇。最近 24 个月，这一空白开始被填补，但结论并不一致。

### 5.1 重要 RCT / 前瞻研究一览

| 研究 | 设计 | 关键结论 |
|---|---|---|
| Goh et al. 2024[^25] | 50 名医生 + ChatGPT (GPT-4) 在 6 个病例上的单盲 RCT | LLM 单独 92% vs. 传统资源组 74%；但**医生+LLM 仅 76%**，与传统组无显著差异（P=.60）。揭示"LLM 性能 ≠ 临床使用价值"。 |
| Bean et al. 2026[^26] | 1,298 名公众随机使用 GPT-4o / Llama 3 / Command R+ 处理 10 类病症 | LLM 单独识别率 90.8%–99.2%；真实用户经 LLM 辅助后病症识别率 <34.5%，**显著低于对照组 55–67%**。 |
| Pais et al. 2024[^27] | Amazon 在线药房前瞻部署 MEDIC | 处方说明相关近失误事件下降 **33%**（CI: 26%–40%），建议覆盖率 +18.3%，采纳率 +28.5%。 |
| Tu et al. 2025[^17] | AMIE × 20 GP 双盲交叉 OSCE，159 场景 | 30/32 专科维度与 25/26 患者维度上 AMIE 显著优于 GP。 |
| O'Sullivan et al. 2026[^19] | 9 心脏内科医生 ± AMIE，对 107 例真实 HCM/心肌病随机交叉 | 整体偏好率 46.7% vs 32.7%（P=.02）；显著错误率 13.1% vs 24.3%。 |
| Tao et al. 2026[^20] | 中国两家三甲医院 2,069 名患者随机使用 PreA（GPT-4o mini + 共同设计） | 问诊时长 −28.7%、协调度 +113.1%、患者沟通 +16.0%。 |
| Rollwage & McFadyen 2026[^21] | 心理治疗"认知层"双盲 RCT + 19,674 条真实世界对话 | CTRS 由 3.16 提升至 4.53（η²=0.36）；高激活用户康复率 51.7% vs 32.8%。 |
| Beale 2025[^28] | 子宫切除术后社交论坛 10 题 × 4 来源（Doximity GPT、ChatGPT、Perplexity、外科医生） | AI 回复共情/质量与医生相当，但阅读级别为大学水平，远高于推荐的六年级。 |

### 5.2 关键洞见

1. **"独立 LLM 强 ≠ 人机协作强"**：Goh[^25] 与 Bean[^26] 互为印证——医生未做提示工程训练、患者表述不完整，这两条断链共同导致"理论 90%"塌缩为"现实 34%"。
2. **流程化、约束化部署最易兑现收益**：MEDIC[^27] 不让 LLM 直接生成文本，而是把它限定为结构化抽取器，再由确定性规则层组装并由五层安全护栏 (GR1–GR5) 拒答，实现产线级 33% 近失误下降。这一"抽取-组装-护栏"范式可推广至病历结构化、医嘱审核、转诊报告等场景。
3. **共同设计 > 单纯数据微调**：PreA[^20]的消融实验显示，参与式共同设计（多轮工作坊 + 对抗测试）在 300 个虚拟患者上的 5 维度评分显著优于在 515 个本地对话上微调的同一基础模型，提示**医疗 AI 的部署成效在很大程度上取决于上下文工程与利益相关方对齐**。
4. **专科扩展正在涌现**：心脏专科 LLM[^19] 是首个心脏病领域 LLM RCT；认知层架构[^21]在 NHS Talking Therapies 中以模块化方式将 CBT 干预与 LLM 解耦，是"专科 + 通用底座"的代表性范式。
5. **对患者直接面向（patient-facing）需更加谨慎**：Beale[^28] 与 Bean[^26] 都揭示了**可读性、健康素养与一致性**问题——同一问题不同 phrasing 可能得到截然相反建议。

---

## 6. 评估体系的演进

### 6.1 从执照考试到工作流覆盖

Bedi 等[^18]发表的 **MedHELM** 是当前覆盖最广的医疗 LLM 整体评估框架：

- 与 29 名临床医生（14 个专科、4 家机构）联合构建 **5 大类、22 子类、121 个临床任务**，子类归类准确率 96.7%；
- 配套 37 个基准（19 个现有 + 5 个重构 + 13 个新建，其中 12 个基于真实 EHR），覆盖公开 / 门控 / 私有数据；
- 提出 **LLM-jury** 方法（三模型评审团，Spearman > 0.85），将主观评分纳入大规模评估；
- 系统比较 9 个前沿模型（DeepSeek R1 与 o3-mini 并列第一，胜率 66%；Claude 3.5 Sonnet 以 15% 更低的成本达到 63% 胜率）；
- 所有模型在"行政与工作流"类别表现最差（0.53–0.63），与现有 LLM 部署落差最大的方向高度一致。

### 6.2 证据分层与 LLM 辅助系统综述

Chen 等[^1]提出的 **Tier 框架**已被广泛引用：

| Tier | 含义 | 现有研究占比（截至 2025-09） |
|---|---|---|
| **S** | 真实临床 RCT | 19 篇 |
| **I** | 真实临床数据回顾/前瞻研究 | 1,048 篇（含 S） |
| **II** | 模拟场景、开放式问答 | 1,857 篇 |
| **III** | 执照考试、多选题 | 1,704 篇 |

> 关键发现：LLM 在 Tier III 任务中 38.4% 胜率优于人类，在 Tier I 任务中仅 25.9%；样本量 <30 的研究 ≥25%，闭源模型占 87.7%[^1]。

### 6.3 多维度评估方法

- **接地与逻辑指标**：RadFact[^24]、F1-RadGraph 等取代 BLEU/ROUGE，对放射报告进行逻辑级蕴含判断。
- **不确定性与校准**：MetaMedQA[^2] 的 Missing Answer Recall、Unknown Recall 与置信度校准曲线已成为元认知评估事实标准。
- **真实用户实验**：HELPMed[^26]（2,400 条对话）首次系统验证了"基准 / 模拟患者 / 真实用户"三层评估之间的不可替代性。
- **报告规范**：TRIPOD-LLM[^7] 与 CONSORT-AI、SPIRIT-AI 一起，正在为医疗 LLM 临床报告提供与药物 RCT 同级的规范化要求。

---

## 7. 安全、可靠性与伦理风险

### 7.1 幻觉与元认知缺失

- LLM 的本质是统计关联建模，"幻觉"或更准确的"事实捏造"是结构性问题[^22]；
- Griot 等[^2]：12 个模型中 9 个 Unknown Recall = 0%；显式告知"某些问题可能超出知识范围"后 GPT-4o 的 Unknown Recall 由 3.7% 跃升至 44.4%（P<10⁻⁴），证明该缺陷部分可由提示与训练弥补；
- Kim 等[^23]：人类医生平均 66%，最优 LLM 52%；医疗专用 Medalpaca / Meditron 系列接近 0%——说明**在通用对抗性推理上专用模型甚至落后于通用模型**。

### 7.2 数据投毒与训练数据安全

Alber 等[^29]在 *Nature Medicine* 揭示：

- 仅**替换 0.001% 训练 token**（约 100 万 / 1000 亿，约 \$5），即可让 4B 模型在医疗有害内容生成率上显著上升 +4.8%（P=0.038）；
- 所有被投毒模型在 5 项主流医疗基准上得分与基线无显著差异——**现有评测体系对该类攻击完全失效**；
- 提出基于 BIOS 生物医学知识图谱的推理阶段防御算法，段落级召回率 91.9%、F1=85.7%，且可在消费级硬件运行；
- 进一步揭示后训练修复（Prompt、RAG、SFT）均无法显著缓解，需从训练数据来源透明度与持续审计入手。

### 7.3 偏见、公平性与文化适配

- AMIE[^17]对低英语读写水平患者表现明显下降，多语言 / 文化公平性是普遍盲区；
- Med-Gemini-Polygenic[^16] 仅在以欧洲裔为主的 UK Biobank 上验证，跨种族泛化性存疑；
- Goh[^25]、Bean[^26]、Beale[^28]共同指出：当前评估几乎全部以英语为主，全球部署面临系统性公平性挑战。

### 7.4 监管与责任

- 现有 FDA / NMPA 框架基于固定权重监管，与持续学习、智能体调用 / 工具使用 / 自我更新的 LLM 范式不匹配[^3][^7]；
- Chen[^1] 提议建立"以患者为中心的 LLM 临床研究路线图"，将真实临床数据、RCT、TRIPOD-LLM 等纳入门槛；
- 法律责任、知识产权与"AI 作者身份"在多家期刊政策上仍存分歧[^22]。

### 7.5 商业/利益冲突

多个高影响力研究 (AMIE[^17]、心脏专科 LLM[^19]、认知层架构[^21]、Med-Gemini[^7]、MAIRA-2[^24]) 来自 Google / Microsoft / 商业心理治疗公司，存在资金或股权层面的利益冲突——综述写作建议在引用其性能结论时显式标注。

---

## 8. 挑战与未来方向

综合 Moor[^6]、Acosta[^5]、Rajpurkar[^3]、Thirunavukarasu[^22]、Teo[^7]、Chen[^1]、Bedi[^18]等关键综述与本文整理的实证证据，可识别出以下八大重点方向：

1. **真实世界证据（RWE）的系统构建**：Tier S/I 研究仍稀缺；亟需以死亡率、再入院、误诊率为终点的多中心前瞻研究[^1]。
2. **评估体系的全工作流覆盖**：MedHELM[^18] 提出 22 子类框架，但 15 个子类仅含 1 个基准；需要扩展行政、患者沟通、转诊、姑息等长期被忽略的子任务。
3. **可追溯检索增强与知识图谱接地**：Chain of Retrieval[^13]、知识图谱事实核查[^29]、RadFact[^24] 共同指向"模型 + 外部验证器"的双层安全架构。
4. **不确定性表达与选择性回答**：未来模型需具备"我不确定"的可信号（基于熵、自一致性或外部分布外检测）[^2][^23]。
5. **真实用户交互测试纳入上市前评估**：Bean[^26] 直接呼吁"将真实用户实验纳入医疗 AI 监管"；这一点已部分被 TRIPOD-LLM 吸纳。
6. **开放与可复现**：闭源模型占比 87.7%[^1]，私有数据（Med-PaLM M、Med-Gemini、MAIRA-2 数据集）难以复核——BiomedGPT 等完全开源轻量化模型代表了重要反向力量[^15]。
7. **本地化与多语种部署**：Co-design[^20]、心理治疗本地适配[^21]显示对中低资源系统，参与式设计可能比单纯微调更有效。
8. **智能体与工具使用的安全边界**：Med-Gemini[^7] 的"不确定性引导搜索"、AMIE 心脏版本[^19] 的多步推理 + 网络搜索 + 自我批判，已经把 LLM 推向"智能体"范式。下一步关键问题是：何时允许其自主调用电子病历写入、医嘱发起、检查预约等具备实质影响力的工具？

---

## 9. 结论

医疗人工智能在过去四年完成了"基础模型化、生成式化、多模态化"三重转向：

- **能力层面**，Med-PaLM 2、Med-Gemini、AMIE 等已经在多个静态基准上超越医生水平，并在 OSCE、心脏专科、心理治疗、转诊等真实场景中取得了首批 RCT 证据；
- **评估层面**，MultiMedQA→MultiMedBench→MedHELM 的演进与 MetaMedQA、RadFact、TRIPOD-LLM 等工具一道，把医疗 AI 评估从"考试驱动"推向"工作流驱动"；
- **安全层面**，数据投毒[^29]、元认知缺陷[^2]、用户交互失败[^26]、过度自信[^23]共同提示我们：**在真实临床部署之前，模型必须证明的不是"能不能答对"，而是"能否在它不应该回答时主动停下"**。

未来三年最值得关注的研究问题，是把"通用 + 专科"、"自主 + 监督"、"基准 + RCT"、"模型 + 外部验证器"四组张力转化为可工程化、可监管化的范式——而非继续在 USMLE 上刷分。

---

## 参考文献

[^1]: Chen, S. F., Oermann, E. K. *et al.* LLM-assisted systematic review of large language models in clinical medicine. *Nature Medicine* **32**, 1152–1159 (2026). DOI: 10.1038/s41591-026-04229-5.
[^2]: Griot, M. *et al.* Large language models lack essential metacognition for reliable medical reasoning. *Nature Communications* (2025). DOI: 10.1038/s41467-024-55628-6.
[^3]: Rajpurkar, P., Chen, E., Banerjee, O. & Topol, E. J. AI in health and medicine. *Nature Medicine* **28**, 31–38 (2022). DOI: 10.1038/s41591-021-01614-0.
[^4]: Bommasani, R. *et al.* On the Opportunities and Risks of Foundation Models. arXiv:2108.07258 (2022).
[^5]: Acosta, J. N., Falcone, G. J., Rajpurkar, P. & Topol, E. J. Multimodal biomedical AI. *Nature Medicine* **28**, 1773–1784 (2022). DOI: 10.1038/s41591-022-01981-2.
[^6]: Moor, M. *et al.* Foundation models for generalist medical artificial intelligence. *Nature* **616**, 259–265 (2023). DOI: 10.1038/s41586-023-05881-4.
[^7]: Teo, Z. L., Thirunavukarasu, A. J. & Ting, D. S. W. Generative artificial intelligence in medicine. *Nature Medicine* **31** (2025). DOI: 10.1038/s41591-025-03983-2.
[^8]: Singhal, K. *et al.* Large language models encode clinical knowledge. *Nature* **620**, 172–180 (2023). DOI: 10.1038/s41586-023-06291-2.
[^9]: Yang, X. *et al.* A large language model for electronic health records (GatorTron). *npj Digital Medicine* (2022). DOI: 10.1038/s41746-022-00742-2.
[^10]: Luo, R. *et al.* BioGPT: generative pre-trained transformer for biomedical text generation and mining. *Briefings in Bioinformatics* **23**, bbac409 (2022). DOI: 10.1093/bib/bbac409.
[^11]: Li, C., Wong, C., Zhang, S. *et al.* LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. *NeurIPS 2023 Datasets and Benchmarks*.
[^12]: Moor, M., Huang, Q. *et al.* Med-Flamingo: A Multimodal Medical Few-Shot Learner. arXiv:2307.15189 (2023).
[^13]: Singhal, K. *et al.* Toward expert-level medical question answering with large language models (Med-PaLM 2). *Nature Medicine* **31**, 943–950 (2025). DOI: 10.1038/s41591-024-03423-7.
[^14]: Tu, T., Azizi, S. *et al.* Towards Generalist Biomedical AI (Med-PaLM M). *NEJM AI* (2024). arXiv:2307.14334.
[^15]: Zhang, K. *et al.* A generalist vision–language foundation model for diverse biomedical tasks (BiomedGPT). *Nature Medicine* (2024). DOI: 10.1038/s41591-024-03185-2.
[^16]: Yang, L. *et al.* Advancing Multimodal Medical Capabilities of Gemini (Med-Gemini-2D/3D/Polygenic). arXiv:2405.03162 (2024).
[^17]: Tu, T., Schaekermann, M., Palepu, A. *et al.* Towards conversational diagnostic artificial intelligence (AMIE). *Nature* **642** (2025). DOI: 10.1038/s41586-025-08866-7.
[^18]: Bedi, S. *et al.* Holistic evaluation of large language models for medical tasks with MedHELM. *Nature Medicine* **32**, 943–951 (2026). DOI: 10.1038/s41591-025-04151-2.
[^19]: O'Sullivan, J. W., Palepu, A., Ashley, E., Tu, T. *et al.* A large language model for complex cardiology care. *Nature Medicine* **32** (2026). DOI: 10.1038/s41591-025-04190-9.
[^20]: Tao, X., Zhou, S., Ding, K. *et al.* An LLM chatbot to facilitate primary-to-specialist care transitions: a randomized controlled trial. *Nature Medicine* **32**, 934–942 (2026). DOI: 10.1038/s41591-025-04176-7.
[^21]: Rollwage, M. & McFadyen, J. *et al.* A cognitive layer architecture to support large-language model performance in psychotherapy interactions. *Nature Medicine* (2026). DOI: 10.1038/s41591-026-04278-w.
[^22]: Thirunavukarasu, A. J. *et al.* Large language models in medicine. *Nature Medicine* **29**, 1930–1940 (2023). DOI: 10.1038/s41591-023-02448-8.
[^23]: Kim, J. & Bernardo, D. *et al.* Limitations of large language models in clinical problem-solving arising from inflexible reasoning. *Scientific Reports* (2025). DOI: 10.1038/s41598-025-22940-0.
[^24]: Bannur, S., Hyland, S. L. *et al.* MAIRA-2: Grounded Radiology Report Generation. arXiv:2406.04449 (2024).
[^25]: Goh, E. *et al.* Large language model influence on diagnostic reasoning: a randomized clinical trial. *JAMA Network Open* (2024). DOI: 10.1001/jamanetworkopen.2024.40969.
[^26]: Bean, A. M. & Mahdi, A. *et al.* Reliability of LLMs as medical assistants for the general public: a randomized preregistered study. *Nature Medicine* **32**, 609–615 (2026). DOI: 10.1038/s41591-025-04074-y.
[^27]: Pais, C., Liu, J. *et al.* Large language models for preventing medication direction errors in online pharmacies (MEDIC). *Nature Medicine* (2024). DOI: 10.1038/s41591-024-02933-8.
[^28]: Beale, S. K. *et al.* Comparing physician and AI chatbot responses to posthysterectomy questions posted to a public social media forum. *AJOG Global Reports* (2025). DOI: 10.1016/j.xagr.2025.100553.
[^29]: Alber, D. A., Oermann, E. K. *et al.* Medical large language models are vulnerable to data-poisoning attacks. *Nature Medicine* (2025). DOI: 10.1038/s41591-024-03445-1.

---

## 附录 A：本综述未充分展开但已纳入文献库的条目

- **AI in Health and Medicine** (Rajpurkar 2022)[^3]——影像、NLP、分子生物学综述基础。
- **Large Language Models—Misdiagnosing Diagnostic Excellence?** (JAMA Network Open, 2024)——配套 Goh 等 RCT 的评论文献，可用于强化"协作落差"的讨论。
- **Limitations of LLMs in Clinical Diagnostic Reasoning** (JAMA Network Open, 2026)——2026-04 发表，主题与 Kim 等[^23]互补。
- **Capabilities of Gemini Models in Medicine**（Med-Gemini 1.x 完整技术报告）[^7] 中关于 ECG、皮肤镜、手术视频的细分实验。
- **From Large Language Models to Multimodal AI: A Scoping Review on the Potential of Generative AI in Medicine** (2025, PMC)——可作为补充检索线索，但写作时优先引用顶刊综述。

## 附录 B：建议下一步可补充的方向（综述初稿审计建议）

1. **多语言 / 中文医疗大模型**：现有文献清单以英文世界为主；建议补充 HuatuoGPT、PULSE、BianQue 等中文工作及其与 NHS / 中国医保体系的部署研究。
2. **影像专用基础模型**：RETFound（眼底）、CONCH（病理）、CT-FM、SAM-Med 等专科基础模型未纳入本初稿，可在多模态章节进一步加入比较。
3. **EHR 时间线预测专门模型**：Foresight / Foresight 2 / Med-BERT / Cehr-BERT，可作为"专用 EHR 模型 vs. 通用 LLM"对比专题。
4. **健康公平性与全球南方**：可补充 Pfohl 等 EquityMedQA（已被 Med-PaLM 2 引用）、STANDING-TOGETHER、FUTURE-AI 等工具的系统介绍。
5. **AI 治理与监管**：欧盟 AI Act、FDA Predetermined Change Control Plan (PCCP)、NICE 医疗 AI 评估指南，可与 TRIPOD-LLM、CONSORT-AI 并列加入"治理"章节。

> 本初稿基于 `Medical_AI_Literature_Search.md` 与 `paper_extra/` 下的 31 份结构化摘要整理，并按综述写作惯例对论点进行了再组织、批判性筛选与补充。所有结论需在投稿前由原始 PDF 全文复核，尤其涉及具体数值（准确率、置信区间、样本量、效应量）和未在 *Nature/NEJM/JAMA* 期刊正式发表的预印本（Med-Flamingo、Med-Gemini、Med-PaLM M、MAIRA-2 等）。
