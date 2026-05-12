# 医疗人工智能与医疗大模型研究综述 — 论证骨架（v0.2）

> **目的**：在动笔之前先把"每章打算证明什么、用什么证据、综述者自己补什么判断"对齐清楚。
> **写作策略**：Part I（§1–§5）做底色——按发展路径与能力图景描述当前医疗 AI 的"是什么"，约占全文一半篇幅；Part II（§6–§10）做诊断——以"通往可信临床部署的关键缺口"为线索揭示**评估、协作、治理**三类结构性挑战，并给出修复方向。
> **语气分层**：Part I 章末用 1 段"小结性判断"完成承上启下；Part II 章首用 1 段"反直觉断言"打开论证，集中释放综述者立场。

---

## 中心命题（贯穿全文）

> 本综述系统回顾 2022–2026 年医疗人工智能从任务专用监督学习到生成式基础模型的范式演进，对当下能力图景（语言、多模态、专科底座）做一次完整快照；继而以"通往可信临床部署的关键缺口"为线索，揭示**评估方法学、人机协作、训练治理**三类尚未解决的结构性挑战，论证医疗 AI 距离"能用、敢用、可监管"的目标仍有相当距离。

锐利的判断不被否定，但**位置后移**：Part I 完成"路径已然清晰"的客观叙述；Part II 完成"距离尚未跨越"的批判性诊断。

---

# Part I · 发展路径与能力图景

## §1 引言

**中心论点**：医疗 AI 在过去四年完成了模型规模、模态广度与交互形态三方面的快速扩张，但同期临床证据、评估方法学与监管治理的增长是线性的——综述要解释的不是"AI 多强"，而是"为什么我们对'多强'的判断如此不可靠"。

**证据链**：
- 三阶段历史脉络：Rajpurkar 等[^3]提出的"专科监督 → 基础模型 → 生成式"轨迹；Bommasani 等[^4]奠定的基础模型概念基础。
- 文献规模与质量错位：Chen 等[^1]系统综述显示 2022 年 1 月至 2025 年 9 月医疗 LLM 论文 4,609 篇、平均 3.2 篇/天，但真实 RCT 仅 19 篇，闭源占 87.7%，样本量 <30 占 25%。
- 当代综述参照：Thirunavukarasu 等[^22]给出"医学 LLM"的分析框架；Teo 等[^7]给出"生成式医疗 AI"作为本阶段标签。

**综述者判断**：把"能力—证据—治理"的速率差作为分析框架明确写出，而不仅作为铺垫。引言末尾给出五个研究问题（Q1 技术演进 / Q2 多模态格局 / Q3 专科与通用关系 / Q4 临床证据与评估转向 / Q5 治理与修复），分别对应 §2–§9。

**表/图**：图 1.1 综述结构总览（Part I × Part II × 三类缺口的对应关系）。

---

## §2 范式演进回顾

**中心论点**：医疗 AI 在 2015–2026 完成三阶段跃迁——任务专用监督学习 → 多模态基础模型 → 生成式智能体——其底层动力是训练目标从"标签驱动"到"语言驱动"再到"工具与反馈驱动"的转变；三个阶段并非简单替代，前阶段方法学在新阶段中作为模块持续发挥作用。

**证据链**：
- 阶段一（任务专用监督学习，~2015–2020）：Gulshan 等糖网筛查、Esteva 等皮肤癌分类、McKinney 等乳腺癌筛查；FDA 至 2026 年批准 AI/ML 医疗器械超过 500 项[^3]，但几乎全部针对单一模态、单一任务。
- 阶段二（基础模型与多模态融合，2020–2023）：技术基石是 Transformer、BERT/BioBERT/ClinicalBERT、CLIP/ConVIRT、GPT-3、AlphaFold；Acosta 等[^5]与 Moor 等[^6]分别提出"多模态生物医学 AI"与"通用医疗 AI（GMAI）"概念，GMAI 的三大核心能力（动态任务指定、灵活多模态、医学知识表示）成为本阶段的目标函数。
- 阶段三（生成式 AI 与智能体，2023–2026）：Med-PaLM[^8] / Med-PaLM 2[^13] / Med-Gemini[^7][^16] / AMIE[^17] 推动 LLM 进入临床问答、对话、报告生成与转诊全流程；DeepSeek-R1、o-series 等推理模型与检索/工具调用智能体进一步扩展能力边界[^7]。
- 训练范式横剖：
  - 领域内全参预训练：GatorTron 89B[^9]（820 亿临床词，NLI +9.6%）；BioGPT 347M[^10]（1500 万 PubMed 摘要，PubMedQA 78.2%）。
  - 指令微调与提示策略：Med-PaLM[^8]仅更新约 65 个软提示参数即可让 Flan-PaLM 科学共识一致率由 61.9% 升至 92.6%；Med-PaLM 2[^13]叠加 Ensemble Refinement 与 Chain of Retrieval。
  - 自博弈与模拟学习：AMIE[^17]内循环 in-context 批评、外循环迭代微调，模拟覆盖 5,230 种疾病。
  - 课程学习与轻量化：LLaVA-Med[^11]两阶段课程（60 万图文对齐 + 6 万自指令对话）、8 张 A100、<15 小时；BiomedGPT[^15] 用 OFA 框架将参数压至 183M，在 25 任务 16 SOTA。
  - 不确定性引导智能体：Med-Gemini[^7]基于熵的网络检索 + 1M token 上下文。

**综述者判断**：三阶段并非替代关系。阶段一的对比学习（CLIP/ConVIRT）成为阶段二多模态对齐组件，阶段二的 BERT 系列变成阶段三的检索召回端编码器——这一观察为 §5 的"专科基础模型不可替代位置"做铺垫，也提示综述读者不要把"通用化"理解为"专科化的终点"。

**表/图**：
- 表 2.1 技术里程碑时间表（保留 v0.1 并扩充：补 RETFound 2023 / CONCH 2024 / CT-FM 2024 / SAM-Med2D 2023 等专科基础模型节点）。
- 图 2.1 三阶段训练目标演化谱（标签 → 语言 → 工具/反馈）。

---

## §3 医疗大语言模型的能力图景

**中心论点**：医疗 LLM 已在静态考试上接近性能上限，并通过指令微调、推理增强、自博弈三条路线分化出三种代表性体系；英文体系以"通过 USMLE / 增益医生工作流"为里程碑，中文体系以"完成一次合规多轮问诊"为产品形态，两条路径共同构成 LLM 的能力图景。

**证据链**：
- 路线 1 指令微调 + 提示策略：Med-PaLM[^8]（MedQA 67.6%、危害比例 5.9% ≈ 医生 5.7%、首个建立 MultiMedQA 与 5 维医生评估框架）；Med-PaLM 2[^13]（Ensemble Refinement + Chain of Retrieval、MedQA 86.5%、9 临床轴 8 轴优于医生答案）。
- 路线 2 不确定性与推理增强：Med-Gemini[^7]（基于 Shannon 熵的网络检索触发、MedQA(USMLE) 91.1%、测题被重新标注后 SOTA 上移至 91.8–92.9%，揭示评测瓶颈已转向标注质量；1M+ token 上下文支持完整 EHR/手术视频解析）。
- 路线 3 自博弈对话：AMIE[^17]（PaLM 2 底座、in-context 批评内循环 + 迭代微调外循环、覆盖 5,230 种疾病；OSCE 159 场景对 20 名 GP，30/32 专家维度与 25/26 患者维度显著领先）。
- 中文路线（agent 检索补充）：
  - HuatuoGPT[^HuatuoGPT]（LLaMA-13B + Huatuo-26M + Med-Dialog；混合数据 SFT + RLMF；对 DoctorGLM 胜率 97%，对 ChatGPT 胜率 62%）。
  - BianQue[^BianQue]（ChatGLM-6B + BianQueCorpus 243 万条；提出 Chain of Questioning (CoQ)、问题/建议比 46.2/53.8；MedDG 上 BLEU-1 14.86 vs ChatGPT 5.11，PQA 0.81 vs 0.63）。
  - PULSE[^PULSE]（OpenMEDLab，BLOOMZ-7B/自研 20B 基座，约 400 万中文医学指令，CMB / MedQA-MCMLE / MedBench 上压制 ChatGLM/Baichuan）。
  - DISC-MedLLM[^DISCMed]（Baichuan-13B + DISC-Med-SFT 47 万样本，三条管线：MedDialog 改写、CMeKG 三元组、行为偏好对齐；多轮一致性与"行为得分"超 HuatuoGPT/BianQue/中文 GPT-3.5）。
- 早期 NLP 基础（背景层）：GatorTron[^9]（89B EHR / NLI 90.20% / MQA 93.10%）、BioGPT[^10]（PubMedQA 78.2%）作为生成式医疗 NLP 的桥梁。
- 评估同步演进：MultiMedQA[^8] → Med-PaLM 2 配对排名[^13] → MedHELM[^18]（§7 详述）。

**综述者判断**：中英文路线的差异反映**目标场景的差异**，而非技术代差。英文社区把 USMLE 通过/医生工作流增益作为里程碑，因而长于"应答能力"；中文社区把"完成合规多轮问诊"作为产品形态，因而长于"主动追问能力"（CoQ 是这一差异的方法学化）。这一观察将在 §7 升级为"评估的英语偏倚"小命题，也是 §9 修复架构中"通用骨架需以本地化数据二次适配"的论据。

**表/图**：
- 表 3.1 代表性医疗 LLM 横向对比（保留 v0.1 并升级：补 HuatuoGPT / BianQue / PULSE / DISC-MedLLM 行，列加入"目标场景"与"评估基准类型"）。
- 图 3.1 训练目标演化（SFT / RLHF / RLMF / 自博弈 / 不确定性引导）。

---

## §4 多模态医疗大模型

**中心论点**：多模态医疗大模型在过去三年完成"接入图像 → 统一视-语-表 → 长上下文 + 接地生成"的演进；架构谱系已收敛为三条并行路线，竞争焦点已从架构转向**数据与评估**。

**证据链**：
- 架构谱系：
  - 早期/联合融合（PaLM-E 式 token 交错）：Med-PaLM M[^14]（12B/84B/562B，MultiMedBench 14 任务统一权重）。
  - 门控交叉注意力（Flamingo 系 + Perceiver Resampler）：Med-Flamingo[^12]（OpenFlamingo + LLaMA-7B，MTB 80 万图、5.84 亿 tokens，少样本 ICL）。
  - 离散 token 化与统一序列建模：BiomedGPT[^15]（VQ-GAN 视觉 token、59,457 维统一词表，33M–182M 参数，25 任务 16 SOTA，相对 562B Med-PaLM M 缩小 3,000+ 倍）；MAIRA-2[^24]（100×100 网格坐标 token，句级 BBox 报告生成）。
- 长上下文与模态扩展：Med-Gemini-2D/3D/Polygenic[^16]（3D CT 报告生成、PRS 零样本、ECG/皮肤镜/手术视频细分实验）；Med-Gemini-M 1.5 1M+ tokens 支持完整 EHR 与手术视频解析[^7]。
- 接地与可追溯：MAIRA-2 配套提出 RadFact[^24]，用 Llama-3-70B 进行句级逻辑蕴含推断，给出 *逻辑/空间 × 精确率/召回率* 四维评估；MIMIC-CXR 上 BLEU-4 23.1、RadGraph-F1 34.6 SOTA，91% 句子被专家直接接受。
- 数据基础：PMC-15M（1500 万生物医学图文对，BiomedCLIP / LLaVA-Med / CONCH 系数据共用）。
- 评估基准：MultiMedBench[^14]（>100 万样本，覆盖 14 任务）。

**综述者判断**：三条架构路线在同等数据规模下性能差距已小于 5 个百分点（MultiMedBench 跨架构对比），但数据可得性差距在**病理、3D CT、心脏超声**等长尾模态上仍超过两个数量级。未来 24 个月最具回报率的工作不在新架构，而在 (a) 病理 / 3D CT / 心脏超声多机构数据联盟，(b) 临床显著性加权评估指标的协议化。这一判断与 §5 在 EHR 时序与影像方向的"专科基础模型不可替代"互为补强。

**表/图**：
- 表 4.1 代表性多模态医疗模型对比（保留 v0.1 升级：补充评估指标列 BLEU/RadFact/Spatial-F1）。
- 图 4.1 三条架构路线的拓扑示意。

---

## §5 专科基础模型的不可替代位置

**中心论点**：在结构化 EHR 时序、专科影像、组学预测等具备强结构先验的子领域，专科自监督基础模型在**精度、样本效率、跨机构泛化**三个维度上系统性优于通用多模态大模型；通用与专科是互补而非进化关系。

**证据链**：
- EHR 时序：
  - Med-BERT[^MedBERT]：1700 万参 BERT 风格、ICD-9/10 编码序列、Cerner 2849 万患者预训练、Prolonged LOS 自监督任务；PaCa-Cerner 提升 +6.14% AUC，300 样本场景下仍可达 0.75，等效训练集 ×10。
  - CEHR-BERT[^CEHRBERT]：约 900 万参、ATT 时间 token + concept/time/age 多嵌入、VTP 替代 NSP；T2DM→HF AUC 80.7%、PR-AUC 32.3%，优于 MedBert 约 2.5/5 个百分点；5% 训练数据微调仍超全量基线。
  - Foresight 2[^Foresight2]：Mistral-7B/LLaMAv2-7B + SNOMED token 嵌入、语境化时间线；MIMIC-III 风险预测 P@5 0.90 vs GPT-4-turbo 0.65；上下文消融显示移除文本上下文性能下降约 40%。
- 眼底视网膜：RETFound[^RETFound]（Nature 2023）—— ViT-L、MAE 自监督，90.4 万 CFP + 73.6 万 OCT，8 项眼科诊断 + 4 项眼-全身预测全面优于 ImageNet-ViT/SL，小样本场景优势最大。
- 病理：CONCH[^CONCH]（Nature Medicine 2024）—— CoCa 框架，117 万图文对 + 1600 万 ROI iBOT 视觉自监督，14 基准 11 SOTA，zero-shot 罕见癌种较 PLIP 提升 10+ 百分点。
- CT：Pai et al. 2024[^CTFM]（Nature Machine Intelligence）—— 3D ResNet-50 SimCLR，1.1 万 CT、60 万 patches；跨机构外部验证 AUC / C-index 显著优于 ImageNet-3D 与从头训练，预后任务尤为稳健。
- 分割：SAM-Med2D[^SAMMed]——SAM 适配 + adapter，31 公开集 460 万 mask / 19.7 万张图、10 模态；point/bbox prompt 下 Dice 较 SAM 提升 10–30 百分点。
- 通用模型的局限证据：Jin 等[^GPT4VLimits] 在 *npj Digital Medicine* 系统揭示 GPT-4V 在多模态医学任务上"表面专家级精度"背后存在大量隐性失效——准确选项下推理路径仍含图像感知错误，与 RETFound / CONCH / Pai et al. 等专科模型在同类任务上的稳定性形成鲜明对照；EHR 时序中 Foresight 2 / Med-BERT 在 ICD/SNOMED 归纳偏置任务上同样压制"把 EHR 序列硬塞进 GPT-4"的方案。

**综述者判断**：把"通用还是专科"的二选一伪问题剥去后，真正的方法论问题是——**结构先验在什么任务上是性能上限的瓶颈，在什么任务上只是优化常数？** 尝试性回答：当下游任务的标签空间结构化、时序性强、样本稀缺时，结构先验是上限；当任务的语言-推理胶水占主导、上下文长且开放时，通用骨架是上限。这一判断是 §9 四要素架构中"专科底座 + 通用骨架"双层设计的直接来源。

**表/图**：
- 表 5.1 通用模型 vs 专科模型在四类任务上的相对优势矩阵（行：影像几何 / 罕见类别长尾 / 结构化时序 / 开放问答；列：通用胜 / 专科胜 / 互补 / 证据不足）。
- 图 5.1 专科自监督任务的设计模式（MAE / iBOT / SimCLR / Prolonged LOS / ATT）的对照。

---

### Part I 小结

把 §2 范式演进、§3 LLM、§4 多模态、§5 专科基础模型并置后能看到一个共同结构：**能力扩张的来源是数据规模 × 训练目标 × 结构先验三者乘积**，没有任何一项能单独决定上限；当下"通用多模态大模型 + 专科基础模型 + 检索/工具调用"的混合架构是这一时期最具代表性的事实图景。Part I 的描述是 Part II 三类缺口诊断的基础——下面的批判性章节都将以"距离这一图景中已被声称的能力还有多远"为线索展开。

---

# Part II · 通往可信部署的三类结构性缺口

## §6 真实临床证据：从基准到 RCT

**反直觉断言**：医疗 AI 在能力指标上的领先与其在真实工作流中的协作收益之间存在系统性脱节——LLM 单独越强，未受训人类协作收益反而可能越小；真正能兑现 RCT 级收益的部署模式是**约束化抽取**与**参与式共同设计**，而不是单纯更换更强的底座。

**证据链**：
- 协作劣化：
  - Goh 等 2024[^25]：50 名医生 6 病例单盲 RCT，LLM 单独 92% / 医生+LLM 仅 76%，与传统资源组 74% 无显著差异（P=.60）。
  - Bean 等 2026[^26]：1,298 公众用户 × 10 类病症，LLM 单独识别率 90.8–99.2%，真实用户经 LLM 辅助后病症识别率 <34.5%，**显著低于对照组 55–67%**。
- 约束化部署：Pais 等 2024 MEDIC[^27]——Amazon 在线药房前瞻部署，把 LLM 限定为结构化抽取器并由五层护栏（GR1–GR5）拒答；处方说明近失误事件下降 33% (CI 26–40%)，建议覆盖率 +18.3%，采纳率 +28.5%。
- 共同设计：Tao 等 2026 PreA[^20]——中国两家三甲、2,069 名患者随机使用 GPT-4o mini + co-design；问诊时长 −28.7%、协调度 +113.1%、患者沟通 +16.0%；消融实验显示共同设计组在 300 个虚拟患者上击败用 515 本地对话微调的同一基础模型。
- 专科与模块化部署：
  - AMIE OSCE[^17]：20 GP 双盲交叉、159 场景，30/32 专家维度与 25/26 患者维度 AMIE 显著领先。
  - 心脏 AMIE[^19]（O'Sullivan 等 2026）：107 例真实 HCM/心肌病随机交叉，整体偏好 46.7% vs 32.7% (P=.02)，显著错误率 13.1% vs 24.3% (P=.033)。
  - 心理治疗认知层[^21]（Rollwage & McFadyen 2026）：CTRS 由 3.16 提升至 4.53 (η²=0.36)；高激活用户康复率 51.7% vs 32.8%；19,674 条真实世界对话。
- 患者侧风险：Beale 2025[^28]——子宫切除社交论坛问答：AI 共情/质量与医生相当，但阅读级别为大学水平，远高于推荐的六年级。

**综述者判断**：这些研究合在一起暗示**医疗 AI 的部署科学正从模型科学分离出来，构成一个独立的研究领域**，其因果变量是工作流约束、用户素养、共同设计强度，而不是基准准确率。综述据此反对"换更强的模型即可"的简单主义，并把"约束化抽取 + 共同设计 + 模块化专科扩展"列为可兑现 RCT 收益的三种可推广范式。

**表/图**：
- 表 6.1 关键 RCT 总览（设计 / N / 干预 / 主要终点 / 效应大小 / 协作收益）。
- 图 6.1 单独 LLM 能力 vs 协作收益的"剪刀差"示意。

---

## §7 评估体系的三重转向

**反直觉断言**：医疗 AI 评估的瓶颈不再是"题目难不难"，而是"评估空间的拓扑够不够大"——任务覆盖、维度覆盖、被试覆盖三个面同时位移，旧基准已经无法捕捉新模型的多数失效模式。

**证据链**：
- 任务覆盖面位移：Bedi 等 2026 MedHELM[^18]——与 29 临床医生共建 5 大类、22 子类、121 临床任务、37 基准（含 12 个 EHR 真实基准），子类归类准确率 96.7%，LLM-jury Spearman > 0.85；DeepSeek R1 / o3-mini 并列第一胜率 66%，Claude 3.5 Sonnet 以 15% 更低成本达 63%；**所有模型在"行政与工作流"上得分最差（0.53–0.63）**。
- 维度面位移（不确定性 / 元认知）：
  - Griot 等 MetaMedQA[^2]——1,373 题含"以上均不正确""我不知道"选项；12 个主流模型 9 个 Unknown Recall = 0%、GPT-4o 仅 3.7%；显式提示后 GPT-4o 跃升至 44.4% (P<10⁻⁴)，证明部分可由训练/提示弥补。
  - Kim 等 mARC-QA[^23]（Einstellung 思维定势）——人类医生 66%、最优 LLM 52%、医疗专用 Medalpaca/Meditron 接近 0%——医疗专用化在静态考试有效，在对抗推理上反而有害。
- 维度面位移（接地与逻辑）：MAIRA-2 RadFact[^24] 提供逻辑/空间双轴评估，取代 BLEU/ROUGE。
- 维度面位移（公平性）：Pfohl 等 EquityMedQA[^Pfohl]——6 维度 × 7 对抗数据集（4,619 样本）× 17,099 评分；专家在 EquityMedQA 上对 Med-PaLM 2 报告偏见率 0.126 vs HealthSearchQA 0.030，验证对抗数据集的敏感性；消费者评分者偏见报告率显著高于医生与公平专家。
- 被试侧位移：HELPMed[^26]（Bean 2026 配套）—— 2,400 条真实对话首次系统验证"基准 / 模拟患者 / 真实用户"三层评估的不可替代性。
- 报告规范：TRIPOD-LLM[^7]；FUTURE-AI[^FUTUREAI]（BMJ 2025，117 国际专家、24 个月、8 轮 Delphi、6 大原则 30 条最佳实践，异议率 < 5%）；与 CONSORT-AI / SPIRIT-AI 协同。
- 证据分层：Chen 等 Tier 框架[^1]——Tier III LLM 胜率 38.4% vs Tier I 25.9%。

**综述者判断**：三个面的位移不是独立的——任务覆盖扩张暴露不确定性盲区，被试切换为真实用户又反过来揭示公平性问题。综述明确指出 MedHELM 仍存在的两个结构盲区——**15 个子类只有 1 个基准、几乎全部以英文为主**——并把 EquityMedQA、FUTURE-AI、中文医疗基准（CMB、CMExam、MedBench[^CMB]）共同作为补齐方案。**评估科学的下一步不是"出更难的题"，而是"扩大评估空间的拓扑"**——这是综述者在 §7 提出的元命题。

**表/图**：
- 表 7.1 评估转向矩阵（行：任务 / 维度 / 被试；列：旧基准 / 新基准 / 未覆盖）。
- 图 7.1 Tier S/I/II/III 文献规模柱图 + LLM 胜率叠加曲线。

---

## §8 训练–治理链路的隐性脆弱

**反直觉断言**：医疗 LLM 的真正系统性风险不在推理时——而是在**训练数据来源、闭源生态、监管框架**三个环节同时表现为不透明——任何一个环节的不透明都足以让事后审计失效；现行评估体系对这一类训练侧风险几乎完全无感。

**证据链**：
- 训练侧攻击：Alber 等 2025[^29]——替换 0.001% 训练 token（约 100 万 / 1000 亿，成本约 \$5），4B 模型有害内容生成率上升 +4.8% (P=0.038)；**所有受感染模型在 5 项主流医疗基准上得分与基线无显著差异**——评测体系**结构性失效**。后训练修复（Prompt、RAG、SFT）均无法显著缓解；作者提出基于 BIOS 医学知识图谱的推理阶段防御算法，段落级召回率 91.9%、F1 85.7%，可在消费级硬件运行。
- 闭源占比与可复核性：Chen 2026[^1] 闭源占 87.7%；Med-PaLM M / Med-Gemini / MAIRA-2 训练数据私有；BiomedGPT[^15] 是当下唯一全开源轻量化反例。
- 利益冲突：AMIE[^17] / 心脏 AMIE[^19] / Med-Gemini[^7] / 认知层架构[^21] / MAIRA-2[^24] 主要研究人员/资金来自 Google / Microsoft / 商业心理治疗公司——综述写作建议在引用其性能结论时显式标注。
- 偏见与公平性盲区：AMIE 对低英语读写水平患者表现明显下降[^17]；Med-Gemini-Polygenic 仅在以欧裔为主的 UK Biobank 验证[^16]；Bean[^26] / Beale[^28] 共同揭示英语为主的评估对全球部署构成系统性公平挑战。
- 监管不匹配：Rajpurkar[^3] 与 Teo[^7] 指出现行 FDA / NMPA 框架基于固定权重监管，与持续学习、智能体调用、自我更新的 LLM 范式不匹配；FUTURE-AI[^FUTUREAI] 明确指出现行法规禁止部署后修改，与框架推荐的持续更新机制矛盾；EU AI Act、FDA PCCP、NICE 医疗 AI 评估指南需协同。
- 责任与作者身份：Thirunavukarasu 等[^22]指出 AI 作者身份与知识产权在期刊政策上仍存分歧。

**综述者判断**：综述提出"**训练数据透明度是医疗 AI 安全性的必要条件**"——该命题比"开源"更强（要求供应链可审计），比"监管"更具体（要求训练数据来源声明强制化）。它直接接续 §9 修复架构的"持续审计层"要素，构成 Part II 三类缺口的最后一环。

**表/图**：图 8.1 训练—治理链路的三处不透明环节示意（数据来源黑箱 / 权重不可访问 / 监管不匹配）。

---

## §9 修复路径：从模型为中心到四要素架构

**反直觉断言**：把 §6–§8 的诊断串起来，结论不是"再训练一个更大的模型"，而是把医疗 AI 重新定义为**四要素系统**——**专科底座 + 通用骨架 + 外部验证器 + 持续审计层**——任何一要素缺失都会让系统重新塌缩到 §6–§8 的三种缺口之一。

**四要素对应表**：
- 专科底座 ← §5：Foresight 2[^Foresight2] / Med-BERT[^MedBERT] / CEHR-BERT[^CEHRBERT] / RETFound[^RETFound] / CONCH[^CONCH] / CT-FM[^CTFM] / SAM-Med2D[^SAMMed]；解决"结构先验缺失"导致的精度塌缩。
- 通用骨架 ← §3 / §4：Med-PaLM 2[^13] / Med-Gemini[^7] / AMIE[^17] / Med-PaLM M[^14]；解决"开放域语言-推理胶水"问题。
- 外部验证器 ← §3 / §4 / §8：Chain of Retrieval[^13] / RadFact[^24] / BIOS 知识图谱投毒防御[^29] / MEDIC 五层护栏[^27]；解决幻觉、接地与训练侧风险。
- 持续审计层 ← §7 / §8：MedHELM[^18] / EquityMedQA[^Pfohl] / TRIPOD-LLM[^7] / FUTURE-AI[^FUTUREAI] / HELPMed 真实用户监测[^26]；解决评估盲区与监管脱节。

**综述者判断**：四要素架构不是技术蓝图，而是**审计清单**——任何具体方案如果只覆盖前两要素（这是当前明星模型的常态），应当被标注为"系统不完整"。综述以此为依据反向标记 AMIE、Med-Gemini、Med-PaLM M 等现有体系的具体缺口（如外部验证器仅覆盖语言端、持续审计层缺失），作为对学界的具体建议。

**表/图**：
- 表 9.1 四要素架构 × 三种缺口对应矩阵。
- 图 9.1 当前明星模型在四要素清单上的覆盖度雷达图（综述者自制对比图）。

---

## §10 结论

**收束论点**：医疗 AI 在能力维度已经完成"基础模型化、生成式化、多模态化"的三重转向，但评估、临床、治理三条配套坐标轴仍滞留在旧体系——综述对未来三年的核心呼吁是把"通用 + 专科、自主 + 监督、基准 + RCT、模型 + 外部验证器、训练 + 审计"这五组张力转化为**可工程化、可监管化、可复现化**的规范，而不是继续在 USMLE 上刷分。

**综述者立场（三条具体主张）**：
1. **评估应当奖励"知道自己不知道"**——选择性回答、不确定性显式化应作为新基准的一等公民。
2. **监管应当强制训练数据透明**——医疗 LLM 上市/部署前需声明训练数据来源、规模、人群分布与潜在投毒检测结果。
3. **临床应当承认协作非线性**——任何 AI 辅助决策工具在上市评估中需明确报告**真实用户协作组**的效果，而不仅是模型独立的基准成绩。

**回扣引用**：[^1][^2][^17][^25][^26][^29][^18][^FUTUREAI]。

---

# 附：写作约定（待你最终确认）

1. **章节内部组织**：每章先 1 段（约 150–200 字）陈述论点 + 锚定 1–2 篇标志性文献；中段 2–3 段连贯叙述铺开证据；章末 1 段综述者判断，自然过渡到下一章。
2. **数值引用**：仅在论点必要时显式引用准确数（如 86.5% / 87.7% / P@5 0.90 vs 0.65），其余以"接近 / 远超 / 数量级差异"等表述减少铺陈感。
3. **表格策略（升级）**：考虑投课程报告/内部综述，允许 5–7 张表 + 3–4 张图，按以下分布：
   - 表 2.1 技术里程碑时间表（Part I § 2）
   - 表 3.1 代表性医疗 LLM 横向对比（含中文路线）
   - 表 4.1 代表性多模态医疗模型对比
   - 表 5.1 通用 vs 专科模型相对优势矩阵
   - 表 6.1 关键 RCT 总览
   - 表 7.1 评估转向矩阵
   - 表 9.1 四要素 × 三缺口对应矩阵
   - 图 1.1 综述结构总览
   - 图 2.1 训练目标演化
   - 图 4.1 多模态架构拓扑
   - 图 6.1 单独能力 vs 协作收益剪刀差
   - 图 7.1 Tier 框架 + LLM 胜率
   - 图 8.1 训练-治理链路三处不透明
   - 图 9.1 明星模型四要素覆盖度雷达图
4. **引用风格**：保留现有 `[^N]` Markdown 脚注体系；新增脚注（已通过 Crossref / arXiv API 核实）：
   - `[^MedBERT]` Rasmy, L., Xiang, Y., Xie, Z. *et al.* Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. *npj Digital Medicine* **4**, 86 (2021). DOI: 10.1038/s41746-021-00455-y
   - `[^CEHRBERT]` Pang, C., Jiang, X., Kalluri, K. S. *et al.* CEHR-BERT: Incorporating temporal information from structured EHR data to improve prediction tasks. *Proceedings of Machine Learning for Health (ML4H)* (2021). arXiv:2111.08585
   - `[^Foresight2]` Kraljevic, Z., Au Yeung, J., Bean, D., Teo, J. & Dobson, R. J. Large Language Models for Medical Forecasting — Foresight 2. arXiv:2412.10848 (2024)
   - `[^RETFound]` Zhou, Y., Chia, M. A., Wagner, S. K. *et al.* A foundation model for generalizable disease detection from retinal images. *Nature* **622**, 156–163 (2023). DOI: 10.1038/s41586-023-06555-x ✅ 已核实
   - `[^CONCH]` Lu, M. Y., Chen, B., Williamson, D. F. K. *et al.* A visual-language foundation model for computational pathology. *Nature Medicine* **30**, 863–874 (2024). DOI: 10.1038/s41591-024-02856-4 ✅ 已核实
   - `[^CTFM]` Pai, S., Bontempi, D., Hadzic, I. *et al.* Foundation model for cancer imaging biomarkers. *Nature Machine Intelligence* **6**, 354–367 (2024). DOI: 10.1038/s42256-024-00807-9 ✅ 已核实
   - `[^SAMMed]` Cheng, J., Ye, J., Deng, Z. *et al.* SAM-Med2D. arXiv:2308.16184 (2023) ✅ 已核实（未发表期刊版本）
   - `[^MedSAM]` Ma, J., He, Y., Li, F. *et al.* Segment anything in medical images. *Nature Communications* **15** (2024). DOI: 10.1038/s41467-024-44824-z ✅ 已核实
   - `[^HuatuoGPT]` Zhang, H., Chen, J., Jiang, F. *et al.* HuatuoGPT, Towards Taming Language Models To Be a Doctor. *Findings of EMNLP* (2023)
   - `[^BianQue]` Chen, Y. *et al.* BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT. arXiv:2310.15896 (2023)
   - `[^PULSE]` Shanghai AI Laboratory. PULSE: Pretrained and Unified Language Service Engine (medical Chinese LLM). OpenMEDLab technical report / repository, https://github.com/openmedlab/PULSE (2023). **注**：未检索到同行评议版本；以技术报告 + 仓库链接方式引用，正文行文显式标注"未经同行评议"。
   - `[^DISCMed]` Bao, Z., Chen, W., Xiao, S. *et al.* DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation. arXiv:2308.14346 (2023) ✅ 已核实
   - `[^CMB]` Wang, X., Chen, G. H., Song, D. *et al.* CMB: A Comprehensive Medical Benchmark in Chinese. *Proceedings of NAACL 2024 (Main Conference, Long Papers)* (2024). DOI: 10.18653/v1/2024.naacl-long.343 ✅ 已核实（**Main**，非 Findings）
   - `[^Pfohl]` Pfohl, S. R., Cole-Lewis, H. *et al.* A toolbox for surfacing health equity harms and biases in large language models. *Nature Medicine* (2024). DOI: 10.1038/s41591-024-03258-2
   - `[^FUTUREAI]` Lekadir, K. *et al.* FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. *BMJ* (2025). DOI: 10.1136/bmj-2024-081554
   - `[^GPT4VLimits]` Jin, Q., Chen, F., Zhou, Y. *et al.* Hidden flaws behind expert-level accuracy of multimodal GPT-4 vision in medicine. *npj Digital Medicine* **7** (2024). DOI: 10.1038/s41746-024-01185-7 ✅ 已核实（替代 v0.2 中误标的 `[^GPT4Veye]`；同时移除无法核实的 `[^GPT4Vpath]`，§5 中 GPT-4V 局限的证据由此条统一承担）
   - 所有上述条目均已通过 Crossref `/works/{DOI}` 与 arXiv `/api/query` 端点核实；最后投稿前再用 WebFetch 抽样复核。
5. **附录处理**：现稿附录 A/B 转化为正文 §10 之后的"未充分展开的子领域"小节（约 300 字），不再单列；致谢前结尾。

---

> **下一步建议**：骨架 v0.2 已落地。等你最后确认（或对个别章节论点提调整）后，开始写 §1 + §2 作为 Part I 的开篇样本（这两章决定全文叙述节奏与术语规范）；§5 与 §6 是最具新意的两章，建议作为风格审核的关键章节，写完后单独发回给你评审。
