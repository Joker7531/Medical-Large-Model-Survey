# 医疗人工智能与医疗大模型综述任务：文献检索清单

检索日期：2026-05-09  
截止范围：纳入 2026-04-30 以前公开发表或公开预印本；不纳入 2026-05-01 之后的新文献。  
检索目标：为“医疗人工智能、医疗大模型、医学多模态大模型”综述先建立结构化论文索引。  

## 检索口径

- 优先级 A：Nature、Nature Medicine、NEJM AI、JAMA/JAMA Network、Nature Communications、npj Digital Medicine、Nature Biomedical Engineering，以及 NeurIPS/ICLR/ICML/ACL/CVPR 等顶会或可信官方论文页。
- 优先级 B：顶级团队的重要 arXiv 技术报告、模型论文或数据集论文，尤其是 Med-PaLM、Med-Gemini、LLaVA-Med 等方向性工作。
- 暂不纳入：来源不清、低影响力期刊、明显三区/四区期刊；若为预印本，仅作为“技术进展/模型线索”，后续写综述时需谨慎表述。
- 获取状态说明：`OA` 表示开放获取；`需机构/下载` 表示可能需要机构访问或由用户下载；`预印本` 表示未必经同行评议。

## 一、综述、框架与证据图谱

| 优先级 | 年份 | 题名 | 来源 | 主题 | 链接/DOI | 获取状态 | 备注 |
|---|---:|---|---|---|---|---|---|
| A | 2022 | AI in health and medicine | Nature Medicine | 医疗 AI 总体进展 | https://doi.org/10.1038/s41591-021-01614-0 | 需机构/下载 | 医疗 AI 综述背景，覆盖影像、协作、偏倚等。 |
| A | 2022 | Multimodal biomedical AI | Nature Medicine | 多模态医疗 AI | https://doi.org/10.1038/s41591-022-01981-2 | 需机构/下载 | 多模态医疗 AI 的早期高影响综述，可作为多模态章节引言。 |
| A | 2023 | Foundation models for generalist medical artificial intelligence | Nature | 通用医疗 AI/基础模型 | https://doi.org/10.1038/s41586-023-05881-4 | 需机构/下载 | 提出 GMAI 框架，是医疗基础模型综述的核心引用。 |
| A | 2023 | Large language models in medicine | Nature Medicine | 医疗 LLM 综述 | https://doi.org/10.1038/s41591-023-02448-8 | 需机构/下载 | 面向临床读者的 LLM 医疗应用、限制与风险综述。 |
| A | 2025 | Generative artificial intelligence in medicine | Nature Medicine | 生成式 AI 综述 | https://doi.org/10.1038/s41591-025-03983-2 | 需机构/下载 | 覆盖 agents、reasoning models、MoE 等新一代生成式 AI 医疗方向。 |
| A | 2026 | LLM-assisted systematic review of large language models in clinical medicine | Nature Medicine | LLM 临床证据图谱 | https://doi.org/10.1038/s41591-026-04229-5 | OA | 截至 2025-09 的临床 LLM 研究系统图谱，指出真实临床证据和 RCT 很少。 |
| B | 2025 | From large language models to multimodal AI: a scoping review on the potential of generative AI in medicine | PMC 可检索综述 | 生成式 AI/多模态 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12411359/ | OA | 可作为补充检索线索；写作时优先引用上方顶刊综述。 |

## 二、医学 LLM 与医学问答/临床知识

| 优先级 | 年份 | 题名 | 来源 | 代表模型/资源 | 链接/DOI | 获取状态 | 备注 |
|---|---:|---|---|---|---|---|---|
| A | 2022 | BioGPT: generative pre-trained transformer for biomedical text generation and mining | Briefings in Bioinformatics | BioGPT | https://doi.org/10.1093/bib/bbac409 | 需机构/下载 | 生物医学文本生成与挖掘的早期代表模型。 |
| A | 2022 | A large language model for electronic health records | npj Digital Medicine | GatorTron | https://doi.org/10.1038/s41746-022-00742-2 | OA | EHR 领域 8.9B 临床语言模型，适合 EHR/临床文本章节。 |
| B | 2022 | BioMedLM | Stanford CRFM | BioMedLM/PubMedGPT | https://crfm.stanford.edu/2022/12/15/biomedlm.html | 官方技术页 | 非期刊论文，但常作为开源生物医学 LLM 背景。 |
| A | 2023 | Large language models encode clinical knowledge | Nature | Med-PaLM, MultiMedQA | https://doi.org/10.1038/s41586-023-06291-2 | OA | 医学 LLM 里程碑；提出 MultiMedQA 与临床长答案人工评价框架。 |
| A | 2025 | Toward expert-level medical question answering with large language models | Nature Medicine | Med-PaLM 2 | https://doi.org/10.1038/s41591-024-03423-7 | OA | Med-PaLM 2，含 ensemble refinement、chain of retrieval、长答案临床评价。 |
| B | 2024 | Capabilities of Gemini Models in Medicine | arXiv / Google | Med-Gemini | https://doi.org/10.48550/arXiv.2404.18416 | 预印本 | 多任务、多模态、长上下文医疗能力报告；影响大但写作需标注预印本。 |
| A | 2026 | Holistic evaluation of large language models for medical tasks with MedHELM | Nature Medicine | MedHELM | https://doi.org/10.1038/s41591-025-04151-2 | 需机构/下载 | 医疗 LLM 综合评测框架，覆盖真实临床任务分类。 |
| A | 2026 | a large language model for complex cardiology care | Nature Medicine | 心血管专科 LLM | https://doi.org/10.1038/s41591-025-04190-9 | OA | 专科医学 LLM 的高影响临床方向论文。 |

## 三、多模态医疗大模型、医学视觉语言模型与影像

| 优先级 | 年份 | 题名 | 来源 | 代表模型/资源 | 链接/DOI | 获取状态 | 备注 |
|---|---:|---|---|---|---|---|---|
| A | 2023 | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day | NeurIPS 2023 Datasets and Benchmarks, Spotlight | LLaVA-Med | https://papers.nips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html | OA | 医学视觉指令微调代表作，顶会 Spotlight。 |
| A | 2024 | Towards Generalist Biomedical AI | NEJM AI | Med-PaLM M, MultiMedBench | https://doi.org/10.1056/AIoa2300138 | 需机构/下载 | 多模态通用生物医学 AI 里程碑；文本、影像、基因组等 14 任务。 |
| A | 2024 | A generalist vision-language foundation model for diverse biomedical tasks | Nature Medicine | BiomedGPT | https://doi.org/10.1038/s41591-024-03185-2 | 需机构/下载 | 开源轻量级医学 VLM；多任务、多模态、人类评价。 |
| B | 2024 | Advancing Multimodal Medical Capabilities of Gemini | arXiv / Google | Med-Gemini-2D/3D/Polygenic | https://doi.org/10.48550/arXiv.2405.03162 | 预印本 | 医学影像、3D CT、病理、眼科、皮肤科、基因风险预测。 |
| A | 2025 | A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image-Text Pairs | NEJM AI | PMC-15M, BiomedCLIP | https://doi.org/10.1056/AIoa2400640 | 需机构/下载 | 1500 万生物医学图文对；BiomedCLIP，适合数据规模与开放模型章节。 |
| B | 2024 | MAIRA-2: Grounded Radiology Report Generation | arXiv / Microsoft | MAIRA-2 | https://doi.org/10.48550/arXiv.2406.04449 | 预印本 | 放射报告生成与 grounding；若综述写影像生成报告可列为技术进展。 |
| B | 2023 | Med-Flamingo: a Multimodal Medical Few-shot Learner | arXiv | Med-Flamingo | https://doi.org/10.48550/arXiv.2307.15189 | 预印本 | 医学 few-shot VLM；可作为早期多模态大模型补充。 |

## 四、临床交互、人机协作与真实任务评估

| 优先级 | 年份 | 题名 | 来源 | 任务 | 链接/DOI | 获取状态 | 备注 |
|---|---:|---|---|---|---|---|---|
| A | 2023 | Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum | JAMA Internal Medicine | 患者问答、质量与同理心 | https://doi.org/10.1001/jamainternmed.2023.1838 | OA | Chatbot 与医生回答盲评比较；适合讨论患者沟通但需注意非临床场景。 |
| A | 2024 | Large language models for preventing medication direction errors in online pharmacies | Nature Medicine | 药房处方说明错误 | https://doi.org/10.1038/s41591-024-02933-8 | OA | 真实工作流相关，属于较强的临床安全应用证据。 |
| A | 2024 | Large Language Model Influence on Diagnostic Reasoning: A Randomized Clinical Trial | JAMA Network Open | 医生诊断推理 RCT | https://doi.org/10.1001/jamanetworkopen.2024.40969 | OA | LLM 辅助医生未显著提升诊断推理，LLM 单独表现较强；人机协作章节关键反例。 |
| A | 2025 | Towards conversational diagnostic artificial intelligence | Nature | AMIE | https://doi.org/10.1038/s41586-025-08866-7 | OA | 诊断对话 AI，随机双盲交叉 OSCE 风格研究；需强调模拟文本问诊限制。 |
| A | 2026 | Reliability of LLMs as medical assistants for the general public: a randomized preregistered study | Nature Medicine | 面向公众的医疗助手可靠性 | https://doi.org/10.1038/s41591-025-04074-y | OA | 患者/公众使用 LLM 医疗建议的高影响研究。 |
| A | 2026 | An LLM chatbot to facilitate primary-to-specialist care transitions: a randomized controlled trial | Nature Medicine | 基层到专科转诊 RCT | https://doi.org/10.1038/s41591-025-04176-7 | 需机构/下载 | 临床转诊流程 RCT，适合“真实临床部署证据”章节。 |
| A | 2026 | A cognitive layer architecture to support large-language model performance in psychotherapy interactions | Nature Medicine | 心理治疗交互 | https://doi.org/10.1038/s41591-026-04278-w | 需机构/下载 | LLM 在心理治疗对话中的架构与评估。 |

## 五、安全、可靠性、偏倚与攻击面

| 优先级 | 年份 | 题名 | 来源 | 风险类型 | 链接/DOI | 获取状态 | 备注 |
|---|---:|---|---|---|---|---|---|
| A | 2025 | Large Language Models lack essential metacognition for reliable medical reasoning | Nature Communications | 过度自信、元认知不足 | https://doi.org/10.1038/s41467-024-55628-6 | OA | MetaMedQA；讨论“高准确率不等于可靠临床推理”的核心文献。 |
| A | 2025 | Medical large language models are vulnerable to data-poisoning attacks | Nature Medicine | 数据投毒/安全攻击 | https://doi.org/10.1038/s41591-024-03445-1 | OA | 医疗 LLM 供应链与训练数据安全。 |
| B | 2025 | Limitations of large language models in clinical problem-solving arising from inflexible reasoning | Scientific Reports | 灵活推理不足、幻觉、过度自信 | https://doi.org/10.1038/s41598-025-22940-0 | OA | 可作补充风险文献；优先级低于 Nature Medicine/Nature Communications。 |
| A | 2026 | Limitations of Large Language Models in Clinical Diagnostic Reasoning | JAMA Network Open | 诊断推理限制 | https://doi.org/10.1001/jamanetworkopen.2026.4014 | 需机构/下载 | 2026-04 发表，适合补充“诊断推理局限”。 |
| A | 2024 | Large Language Models-Misdiagnosing Diagnostic Excellence? | JAMA Network Open | 评论/临床解释 | https://doi.org/10.1001/jamanetworkopen.2024.40901 | OA | 配套 Goh 等 RCT 的评论，可用于解释 LLM 与医生协作落差。 |

## 六、可优先下载全文的候选清单

建议先下载/整理以下 15 篇全文，基本可支撑综述主体结构：

1. Moor et al., 2023, `Foundation models for generalist medical artificial intelligence`, Nature.
2. Thirunavukarasu et al., 2023, `Large language models in medicine`, Nature Medicine.
3. Acosta et al., 2022, `Multimodal biomedical AI`, Nature Medicine.
4. Chen et al., 2026, `LLM-assisted systematic review of large language models in clinical medicine`, Nature Medicine.
5. Singhal et al., 2023, `Large language models encode clinical knowledge`, Nature.
6. Singhal et al., 2025, `Toward expert-level medical question answering with large language models`, Nature Medicine.
7. Tu et al., 2024, `Towards Generalist Biomedical AI`, NEJM AI.
8. Zhang et al., 2024, `A generalist vision-language foundation model for diverse biomedical tasks`, Nature Medicine.
9. Zhang et al., 2025, `A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image-Text Pairs`, NEJM AI.
10. Li et al., 2023, `LLaVA-Med`, NeurIPS 2023.
11. Tu et al., 2025, `Towards conversational diagnostic artificial intelligence`, Nature.
12. Goh et al., 2024, `Large Language Model Influence on Diagnostic Reasoning`, JAMA Network Open.
13. Pais et al., 2024, `Large language models for preventing medication direction errors in online pharmacies`, Nature Medicine.
14. Bedi et al., 2026, `Holistic evaluation of large language models for medical tasks with MedHELM`, Nature Medicine.
15. Griot et al., 2025, `Large Language Models lack essential metacognition for reliable medical reasoning`, Nature Communications.

## 七、后续综述写作建议结构

- 背景：从传统医疗 AI 到基础模型、生成式 AI、GMAI。
- 医疗 LLM：生物医学文本预训练、临床知识问答、Med-PaLM/Med-PaLM 2/Med-Gemini。
- 多模态医疗大模型：医学图文数据、VLM、放射报告、病理/眼科/皮肤科、基因组与 EHR 长上下文。
- 评测体系：MultiMedQA、MultiMedBench、MedHELM、人工医生评价、真实临床数据与 RCT 证据分层。
- 临床应用：患者问答、诊断推理、转诊、药房安全、心理治疗、专科 care。
- 风险治理：幻觉、过度自信、元认知不足、偏倚、公平性、数据污染、隐私与监管。
- 未来方向：真实世界前瞻性验证、人机协作界面、可追溯检索增强、模型更新与监管闭环。

## 八、暂不纳入或仅作背景的条目

- 2026-05-01 以后发表的论文不纳入本轮清单，即使主题相关。
- 低影响力或来源不明的“医疗 LLM 综述”暂不纳入主体引用。
- 单病种、单科室、样本很小的 ChatGPT 考试类论文数量很多，后续可按专科需要补充，但不建议作为综述主干。
