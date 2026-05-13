# Medical AI Survey LaTeX v7

该目录将 `Medical_AI_Survey_Draft_v7.md` 改写为中文 LaTeX 论文。

## 文件结构

- `main.tex`：主文件。
- `sections/`：按章节拆分的正文与参考文献。

## 编译方式

建议使用 `XeLaTeX`：

```bash
cd Medical_AI_Survey_LaTeX_v7
xelatex main.tex
xelatex main.tex
```

如果本机安装了 `latexmk`，也可使用：

```bash
cd Medical_AI_Survey_LaTeX_v7
latexmk -xelatex main.tex
```

## 说明

- 当前环境中未安装 `xelatex` 或 `latexmk`，因此我无法在本地完成编译验证。
- 文中引文沿用数字编号样式，参考文献见 `sections/08-references.tex`。
- 原稿中的图 2.1 未提供图像资源，已在正文中保留为占位图环境，后续可替换为正式图片。
