# ELF Transformer 去噪机制：公式与维度

ELF 的核心不是让 Transformer 在一次前向中反复去噪，而是：

> Transformer 根据当前带噪 latent $\mathbf Z_t$ 直接预测干净端点
> $\widehat{\mathbf X}_0$；外层 ODE sampler 再把该预测换算成速度，沿
> $t:0\rightarrow1$ 多步更新 $\mathbf Z_t$。

## 1. 符号与维度

| 符号 | 含义 | 维度 |
|---|---|---|
| $B$ | batch size | 标量 |
| $M$ | source 与 target 拼接后的序列长度 | 标量 |
| $D$ | encoder latent 维度；当前 T5-small 为 $512$ | 标量 |
| $H$ | Transformer hidden size；ELF-B 为 $768$ | 标量 |
| $A$ | attention head 数；ELF-B 为 $12$ | 标量 |
| $d_h=H/A$ | 每个 attention head 的维度；ELF-B 为 $64$ | 标量 |
| $R$ | 输入 bottleneck 维度；默认 $128$ | 标量 |
| $P$ | time、CFG 和 mode control token 总数；当前默认 $12$ | 标量 |
| $N=M+P$ | Transformer 实际处理的总长度 | 标量 |
| $\mathbf X_0$ | 归一化后的干净文本 latent | $\mathbb R^{B\times M\times D}$ |
| $\mathbf Z_t$ | 时刻 $t$ 的带噪 latent | $\mathbb R^{B\times M\times D}$ |
| $t$ | Flow Matching 时间，$0$ 为噪声、$1$ 为数据 | $\mathbb R^B$ |

## 2. 从文本构造带噪 latent

离散 token 序列 $\mathbf S\in\{1,\ldots,|\mathcal V|\}^{B\times M}$ 先经过冻结的
上下文 encoder：

$$
\mathbf X_0
=\frac{E(\mathbf S)-\mu}{s}
\in\mathbb R^{B\times M\times D}.
$$

$E$ 是 T5 encoder，$\mu$ 和 $s$ 是预先统计的 latent 均值与标准差。
训练时采样 $\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\mathbf I)$，并使用线性
Flow Matching 路径：

$$
\mathbf Z_t
=t\mathbf X_0+(1-t)\sigma\boldsymbol\varepsilon,
\qquad
\mathbf Z_t\in\mathbb R^{B\times M\times D}.
$$

$\sigma$ 是噪声尺度，当前正式配置取 $\sigma=2$。因此
$\mathbf Z_0=\sigma\boldsymbol\varepsilon$，$\mathbf Z_1=\mathbf X_0$。

条件生成时，令 $\mathbf C\in\{0,1\}^{B\times M\times1}$ 表示干净 source 位置：

$$
\mathbf Z_t^{\mathrm{cond}}
=\mathbf C\odot\mathbf X_0
+(1-\mathbf C)\odot\mathbf Z_t.
$$

即 source latent 始终固定，只有 target latent 从噪声流向数据。

## 3. Transformer 如何预测干净端点

### 3.1 输入与控制 token

若启用 self-conditioning，将上一步的干净预测
$\widehat{\mathbf X}_0^{\mathrm{prev}}$ 与当前状态拼接后投影：

$$
\bar{\mathbf Z}_t
=\bigl[\mathbf Z_t;\widehat{\mathbf X}_0^{\mathrm{prev}}\bigr]
\mathbf W_{\mathrm{sc}}+\mathbf b_{\mathrm{sc}}
\in\mathbb R^{B\times M\times D},
\qquad
\mathbf W_{\mathrm{sc}}\in\mathbb R^{2D\times D}.
$$

首个采样步令 target 位置的 $\widehat{\mathbf X}_0^{\mathrm{prev}}=\mathbf0$；条件生成的
source 位置仍恢复为干净 latent。不使用 self-conditioning 时直接令
$\bar{\mathbf Z}_t=\mathbf Z_t$。

文本 latent 通过无激活的 bottleneck 投影到 Transformer hidden space：

$$
\mathbf H_{\mathrm{text}}^{(0)}
=\left(\bar{\mathbf Z}_t\mathbf W_1\right)\mathbf W_2+\mathbf b_2
\in\mathbb R^{B\times M\times H},
$$

其中 $\mathbf W_1\in\mathbb R^{D\times R}$，
$\mathbf W_2\in\mathbb R^{R\times H}$。时间 $t$ 被编码为
$\mathbf e_t\in\mathbb R^{B\times H}$，加到 $P_t=4$ 个可学习 time tokens 上。
再拼接 CFG tokens 和 mode tokens，得到

$$
\mathbf H^{(0)}
=\bigl[\mathbf C_t;\mathbf C_{\mathrm{cfg}};\mathbf C_{\mathrm{mode}};
\mathbf H_{\mathrm{text}}^{(0)}\bigr]
\in\mathbb R^{B\times N\times H}.
$$

这些 control tokens 使同一个 Transformer 知道当前噪声时刻和工作模式；时间条件不是
通过每层 AdaLN 注入的。

### 3.2 一层 ELF Transformer

对第 $\ell$ 层，先做 pre-norm 多头双向自注意力：

$$
\begin{aligned}
\mathbf U
&=\operatorname{RMSNorm}(\mathbf H^{(\ell-1)})
\in\mathbb R^{B\times N\times H},\\
[\mathbf Q,\mathbf K,\mathbf V]
&=\operatorname{reshape}_A(\mathbf U\mathbf W_{QKV}+\mathbf b_{QKV}),\\
\mathbf Q,\mathbf K,\mathbf V
&\in\mathbb R^{B\times A\times N\times d_h},\\
\widetilde{\mathbf Q},\widetilde{\mathbf K}
&=\operatorname{RoPE}\!\left(
\operatorname{RMSNorm}(\mathbf Q),
\operatorname{RMSNorm}(\mathbf K)\right),\\
\mathbf A_{\mathrm{attn}}
&=\operatorname{softmax}_{j}\!\left(
\frac{\widetilde{\mathbf Q}\widetilde{\mathbf K}^{\mathsf T}}{\sqrt{d_h}}
+\mathbf M_{\mathrm{attn}}\right)
\in\mathbb R^{B\times A\times N\times N},\\
\widetilde{\mathbf H}^{(\ell)}
&=\mathbf H^{(\ell-1)}
+\operatorname{ConcatHeads}(\mathbf A_{\mathrm{attn}}\mathbf V)\mathbf W_O
\in\mathbb R^{B\times N\times H}.
\end{aligned}
$$

$\mathbf W_{QKV}\in\mathbb R^{H\times3H}$，
$\mathbf W_O\in\mathbb R^{H\times H}$；
$\mathbf M_{\mathrm{attn}}\in\{0,-\infty\}^{B\times1\times1\times N}$ 屏蔽 padding key。
这里没有 causal mask，因此每个 target 位置可同时读取整个有效序列，并利用其他 noisy
target 与干净 source 的上下文。

随后执行 pre-norm SwiGLU FFN。令 $F$ 为 SwiGLU 中间维度：

$$
\begin{aligned}
\mathbf G
&=\operatorname{RMSNorm}(\widetilde{\mathbf H}^{(\ell)})
\in\mathbb R^{B\times N\times H},\\
[\mathbf G_1,\mathbf G_2]
&=\mathbf G\mathbf W_{12}+\mathbf b_{12},
\qquad
\mathbf G_1,\mathbf G_2\in\mathbb R^{B\times N\times F},\\
\mathbf H^{(\ell)}
&=\widetilde{\mathbf H}^{(\ell)}
+\left[\operatorname{SiLU}(\mathbf G_1)\odot\mathbf G_2\right]
\mathbf W_3+\mathbf b_3.
\end{aligned}
$$

$\mathbf W_{12}\in\mathbb R^{H\times2F}$，
$\mathbf W_3\in\mathbb R^{F\times H}$。ELF-B 串联 $L=12$ 层；这些层负责综合全序列
信息，但一次前向只产生一次干净端点估计。

### 3.3 输出 clean latent

移除前面的 $P$ 个 control tokens，仅保留文本位置：

$$
\widehat{\mathbf X}_0
=f_\theta(\mathbf Z_t,t)
=\operatorname{RMSNorm}\!\left(
\mathbf H^{(L)}_{[:,P:P+M,:]}\right)\mathbf W_{\mathrm{out}}
+\mathbf b_{\mathrm{out}}
\in\mathbb R^{B\times M\times D},
$$

其中 $\mathbf W_{\mathrm{out}}\in\mathbb R^{H\times D}$。因此 Transformer 的直接
输出是预测的干净 latent，不是噪声，也不是速度。

## 4. 训练目标：由 clean prediction 换算速度

线性路径的真实速度为

$$
\mathbf V_t^*
=\frac{\partial\mathbf Z_t}{\partial t}
=\mathbf X_0-\sigma\boldsymbol\varepsilon
=\frac{\mathbf X_0-\mathbf Z_t}{1-t}
\in\mathbb R^{B\times M\times D}.
$$

模型输出先换算为预测速度：

$$
\widehat{\mathbf V}_t
=\frac{\widehat{\mathbf X}_0-\mathbf Z_t}
{\max(1-t,\epsilon_t)},
\qquad \epsilon_t=0.05.
$$

令 $q_{bi}=a_{bi}(1-c_{bi})$，其中 $a_{bi}$ 表示有效 token，$c_{bi}$ 表示干净
source；实际 denoising loss 为

$$
\mathcal L_{\mathrm{FM}}
=\frac{1}{\sum_{b,i}q_{bi}}
\sum_{b,i}q_{bi}
\frac{\left\lVert
\widehat{\mathbf V}_{t,bi:}-\mathbf V^*_{t,bi:}
\right\rVert_2^2}{D}.
$$

忽略 $t\rightarrow1$ 时的数值截断，上式等价于对 clean prediction 使用
$1/(1-t)^2$ 加权的 MSE。模型虽然输出 $\widehat{\mathbf X}_0$，真正监督的仍是
Flow Matching 速度。

## 5. 推理时怎样逐步去噪

取时间网格 $0=t_0<t_1<\cdots<t_K=1$，初始化
$\mathbf Z_{t_0}=\sigma\boldsymbol\varepsilon$。第 $k$ 个 ODE Euler step 为

$$
\begin{aligned}
\widehat{\mathbf X}_0^{(k)}
&=f_\theta\!\left(
\bigl[\mathbf Z_{t_k};\widehat{\mathbf X}_0^{(k-1)}\bigr],t_k
\right),\\
\widehat{\mathbf V}_{t_k}
&=\frac{\widehat{\mathbf X}_0^{(k)}-\mathbf Z_{t_k}}
{\max(1-t_k,\epsilon_t)},\\
\mathbf Z_{t_{k+1}}
&=\mathbf Z_{t_k}
+(t_{k+1}-t_k)\widehat{\mathbf V}_{t_k}.
\end{aligned}
$$

这三行就是 ELF 的完整去噪主循环。若暂时忽略 $\epsilon_t$，令
$\alpha_k=(t_{k+1}-t_k)/(1-t_k)$，则

$$
\mathbf Z_{t_{k+1}}
=(1-\alpha_k)\mathbf Z_{t_k}
+\alpha_k\widehat{\mathbf X}_0^{(k)}.
$$

因此每一步都把当前 noisy latent 向 Transformer 预测的 clean endpoint 推近；下一步
Transformer 再根据更新后的全序列重新估计端点。条件生成时 source 位置的
$\widehat{\mathbf V}$ 被置零，故只有 target 位置发生移动。

## 6. Self-conditioning、CFG 与最终解码

- **Self-conditioning**：将上一步的 $\widehat{\mathbf X}_0$ 作为下一步额外输入，帮助
  Transformer 修正自己的历史估计；它不是新的扩散状态或独立 loss。
- **CFG**：基本形式为
  $\mathbf V_{\mathrm{cfg}}=\omega\mathbf V_{\mathrm{cond}}+(1-\omega)
  \mathbf V_{\mathrm{uncond}}$，只改变采样所用速度，不改变上述 Flow Matching 路径。
- **离散解码**：完成 ODE 后，在 $t=1$ 切换到 decode mode：

$$
\begin{aligned}
\mathbf Y
&=\operatorname{GELU}(\mathbf H_{\mathrm{text}}^{(L)}\mathbf W_p+\mathbf b_p)
\in\mathbb R^{B\times M\times D},\\
\mathbf L
&=\mathbf Y\mathbf W_u+\mathbf b_u
\in\mathbb R^{B\times M\times|\mathcal V|},\\
\widehat{\mathbf S}_{bi}
&=\arg\max_v L_{biv}.
\end{aligned}
$$

去噪始终发生在连续 latent 空间；只有最后一步才映射回离散 token。

## 核心结论

$$
\boxed{
\mathbf Z_t
\xrightarrow{\text{Transformer}}
\widehat{\mathbf X}_0
\xrightarrow{(\widehat{\mathbf X}_0-\mathbf Z_t)/(1-t)}
\widehat{\mathbf V}_t
\xrightarrow{\text{Euler}}
\mathbf Z_{t+\Delta t}
}
$$

Transformer 的职责是基于全序列上下文预测 clean endpoint；Flow Matching 定义学习目标，
ODE sampler 才负责把高斯噪声逐步推进为文本 latent。

实现对应：`src/modules/model.py`、`src/modules/layers.py`、`src/train_step.py`、
`src/utils/sampling_utils.py`；理论对应 ELF 论文 Sec. 3.1-3.3 与 Appendix C。
