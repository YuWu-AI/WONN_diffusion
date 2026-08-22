# 当前 ELF-WONN 架构

下面按照一次 denoising forward 的实际顺序展开。$B$ 是 batch size，$M$ 是
source 与 target 拼接后的序列长度，$D$ 是 T5 latent 维度，$K$ 是每个 token 的
振子数。Phase 5 正式实验中，$M=128$、$D=512$、$K=384$。

## 1. 构造带噪 latent

$$
\mathbf Z_t
=\mathbf c\odot\mathbf X_0
+(1-\mathbf c)\odot
\left[t\mathbf X_0+(1-t)\sigma\boldsymbol\varepsilon\right],
\qquad
\mathbf Z_t\in\mathbb R^{B\times M\times D}.
$$

$\mathbf X_0\in\mathbb R^{B\times M\times D}$ 是 T5 编码的干净 latent；
$\boldsymbol\varepsilon$ 是同维标准高斯噪声；$t\in[0,1]$ 是 Flow Matching 时间；
$\sigma=2$ 是噪声尺度；$\mathbf c\in\{0,1\}^{B\times M\times1}$ 是 source mask。
因此 source 始终保持干净，target 在 $t=0$ 时为纯噪声，在 $t=1$ 时为干净 latent。

## 2. 初始化振子状态

$$
\begin{aligned}
\bar{\mathbf Z}_t
&=\Pi_{\mathrm{sc}}\!\left(
[\mathbf Z_t;\widehat{\mathbf X}^{\mathrm{prev}}_0]
\right)
\in\mathbb R^{B\times M\times D},\\
\mathbf H^{(0)}
&=[\mathbf C_t;\Pi_{\mathrm{text}}(\bar{\mathbf Z}_t)]
\in\mathbb R^{B\times N\times2K},
\qquad N=M+P,\\
\mathbf A
&=\operatorname{reshape}_{K\times2}\!\left(
\operatorname{RMSNorm}(\mathbf H^{(0)})\mathbf W_{\theta}
+\mathbf b_{\theta}\right),\\
\Theta^{(0)}_{bik}
&=\operatorname{atan2}(A_{bik2},A_{bik1}),
\qquad
\boldsymbol\Theta^{(0)}\in[-\pi,\pi]^{B\times N\times K},\\
\boldsymbol\Omega^{(0)}
&=\operatorname{RMSNorm}(\mathbf H^{(0)})\mathbf W_{\omega}
+\mathbf b_{\omega}
\in\mathbb R^{B\times N\times K}.
\end{aligned}
$$

$\widehat{\mathbf X}^{\mathrm{prev}}_0$ 是上一次 self-conditioning 预测；不用
self-conditioning 时令 $\bar{\mathbf Z}_t=\mathbf Z_t$。$\Pi_{\mathrm{sc}}:2D\to D$，
$\Pi_{\mathrm{text}}:D\to128\to2K$；$\mathbf C_t$ 是 $P=12$ 个 time、CFG 和 mode
control tokens，因此 $N=M+P=140$。$\mathbf W_\theta\in\mathbb R^{2K\times2K}$，
$\mathbf W_\omega\in\mathbb R^{2K\times K}$，$\mathbf b$ 表示对应输出维度的可学习
偏置。每个位置最终被表示为 $K$ 个相位 $\boldsymbol\Theta$ 和 $K$ 个频率
$\boldsymbol\Omega$；$b,i,k$ 分别表示 batch、位置和振子索引。

## 3. 进行 Winfree 耦合

对第 $\ell$ 层的第 $r$ 个 inner step：

$$
\begin{aligned}
\mathbf p_{bik}
&=[\sin\Theta_{bik},\cos\Theta_{bik}],\\
\mathbf S_{bik}
&=\operatorname{MLP}_{S,k}(\mathbf p_{bik}),
\qquad
\mathbf I_{bik}=\operatorname{MLP}_{I,k}(\mathbf p_{bik}),\\
[\mathbf Q;\mathbf K_{\mathrm{key}};\mathbf V_{\mathrm{attn}}]
&=\mathbf I\mathbf W_{qkv}+\mathbf b_{qkv}
\in\mathbb R^{B\times N\times3K},\\
\mathbf Q,\mathbf K_{\mathrm{key}}
&=\operatorname{RoPE}_{1D}(\mathbf Q,\mathbf K_{\mathrm{key}})
\in\mathbb R^{B\times H\times N\times C},\\
\mathbf A_{bhij}
&=\operatorname{softmax}_{j}\!\left(
\frac{\mathbf Q_{bhi:}\mathbf K_{\mathrm{key},bhj:}^{\mathsf T}}
{\sqrt C}\right),\\
\mathbf M
&=\operatorname{MergeHeads}(\mathbf A\mathbf V_{\mathrm{attn}})
\in\mathbb R^{B\times N\times K},\\
\mathbf F
&=\operatorname{ReLU}\!\left(
\operatorname{RMSNorm}(\mathbf M\mathbf W_o+\mathbf b_o)
\right)
\in\mathbb R^{B\times N\times K},\\
\mathbf V^{(\ell,r)}
&=\boldsymbol\Omega^{(\ell)}
+\mathbf S^{(\ell,r)}\odot\mathbf F^{(\ell,r)},\\
\boldsymbol\Theta^{(\ell,r+1)}
&=\operatorname{wrap}_{[-\pi,\pi]}\!\left(
\boldsymbol\Theta^{(\ell,r)}
+\Delta_{\ell}\mathbf V^{(\ell,r)}\right).
\end{aligned}
$$

$\operatorname{MLP}_{S,k}$ 与 $\operatorname{MLP}_{I,k}$ 是第 $k$ 个振子的独立网络，结构分别为
$2\to2\to1$ 和 $2\to4\to1$，中间使用 ReLU，输出不加 $\tanh$。$H$ 是 head 数，
$C=K/H$ 同时是每个 head 的振子数和 Q/K/V 维度；正式配置为 $H=12$、$C=32$。
$\mathbf W_{qkv}\in\mathbb R^{K\times3K}$，$\mathbf W_o\in\mathbb R^{K\times K}$。
padding mask 只屏蔽 key 位置；普通路径使用 SDPA，只有 diagnostics 显式计算 $\mathbf A$。
$\mathbf F$ 不包含 influence residual、phase residual 或 Transformer FFN。
$\Delta_\ell\in(0,0.25)$ 是第 $\ell$ 层的可学习步长。

## 4. 更新层间频率

一层完成 $T$ 个 inner steps 后，在非末层更新频率：

$$
\begin{aligned}
\mathbf E_\theta^{(\ell)}
&=\operatorname{ThetaEmbedding}\!\left(
[\sin\boldsymbol\Theta^{(\ell,T)};
\cos\boldsymbol\Theta^{(\ell,T)}]
\right)
\in\mathbb R^{B\times N\times K},\\
\mathbf U^{(\ell)}
&=\operatorname{RMSNorm}\!\left(
[\mathbf E_\theta^{(\ell)};\boldsymbol\Omega^{(\ell)}]
\right)
\in\mathbb R^{B\times N\times2K},\\
\mathbf h^{(\ell)}
&=\operatorname{ReLU}\!\left(
\mathbf U^{(\ell)}\mathbf W_1^{(\ell)}+\mathbf b_1^{(\ell)}
\right)
\in\mathbb R^{B\times N\times K},\\
\Delta\boldsymbol\Omega^{(\ell)}
&=\mathbf h^{(\ell)}\mathbf W_2^{(\ell)}+\mathbf b_2^{(\ell)},\\
\alpha_\ell
&=0.25\,\operatorname{sigmoid}(a_\ell),\\
\boldsymbol\Omega^{(\ell+1)}
&=\boldsymbol\Omega^{(\ell)}
+\alpha_\ell\Delta\boldsymbol\Omega^{(\ell)},\\
\boldsymbol\Theta^{(\ell+1,0)}
&=\boldsymbol\Theta^{(\ell,T)}.
\end{aligned}
$$

$\operatorname{ThetaEmbedding}$ 对每个振子独立执行 $2\to1$ 仿射映射，不混合 token 或振子；
$\mathbf W_1^{(\ell)}\in\mathbb R^{2K\times K}$，
$\mathbf W_2^{(\ell)}\in\mathbb R^{K\times K}$。$\alpha_\ell$ 初值为 $0.1$，上界为 $0.25$。
transition 不使用第二套 attention、卷积、末端 $\tanh$ 或零初始化 gate。
正式模型共有 $L=6$ 层、每层 $T=3$ 步，因此执行 18 次 coupling 和 5 次层间
frequency update；最后一层不再更新 frequency。

## 5. 输出 clean latent 并计算 loss

$$
\begin{aligned}
\mathbf H_{\mathrm{out}}
&=[\sin\boldsymbol\Theta^*;\cos\boldsymbol\Theta^*]_{[:,P:P+M,:]}
\in\mathbb R^{B\times M\times2K},\\
\widehat{\mathbf X}_0
&=\operatorname{Linear}_{\mathrm{out}}\!\left(
\operatorname{RMSNorm}(\mathbf H_{\mathrm{out}})\right)
\in\mathbb R^{B\times M\times D},\\
\widehat{\mathbf V}_t
&=\frac{\widehat{\mathbf X}_0-\mathbf Z_t}
{\max(1-t,\epsilon)},
\qquad
\mathbf V_t^*=\frac{\mathbf X_0-\mathbf Z_t}
{\max(1-t,\epsilon)},\\
\mathcal L_{\mathrm{FM}}
&=\frac{1}{\sum_{b,i}q_{bi}}
\sum_{b,i}q_{bi}
\frac{\lVert\widehat{\mathbf V}_{t,bi:}
-\mathbf V^*_{t,bi:}\rVert_2^2}{D}.
\end{aligned}
$$

$\boldsymbol\Theta^*$ 是最后一次 coupling 后的相位；$\epsilon=0.05$ 防止分母过小；
$q_{bi}=p_{bi}(1-c_{bi})\in\{0,1\}$ 是 target loss mask，$p_{bi}$ 是有效 token
mask。$\operatorname{Linear}_{\mathrm{out}}:2K\to D$。模型只在有效 target 位置计算
Flow Matching velocity MSE，最终只读取相位特征，不读取最终 frequency。
