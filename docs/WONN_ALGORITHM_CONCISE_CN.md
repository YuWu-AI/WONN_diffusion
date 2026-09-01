# ELF-WONN：训练与取样公式简版

> T5 latent 加噪 $\rightarrow$ WONN 预测 clean latent $\rightarrow$ Flow Matching $\rightarrow$ ODE 取样 $\rightarrow$ token。

## 0. 量与维度

| 量 | 维度 | Phase 5 正式配置 |
|---|---:|---:|
| $\mathbf S$；$\mathbf X_0,\mathbf Z_t,\widehat{\mathbf X}_0$ | $B\times M$；$B\times M\times D$ | $M=128,D=512$ |
| $\boldsymbol\Theta,\boldsymbol\Omega$ | $B\times N\times K$ | $K=384$ |
| $\mathbf S_\theta,\mathbf I_\theta,\mathbf F$ | $B\times N\times K$ | sensitivity / influence / field |
| $\mathbf Q,\mathbf K_a,\mathbf V_a$ | $B\times A\times N\times C$ | $A=12,C=K/A=32$ |
| $\mathbf A_{\rm attn}$ | $B\times A\times N\times N$ | attention |
| control / total length | $P,N$ | $P=12,N=M+P=140$ |
| layers / inner steps | $L,T$ | $L=6,T=3$ |

$\mathbf c\in\{0,1\}^{B\times M\times1}$ 为 source mask，且

$$
\mathcal R_{\mathbf c}(\mathbf Y,\mathbf X)
=\mathbf c\odot\mathbf X+(1-\mathbf c)\odot\mathbf Y
$$

表示 source 取 $\mathbf X$、target 取 $\mathbf Y$。

---

## 1. 训练顺序

### 1.1 token → noisy latent

$$
\begin{aligned}
\mathbf X_0
&=\frac{E_{\rm T5}(\mathbf S)-\mu}{s}
\in\mathbb R^{B\times M\times D},\\
t_b&=\operatorname{sigmoid}(u_b),\qquad u_b\sim\mathcal N(-1.5,0.8^2),\\
\boldsymbol\varepsilon&\sim\mathcal N(\mathbf0,\mathbf I),\\
\mathbf Z_t
&=\mathcal R_{\mathbf c}\!\left(
t\mathbf X_0+(1-t)\,2\boldsymbol\varepsilon,\mathbf X_0\right)
\in\mathbb R^{B\times M\times D},\\
m_b&\sim\operatorname{Bernoulli}(0.5),\\
\mathbf X_{\rm sc}
&=\mathcal R_{\mathbf c}\!\left(
m\,\operatorname{stopgrad}(\widehat{\mathbf X}_0^{\rm init}),\mathbf X_0\right),\\
\widehat{\mathbf X}_0&=f_\phi(\mathbf Z_t,t,\mathbf X_{\rm sc}).
\end{aligned}
$$

### 1.2 WONN 初始化

$$
\begin{aligned}
\bar{\mathbf Z}_t
&=\Pi_{\rm sc}([\mathbf Z_t;\mathbf X_{\rm sc}])
\in\mathbb R^{B\times M\times D},
&\Pi_{\rm sc}&:2D\to D,\\
\mathbf H^{(0)}
&=[\mathbf C_t;\mathbf C_{\rm sc};\mathbf C_{\rm mode};
\Pi_{\rm text}(\bar{\mathbf Z}_t)]
\in\mathbb R^{B\times N\times2K},
&\Pi_{\rm text}&:D\to128\to2K,\\
\mathbf U^{(0)}&=\operatorname{RMSNorm}(\mathbf H^{(0)}),\\
\mathbf P_\theta
&=\operatorname{reshape}_{K\times2}
(\mathbf U^{(0)}\mathbf W_\theta+\mathbf b_\theta),
&\mathbf W_\theta&\in\mathbb R^{2K\times2K},\\
\Theta^{(0)}_{bik}
&=\operatorname{atan2}(P_{\theta,bik,2},P_{\theta,bik,1}),\\
\boldsymbol\Omega^{(0)}
&=\mathbf U^{(0)}\mathbf W_\omega+\mathbf b_\omega
\in\mathbb R^{B\times N\times K},
&\mathbf W_\omega&\in\mathbb R^{2K\times K}.
\end{aligned}
$$

### 1.3 WONN 层：$\ell=0,\ldots,L-1$，$r=0,\ldots,T-1$

$$
\begin{aligned}
\mathbf p_{bik}&=[\sin\Theta_{bik},\cos\Theta_{bik}]\in\mathbb R^2,\\
\mathbf S_\theta&=\operatorname{MLP}_{S,k}(\mathbf p),\quad 2\to2\to1,
&\mathbf I_\theta&=\operatorname{MLP}_{I,k}(\mathbf p),\quad 2\to4\to1,\\
[\mathbf Q;\mathbf K_a;\mathbf V_a]
&=\operatorname{SplitHeads}(\mathbf I_\theta\mathbf W_{qkv}+\mathbf b_{qkv}),
&\mathbf W_{qkv}&\in\mathbb R^{K\times3K},\\
\widetilde{\mathbf Q},\widetilde{\mathbf K}_a
&=\operatorname{RoPE}_{1D}(\mathbf Q,\mathbf K_a),\\
\mathbf A_{\rm attn}
&=\operatorname{softmax}_{j}\!\left(
\widetilde{\mathbf Q}\widetilde{\mathbf K}_a^{\mathsf T}/\sqrt C\right),\\
\mathbf F
&=\operatorname{ReLU}\!\left[
\operatorname{RMSNorm}\!\left(
\operatorname{MergeHeads}(\mathbf A_{\rm attn}\mathbf V_a)\mathbf W_o+\mathbf b_o
\right)\right]
\in\mathbb R^{B\times N\times K},\\
\dot{\boldsymbol\Theta}^{(\ell,r)}
&=\boldsymbol\Omega^{(\ell)}+\mathbf S_\theta\odot\mathbf F,\\
\Delta_\ell&=0.25\operatorname{sigmoid}(d_\ell),\\
\boldsymbol\Theta^{(\ell,r+1)}
&=\operatorname{wrap}_{[-\pi,\pi]}\!\left(
\boldsymbol\Theta^{(\ell,r)}+\Delta_\ell\dot{\boldsymbol\Theta}^{(\ell,r)}\right).
\end{aligned}
$$

同层 $T=3$ 步共享参数且 $\boldsymbol\Omega^{(\ell)}$ 不变；$\mathbf W_o\in\mathbb R^{K\times K}$。
若 $\ell<L-1$，在层末更新一次 frequency：

$$
\begin{aligned}
\mathbf E_\theta^{(\ell)}
&=\operatorname{GroupedLinear}_{2\to1}
([\sin\boldsymbol\Theta^{(\ell,T)},\cos\boldsymbol\Theta^{(\ell,T)}])
\in\mathbb R^{B\times N\times K},\\
\Delta\boldsymbol\Omega^{(\ell)}
&=\operatorname{Linear}_{K\to K}^{(\ell)}\!\left(
\operatorname{ReLU}\!\left[
\operatorname{Linear}_{2K\to K}^{(\ell)}\!\left(
\operatorname{RMSNorm}([\mathbf E_\theta^{(\ell)};\boldsymbol\Omega^{(\ell)}])
\right)\right]\right),\\
\alpha_\ell&=0.25\operatorname{sigmoid}(a_\ell),\\
\boldsymbol\Omega^{(\ell+1)}
&=\boldsymbol\Omega^{(\ell)}+\alpha_\ell\Delta\boldsymbol\Omega^{(\ell)},\\
\boldsymbol\Theta^{(\ell+1,0)}&=\boldsymbol\Theta^{(\ell,T)}.
\end{aligned}
$$

一次 forward：$LT=18$ 次 coupling，$L-1=5$ 次 frequency transition。

### 1.4 phase → clean latent → Flow Matching loss

$$
\begin{aligned}
\mathbf H_\theta^*
&=[\sin\boldsymbol\Theta^*;\cos\boldsymbol\Theta^*]_{[:,P:P+M,:]}
\in\mathbb R^{B\times M\times2K},\\
\widehat{\mathbf X}_0
&=\operatorname{Linear}_{2K\to D}
(\operatorname{RMSNorm}(\mathbf H_\theta^*))
\in\mathbb R^{B\times M\times D},\\
\widehat{\mathbf V}_t
&=\frac{\widehat{\mathbf X}_0-\mathbf Z_t}{\max(1-t,0.05)},
&\mathbf V_t^*&=\frac{\mathbf X_0-\mathbf Z_t}{\max(1-t,0.05)},\\
\mathcal L_{\rm FM}
&=\frac{\sum_{b,i}q_{bi}
\|\widehat{\mathbf V}_{t,bi:}-\mathbf V^{\rm tar}_{t,bi:}\|_2^2/D}
{\sum_{b,i}q_{bi}},
&q_{bi}&=a_{bi}(1-c_{bi}),\\
\mathbf V_t^{\rm tar}
&=\mathbf V_t^*+
m\left(1-\frac1w\right)(\mathbf V_{\rm sc}-\mathbf V_{\rm no\text{-}sc}).
\end{aligned}
$$

$\mathbf a\in\{0,1\}^{B\times M}$ 为有效 token mask；$w$ 为 self-conditioning CFG
scale；$m=0$ 时 $\mathbf V_t^{\rm tar}=\mathbf V_t^*$。
训练还以 $0.2$ 概率混入 shared-decoder CE，以 $0.1$ 概率删除 source 条件。

---

## 2. ODE 取样顺序

当前：$0=t_0<t_1<\cdots<t_{64}=1$，CFG scale $g=2$。

$$
\mathbf Z_{t_0}=\mathcal R_{\mathbf c}(2\boldsymbol\varepsilon,\mathbf X_{\rm src}),
\qquad
\widehat{\mathbf X}_0^{(-1)}
=\mathcal R_{\mathbf c}(\mathbf0,\mathbf X_{\rm src}).
$$

对 $k=0,\ldots,63$：

$$
\begin{aligned}
\widehat{\mathbf X}_{0,c}^{(k)}
&=f_\phi(\mathbf Z_{t_k},t_k,\widehat{\mathbf X}_0^{(k-1)};
\mathbf X_{\rm src}),\\
\widehat{\mathbf V}_{c}^{(k)}
&=\frac{\widehat{\mathbf X}_{0,c}^{(k)}-\mathbf Z_{t_k}}{\max(1-t_k,0.05)},\\
\widehat{\mathbf X}_{0,u}^{(k)},\widehat{\mathbf V}_{u}^{(k)}
&=\operatorname{UncondForward}(\mathbf Z_{t_k},t_k),\\
\widehat{\mathbf V}_{\rm cfg}^{(k)}
&=(1-\mathbf c)\odot\!\left[
\widehat{\mathbf V}_{u}^{(k)}
+g(\widehat{\mathbf V}_{c}^{(k)}-\widehat{\mathbf V}_{u}^{(k)})\right],\\
\widehat{\mathbf X}_0^{(k)}
&=\mathcal R_{\mathbf c}\!\left(
\widehat{\mathbf X}_{0,u}^{(k)}
+g(\widehat{\mathbf X}_{0,c}^{(k)}-\widehat{\mathbf X}_{0,u}^{(k)}),
\mathbf X_{\rm src}\right),\\
\mathbf Z_{t_{k+1}}
&=\mathcal R_{\mathbf c}\!\left(
\mathbf Z_{t_k}+(t_{k+1}-t_k)\widehat{\mathbf V}_{\rm cfg}^{(k)},
\mathbf X_{\rm src}\right).
\end{aligned}
$$

最后切换到 decode mode：

$$
\begin{aligned}
\mathbf H_{\rm dec}
&=\operatorname{WONN}_\phi([\mathbf Z_1;\mathbf0],t=1,\mathrm{decode})
\in\mathbb R^{B\times M\times2K},\\
\mathbf L_{\rm tok}
&=\operatorname{GELU}(\mathbf H_{\rm dec}\mathbf W_p+\mathbf b_p)
\mathbf W_u+\mathbf b_u
\in\mathbb R^{B\times M\times|\mathcal V|},\\
\widehat S_{bi}&=\arg\max_v L_{{\rm tok},biv}.
\end{aligned}
$$

$$
\boxed{
\mathbf S\xrightarrow{E_{\rm T5}}\mathbf X_0
\xrightarrow{\rm noise}\mathbf Z_t
\xrightarrow{\rm WONN}\widehat{\mathbf X}_0
\xrightarrow{\rm FM}\widehat{\mathbf V}_t
\xrightarrow{\rm Euler}\mathbf Z_1
\xrightarrow{\rm decoder}\widehat{\mathbf S}}
$$

口径：上文对应已训练的 Phase 5 正式 50K 配置 $T=3$；当前 main 通用 WMT YAML
默认 $T=2$，其余公式不变。

实现：src/modules/wonn_model.py；src/modules/wonn_layers.py；src/train_step.py；
src/utils/sampling_utils.py；src/utils/generation_utils.py。
