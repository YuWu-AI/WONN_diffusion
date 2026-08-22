#!/usr/bin/env python
"""Render a self-contained Chinese report for the Phase 5 mechanism study."""

import argparse
import json
from pathlib import Path


def render_report(summary: dict, output: Path) -> None:
    if summary.get("status") != "complete":
        raise ValueError("mechanism summary is not complete")
    tables = summary.get("tables", {})
    for name in ("training", "evaluation", "diagnostics"):
        if not isinstance(tables.get(name), list) or not tables[name]:
            raise ValueError(f"mechanism summary has no {name} rows")

    payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    html = REPORT_TEMPLATE.replace("__MECHANISM_PAYLOAD__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)


REPORT_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 5 WONN 高噪语义机制实验</title>
<style>
:root{color-scheme:light;--ink:#17211b;--muted:#5d675f;--line:#d8ddd9;--paper:#fff;--wash:#f5f7f5;--green:#16784a;--red:#b33b32;--blue:#2466a8;--amber:#a56a0b}
*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}
header{border-bottom:1px solid var(--line);background:#f0f4f1;padding:32px max(24px,calc((100vw - 1180px)/2)) 26px}header h1{font-size:30px;line-height:1.25;margin:0 0 10px}header p{max-width:900px;margin:0;color:var(--muted)}
main{max-width:1180px;margin:auto;padding:24px}.band{padding:24px 0;border-bottom:1px solid var(--line)}h2{font-size:21px;margin:0 0 14px}h3{font-size:16px;margin:18px 0 8px}.lede{max-width:960px;color:#34413a}.experiment-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.experiment{border-left:4px solid var(--line);padding:2px 0 2px 14px}.experiment:nth-child(1){border-color:var(--blue)}.experiment:nth-child(2){border-color:var(--green)}.experiment:nth-child(3){border-color:var(--red)}.experiment strong{display:block;font-size:16px}.experiment p{margin:4px 0;color:var(--muted)}
.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:14px 0}.segmented{display:inline-flex;border:1px solid #aeb7b0;background:#fff}.segmented button{border:0;border-right:1px solid #aeb7b0;background:#fff;color:var(--ink);padding:8px 13px;cursor:pointer}.segmented button:last-child{border-right:0}.segmented button[aria-selected=true]{background:var(--ink);color:#fff}.models{display:flex;flex-wrap:wrap;gap:12px}.models label{display:flex;align-items:center;gap:6px}.select-wrap{display:flex;align-items:center;gap:8px}.select-wrap select{border:1px solid #aeb7b0;background:#fff;padding:7px 30px 7px 9px;color:var(--ink)}
.panel[hidden]{display:none}.chart{width:100%;min-height:390px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--wash)}svg{display:block;width:100%;height:auto}.axis{stroke:#7c877f;stroke-width:1}.grid{stroke:#dfe4e0;stroke-width:1}.tick{fill:#5d675f;font-size:12px}.legend{fill:#253029;font-size:12px}.series{fill:none;stroke-width:3}.point{stroke:#fff;stroke-width:1.5}
.table-wrap{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;min-width:760px}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:12px;color:var(--muted);background:var(--wash);position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.headline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:16px 0}.headline div{background:#fff;padding:16px}.headline span{display:block;color:var(--muted);font-size:12px}.headline strong{font-size:22px;font-variant-numeric:tabular-nums}
.glossary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 28px}.term{padding:12px 0;border-bottom:1px solid var(--line)}.term dt{font-weight:700}.term dd{margin:3px 0;color:var(--muted)}.note{border-left:4px solid var(--amber);padding:8px 14px;background:#fff9ed;color:#5e461c}.empty{padding:80px 20px;text-align:center;color:var(--muted)}
@media(max-width:760px){header{padding:24px 18px}header h1{font-size:25px}main{padding:16px}.experiment-grid,.headline,.glossary{grid-template-columns:1fr}.chart{min-height:300px}.segmented{width:100%}.segmented button{flex:1;padding:8px 5px}}
</style>
</head>
<body>
<header><h1>Phase 5 WONN 高噪语义机制实验</h1><p>回答一个具体问题：WONN 是否已经会处理中低噪 latent，却没有在接近纯噪声的起点依赖德语 source 建立正确的英语语义表达。</p></header>
<main>
<section class="band"><h2>实验内容</h2><p class="lede">三组实验共享同一批 50,000 条短句训练数据、2,000 条 held-out 数据、L4T3/O192/H6 小模型、batch 12 和 50,000 optimizer steps。辅助目标从 5k 开始、用 5k 线性 warmup，仅作用于 <strong>t ≤ 0.25</strong> 的高噪 denoiser 样本；本项目定义 <code>z=t*x0+(1-t)*noise</code>，所以 t 越小，输入越接近纯噪声。</p>
<div class="experiment-grid"><div class="experiment"><strong>S-Base</strong><p>原始 Flow Matching + shared decoder CE，不增加 denoiser 语义监督。</p></div><div class="experiment"><strong>S-Token</strong><p>用冻结 EMA teacher 将高噪 <code>x_pred</code> 解码为 token，并增加权重 0.02 的 target CE。</p></div><div class="experiment"><strong>S-Token-Contrast</strong><p>在 S-Token 上增加权重 0.1 的 source 错配 margin，要求正确 source 比近长度错误 source 更匹配 target。</p></div></div></section>

<section class="band"><h2>结果总览</h2><div id="headline" class="headline"></div><div id="terminal-table" class="table-wrap"></div><p class="note">单个随机种子只能判断机制方向是否值得继续，不能估计训练随机性。自由采样指标应与高噪 x_pred、source sensitivity 和训练轨迹联合解释。</p></section>

<section class="band"><h2>交互分析</h2><div class="toolbar"><div class="segmented" role="tablist" aria-label="分析视图"><button role="tab" aria-selected="true" data-view="training">训练</button><button role="tab" aria-selected="false" data-view="evaluation">自由采样</button><button role="tab" aria-selected="false" data-view="diagnostics">机制诊断</button></div><div class="models" id="model-filter"></div><label class="select-wrap">指标<select id="metric-select"></select></label></div>
<div id="training" class="panel"><div id="training-chart" class="chart"></div><div id="training-table" class="table-wrap"></div></div>
<div id="evaluation" class="panel" hidden><div id="evaluation-chart" class="chart"></div><div id="evaluation-table" class="table-wrap"></div></div>
<div id="diagnostics" class="panel" hidden><div class="toolbar"><label class="select-wrap">数据集<select id="split-select"><option value="heldout">held-out</option><option value="train">train</option></select></label><label class="select-wrap">探针<select id="probe-select"><option value="time_bin">时间分桶</option><option value="conditioning">source 条件</option><option value="clean_x0">干净 latent 解码</option><option value="training_noised_x0">训练噪声 latent 解码</option></select></label></div><div id="diagnostics-chart" class="chart"></div><div id="diagnostics-table" class="table-wrap"></div></div>
</section>

<section class="band"><h2>Metric 说明</h2><dl class="glossary">
<div class="term"><dt>loss / L2 loss</dt><dd>总训练目标与 Flow Matching velocity 均方误差。L2 下降只说明局部向量场更准确，不必然等价于完整采样翻译更好。</dd></div>
<div class="term"><dt>CE loss</dt><dd>shared decoder 在 decoder branch 上恢复 target token 的交叉熵，越低越好。</dd></div>
<div class="term"><dt>token loss</dt><dd>EMA teacher 从高噪 denoiser 的 <code>x_pred</code> 恢复 target token 的交叉熵，直接测量 latent 是否已携带可解码语义，越低越好。</dd></div>
<div class="term"><dt>source contrastive loss</dt><dd>正确 source 与错误 source 的 margin 违例。趋近 0 表示正确 source 已比错配 source 更支持目标句，但必须结合 source 条件差值判断是否退化。</dd></div>
<div class="term"><dt>BLEU</dt><dd>基于 n-gram 精确率的语料级翻译指标，0–100，越高越好；对措辞变化较敏感。</dd></div>
<div class="term"><dt>chrF++</dt><dd>字符 n-gram 与词 n-gram 的 F-score，0–100，越高越好；对形态变化和低分模型通常比 BLEU 更平滑。</dd></div>
<div class="term"><dt>TER</dt><dd>将生成文本编辑为参考文本所需的归一化编辑次数，越低越好。</dd></div>
<div class="term"><dt>correct − zero chrF++</dt><dd>保留 source 相比清零 source 的 chrF++ 增益。正值越大，说明自由采样更依赖 source；接近 0 表示条件可能被忽略。</dd></div>
<div class="term"><dt>velocity MSE / cosine</dt><dd>固定 t 下预测速度与理想速度的误差及方向一致性；MSE 越低、cosine 越接近 1 越好。</dd></div>
<div class="term"><dt>x_pred MSE</dt><dd>固定 t 下预测 clean latent 与真实 x0 的均方误差，越低越好；t=0 是最关键的纯噪起点探针。</dd></div>
</dl></section>
</main>
<script>const report=__MECHANISM_PAYLOAD__;</script>
<script>
const colors={"S-Base":"#2466a8","S-Token":"#16784a","S-Token-Contrast":"#b33b32"};
const labels={loss:"总 loss",l2_loss:"L2 loss",ce_loss:"CE loss",aux_loss:"辅助 loss",token_loss:"token loss",source_contrastive_loss:"source contrastive",bleu:"BLEU",chrf2:"chrF++",ter:"TER",empty_rate_pct:"空输出率",velocity_mse:"velocity MSE",velocity_cosine:"velocity cosine",x_pred_mse:"x_pred MSE",correct_minus_zero_chrf2:"correct-zero chrF++",ce_loss_probe:"decoder CE"};
let view="training";const selected=new Set(Object.keys(report.runs));
const metricOptions={training:["loss","l2_loss","ce_loss","aux_loss","token_loss","source_contrastive_loss"],evaluation:["bleu","chrf2","ter","empty_rate_pct"],diagnostics:[]};
const finite=v=>typeof v==="number"&&Number.isFinite(v);const fmt=v=>finite(v)?(Math.abs(v)>=1000?v.toLocaleString("zh-CN",{maximumFractionDigits:1}):v.toFixed(Math.abs(v)<1?4:2)):"—";
function table(target,rows,fields){const el=document.getElementById(target);if(!rows.length){el.innerHTML='<div class="empty">没有可显示的数据</div>';return}el.innerHTML='<table><thead><tr>'+fields.map(f=>`<th>${f[1]}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+fields.map(f=>`<td class="${finite(r[f[0]])?'num':''}">${finite(r[f[0]])?fmt(r[f[0]]):(r[f[0]]??'—')}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
function chart(target,rows,xKey,yKey,xLabel){const el=document.getElementById(target);const groups={};rows.filter(r=>selected.has(r.model)&&finite(r[xKey])&&finite(r[yKey])).forEach(r=>(groups[r.model]??=[]).push(r));const points=Object.values(groups).flat();if(!points.length){el.innerHTML='<div class="empty">当前筛选下没有该指标</div>';return}const W=1080,H=390,m={l:72,r:25,t:38,b:56},pw=W-m.l-m.r,ph=H-m.t-m.b;let xs=points.map(r=>r[xKey]),ys=points.map(r=>r[yKey]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmin===xmax)xmax=xmin+1;if(ymin===ymax){ymin-=.5;ymax+=.5}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;const sx=x=>m.l+(x-xmin)*pw/(xmax-xmin),sy=y=>m.t+(ymax-y)*ph/(ymax-ymin);let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${labels[yKey]??yKey} 图表">`;for(let i=0;i<6;i++){let q=i/5,x=xmin+q*(xmax-xmin),y=ymin+q*(ymax-ymin);svg+=`<line class="grid" x1="${sx(x)}" y1="${m.t}" x2="${sx(x)}" y2="${m.t+ph}"/><text class="tick" x="${sx(x)}" y="${H-24}" text-anchor="middle">${fmt(x)}</text><line class="grid" x1="${m.l}" y1="${sy(y)}" x2="${m.l+pw}" y2="${sy(y)}"/><text class="tick" x="${m.l-10}" y="${sy(y)+4}" text-anchor="end">${fmt(y)}</text>`}svg+=`<line class="axis" x1="${m.l}" y1="${m.t+ph}" x2="${m.l+pw}" y2="${m.t+ph}"/><line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t+ph}"/><text class="legend" x="${m.l+pw/2}" y="${H-5}" text-anchor="middle">${xLabel}</text>`;Object.entries(groups).forEach(([model,items],i)=>{items.sort((a,b)=>a[xKey]-b[xKey]);const color=colors[model]??"#333",path=items.map((r,j)=>`${j?'L':'M'} ${sx(r[xKey])} ${sy(r[yKey])}`).join(' ');svg+=`<path class="series" d="${path}" stroke="${color}"/>`+items.map(r=>`<circle class="point" cx="${sx(r[xKey])}" cy="${sy(r[yKey])}" r="4" fill="${color}"/>`).join('')+`<text class="legend" x="${m.l+12}" y="${18+i*17}" fill="${color}">${model}</text>`});el.innerHTML=svg+'</svg>'}
function updateMetrics(){const select=document.getElementById('metric-select');let values=metricOptions[view];if(view==='diagnostics'){const probe=document.getElementById('probe-select').value;values=probe==='time_bin'?["velocity_mse","velocity_cosine","x_pred_mse"]:probe==='conditioning'?["correct_minus_zero_chrf2"]:["ce_loss"]}const current=select.value;select.innerHTML=values.map(v=>`<option value="${v}">${labels[v]??v}</option>`).join('');if(values.includes(current))select.value=current}
function render(){updateMetrics();const metric=document.getElementById('metric-select').value;if(view==='training'){const rows=report.tables.training;chart('training-chart',rows,'step',metric,'optimizer step');table('training-table',rows.filter(r=>selected.has(r.model)),[["model","模型"],["step","step"],[metric,labels[metric]??metric],["samples_seen","已见样本"],["elapsed_training_seconds","训练秒数"]])}else if(view==='evaluation'){const rows=report.tables.evaluation;chart('evaluation-chart',rows,'step',metric,'checkpoint step');table('evaluation-table',rows.filter(r=>selected.has(r.model)),[["model","模型"],["step","step"],["num_samples","样本数"],["bleu","BLEU"],["chrf2","chrF++"],["ter","TER"],["empty_rate_pct","空输出率 %"]])}else{const split=document.getElementById('split-select').value,probe=document.getElementById('probe-select').value,rows=report.tables.diagnostics.filter(r=>r.split===split&&r.probe===probe);const xKey=probe==='time_bin'?'t':'step';chart('diagnostics-chart',rows,xKey,metric,probe==='time_bin'?'flow time t':'checkpoint step');const preferred=[["model","模型"],["step","step"],["split","数据集"],["probe","探针"],[xKey,xKey],[metric,labels[metric]??metric]];table('diagnostics-table',rows.filter(r=>selected.has(r.model)),preferred.filter((f,i,a)=>a.findIndex(x=>x[0]===f[0])===i))}}
function overview(){const terminal=report.tables.evaluation.filter(r=>r.step===50000).sort((a,b)=>b.bleu-a.bleu);const best=terminal[0]??{};const base=terminal.find(r=>r.model==='S-Base')??{};document.getElementById('headline').innerHTML=`<div><span>50k 最佳 BLEU</span><strong>${fmt(best.bleu)}</strong>${best.model??'—'}</div><div><span>50k 最佳 chrF++</span><strong>${fmt(Math.max(...terminal.map(r=>r.chrf2)))}</strong></div><div><span>最佳模型相对 S-Base BLEU</span><strong>${fmt((best.bleu??0)-(base.bleu??0))}</strong></div>`;table('terminal-table',terminal,[["model","模型"],["num_samples","样本数"],["bleu","BLEU"],["chrf2","chrF++"],["ter","TER"],["empty_rate_pct","空输出率 %"],["unique_rate_pct","唯一输出率 %"],["length_ratio","长度比"]])}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==view);render()}));
const filter=document.getElementById('model-filter');Object.keys(report.runs).forEach(model=>{const label=document.createElement('label');label.innerHTML=`<input type="checkbox" checked value="${model}"><span style="color:${colors[model]}">${model}</span>`;label.querySelector('input').addEventListener('change',e=>{e.target.checked?selected.add(model):selected.delete(model);render()});filter.appendChild(label)});
['metric-select','split-select','probe-select'].forEach(id=>document.getElementById(id).addEventListener('change',render));overview();render();
</script>
</body></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=Path(
            "outputs/phase5/redesign_v2/mechanism50k/analysis/mechanism_summary.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/phase5/redesign_v2/mechanism50k/analysis/report.html"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_report(summary, args.output)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
