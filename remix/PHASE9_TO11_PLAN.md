# Work packages 1–3: frozen development acceptance

Current state: **1 open** (DFZ forward peak fidelity); **2 passed** (semantic
knob development/runtime gates); **3 passed** (order plus full-chain frozen
development gates, with disclosed audio/input-level limitations). This is not
completion of all three while the DFZ gate remains open.

The user authorized completing DFZ forward, semantic knob recovery, and order /
full-chain development work continuously, with autonomous model choice. This
does not authorize UI integration, audio-device enumeration/playback/recording,
source-audio mutation, new locked-final evaluation, or commercial release.
Existing accepted artifacts remain immutable until a replacement passes.

## 自我检查与实验决策（2026-08-31，优先于下方历史探索记录）

问题不是“还能试什么”，而是“下一次投入能否回答一个明确问题，并推动原有音质/部署门槛”。
审查结论：**支持报告已观察到的失败；不支持自动追加长训练，也不支持宣称 DFZ 已改善至可用。**
旋钮恢复和顺序/全链路的既有开发验收保持冻结；DFZ forward 仍是未完成项。

### 已发现的问题及处置

| 问题 | 核实证据 | 对决策的影响 |
|---|---|---|
| 进度参照漂移 | 保留的非线性读出 CPU54 峰值 P95 0.027221；宽网络 0.062039，条件阈值网络 0.060195 | “本轮最佳”不等于项目改善；以后同时列保留基线和同架构起点 |
| 没有及时收束无效路线 | L1 8,000 步 + 固定文件能量 6,000 步均选回 step0，权重 SHA 与起点完全相同，合计约36分钟 | 两种损失改法没有产出改进；不自动再换损失或续训 |
| 小样本诊断被赋予过大决策权 | 坐标实验只有一条 fit 录音、一个种子，两个方案连样本内门槛都未通过 | 只能提示优化现象，不能支持新的20,000步阶段；已撤回未运行续训脚手架 |
| 技术检查与音质证据混用风险 | 合成未训练 ONNX 可通过，而两份已训练有限历史模型实际 CPU/ONNX 均超过2e-6 | 单元测试、合成预检不算音质/部署准入，也不能抵消保真退步 |
| 选择集被反复使用 | 同一234 fit /54 calibration、多轮选优；旧公开预训练模型可能已见过全部官方 train | 54条仅用于开发决策，不是独立泛化证据；不把文件数当成独立演奏者数 |
| 原因尚未被实验隔离 | 最新从零模型同时改变拓扑、损失尺度和学习率；输出梯度诊断不是参数梯度或因果证明 | 不声称已找出失败根因，也不凭论文架构名判断路线更好 |

指标口径保持原样：48kHz、每条144,000帧；保留完整因果历史，评分只排除前1,024帧。
`absolute_peak_error_p95` 是各文件 `abs(max(abs(pred))-max(abs(wet)))` 的95分位，
**不是逐采样最大残差**。全局 ESR 是汇总误差能量/汇总 Wet 能量，不是文件 ESR 的平均。
旧字段 `by_attack` 对 DFZ 实际表示第一个旋钮 fuzz blend；九个旋钮角各6条 calibration。
CPU 最终结果用于正式同条件比较，MPS 训练期结果只能标作暂定值。

### 同条件复测记录

2026-08-31 重新加载保留的 `dfz-nonlinear-readout-phase9/candidate.pt`，
在最新明确列出的同一54条 calibration 上使用当前评分实现复测：CPU float32、2线程、batch9、
4096帧流式、有完整前缀、排除首1024帧评分；耗时40.85秒（不是实时性能基准）。
结果：global ESR **0.00956541440279943**、峰值 P95 **0.0272210031747818**、
最差 fuzz-blend P95 **0.03690158426761624**，依然失败。
后两项与旧报告完全相同；global ESR 相差约5.45e-11，为不同批次汇总的数值差异。

清单核验：234+54条路径全部唯一，全部属于 train；take-modulo-5 分区一致；54条的九角各6条。
288条原始文件 SHA 在复测前后均匹配清单；解码的54条均为有限值、144000帧。
本次没有读取 official eval，没有训练、保存音频或改动 checkpoint。
这确认当前开发基线可复现，不证明旧预训练数据独立性，也不是完整导出准入。

- Checkpoint SHA256：`89640c9ee860544a8f26f408a883311657f47de60b7076d5315d00599890037a`
- 明确路径清单：`runs/dfz-centered-core-phase11/metrics.json` 的 `audio_provenance.calibration`；报告 SHA256：`8a6f337e40e333837ec1be93b95556d3e73a727a703e88927c16ec1e1ef43251`
- 当前评分代码 `train_asrnn_phase7.py` SHA256：`a9b71deabf70a103e903d16d40a2039488b8723484e220e0a8c9e883611b4551`
- 顶层加载/读出实现 `stable_effect.py` / `stable_nonlinear_readout.py` SHA256：`2a9ee24e4abb8c2b24c4b4dd31de948c5b118c0181ad60dd92ba3e0fd35e9261` / `ab57982b787140d48a5e1e4dece6d6d7621c6553555f30fb22dfd963cbc723bf`

核心复测代码（仓库根目录、`.venv310/bin/python`；仅按显式清单读取）：

```python
import json
from pathlib import Path
import torch
from remix.stable_effect import load_stable_effect
from remix.train_asrnn_phase7 import _evaluate, _rows
from remix.precheck_multirate_fuzz import sha256
provenance = json.loads(Path('remix/runs/dfz-centered-core-phase11/metrics.json').read_text())['audio_provenance']
records = provenance['calibration']
assert len(records) == 54
assert all('train' in Path(r['path']).parts and sha256(r['path']) == r['sha256'] for r in records)
torch.set_num_threads(2)
model, _ = load_stable_effect(Path('remix/runs/dfz-nonlinear-readout-phase9/candidate.pt'))
print(_evaluate(model, _rows([Path(r['path']) for r in records], 'dfz'), torch.device('cpu'), 9))
```

### 后续实验的启动与停止规则

1. **启动前留一条决策记录**：待证伪假设、对应失败文件/分组、对照、唯一改动（或明确说明混杂）、主要指标、资源上限、成功及停止条件。缺项则不启动长训练。
2. **先有可区分解释的证据**：使用预先固定的 fit-only 多录音/多旋钮对照；不能按结果挑一条有利样本，也不能用校准集选出的改进当作因果证明。
3. **比较两条轴，分别判定**：音质对照保留的非线性读出；实现/速度对照已测运行时。保真退步不能被更小参数量或合成 parity 抵消。若专做部署研究，先声明其问题与预算，不宣传为音质改善。
4. **明确止损**：到预定检查点未满足该实验事先规定的继续条件即停；阶段结束失败则归档，不自动追加步数、种子或另一结构。相同权重无需重复导出审计。只有能改变下一项决策的检查才运行。
5. **准入不变**：峰值 P95≤0.02、最差 fuzz-blend 分组≤0.03，以及原有完整质量/静音/流式/float32 parity/速度门槛全部通过，才允许下一层验证。不以新增宽松“研究阈值”替代原门槛。
6. **数据与安全边界不变**：不因本次欠拟合就下载更多无配对音频；不打开 official eval/locked-final 调参，不接 UI，不访问实体音频设备，不改音源或补增益/限幅。

唯一在途的 `dfz-cascaded-core-phase11` 已完成固定20,000步及最终 CPU54（共2606.60秒）。
选中20,000步：ESR **0.02143497085236875**、峰值 P95 **0.1055259972810745**、
最差 fuzz-blend P95 **0.11223839521408079**。与上述旧基线同清单、同评分实现、同 CPU 设置，
两项峰值门槛均失败，且保真落后于保留基线。**此轮探索结束，没有运行中或排队的续训。**
不替换模型，不打开更多评估数据；真实权重的额外 ONNX 审计不再运行，因为即使通过也不能改变本轮保真拒绝结论。
这不等于证明该结构永远无效，只表明本轮预算下没有形成继续长训练的充分依据。
checkpoint SHA256：`e5a65dd9ec6f07724a15582110595088dc669ae9ca2c8104d0943bf9d94bdb0a`；
`runs/dfz-cascaded-core-phase11/metrics.json` SHA256：`6bb432e945a6cd81dece056d2698cc1be65785b8cded5b0e0d6b4348644dfc66`。
源代码、原音频、清单和权重哈希均完成核验；该阶段只留一个候选及两份记录，总358,405字节。
240项回归测试通过，仅代表实现回归，不作为音质证据。撤回的未运行续训脚手架可从任务修改历史恢复，无音频或权重被删除。
自查证据直接收敛到本计划和既有运行记录，不新增仪表盘、音频副本或特征缓存。

## 1. DFZ forward

### 2026-08-31 用户续作：冻结读出的奇偶特征对照

这是自查后的一个新决策，不是续跑被拒绝的两级网络。假设：保留模型的
`NonlinearReadout` 对隐藏状态使用奇函数（线性+tanh），限制了对状态相关不对称
残差的表达。隐藏状态本身不一定对 Dry 是奇函数，因此这不是电路不对称的证明。
此前的平方电荷/原输出特征并没有检验隐藏状态的偶次特征。

- 核心、输入、旧输出和旋钮语义完全冻结，保留 checkpoint SHA `89640c9e…`。
- 仅使用原234 fit；预先按 take%4==1 留54条内部检验，其余180条拟合，每角20/6条。
  不读原cal54或official eval音频。此内部拆分也不是新设备/演奏者独立最终集。
- 固定四臂：不变；80维现有奇特征；160维奇+三次特征（匹配容量对照）；160维奇+平方特征。
  特征为最后64维状态及原16维tanh投影；没有新增递归层、录音增益或事后限幅。
- 三臂使用同一固定ridge1e-3、相同样本、同一完整文件能量权重；无超参菜单或随机重跑。
  每条2048均匀点和Wet/旧输出正负极值各±128帧，均匀与事件权重各半。
  拟合用float64正规方程，系数折回float32；完整CPU流式回放54条内部检验，评分口径不变。
- 仅当平方臂相对不变和两个对照的峰值P95均改善≥10%、相对不变global ESR/最差分组不恶化，
  且配对文件均值峰值误差差的固定bootstrap95%上界<0，才允许在全部234 fit上重拟合一次、
  打开原cal54作原门槛验收。这是投入判据，不替代原0.02/0.03音质门槛。
- 预算：一次CPU提取/拟合/回放，15分钟墙钟上限，2线程；不建音频/隐藏状态磁盘缓存。
  无显著信号即关闭这条假设，不自动加维、改ridge或开长训练。只留脚本/测试与紧凑JSON。

该对照已完成且**关闭偶次特征路线**。内部54条不变/奇/等维三次/平方臂峰值P95分别为
0.026378 /0.068949 /0.162478 /0.129548；平方臂global ESR0.011097，差于不变0.010152。
所有预定继续条件均失败，没有保存拟合权重，没有打开原cal54。
`runs/dfz-readout-parity-phase11/metrics.json` 保存逐文件数据、源/音频SHA和决策。

### 独立后续问题：全段约束与特征扩展分开检验

新证据是相同80维奇特征的ESR下降到0.007505而峰值升高：不能用均方拟合改善推断峰值改善。
下一项诊断只保留原80维特征，比较**相同全段正规方程**的无约束ridge与带整段峰值约束的ridge，
不增加奇偶特征、不重启神经网络训练。180/54内部划分、原核心、float32输出和ridge1e-3仍冻结。

- 在180条完整拟合录音上累积带整段能量尺度的波形正规方程。
  每条训练录音约束全段预测绝对幅度≤Wet峰值+0.015，并在Wet峰值的同一采样点保持对应符号、幅度≥Wet峰值−0.015。
  这些只约束**训练系数**，推理没有Wet输入、增益估计或限幅器。
- 用线性可行性检查和凸二次优化，逐轮加入全段最严重违反的采样点，至多8轮；每角20条完整录音核验。
  任一角不可行、数值求解失败、8轮后仍有违反或总15分钟预算耗尽，则停止并保留明确原因，不放宽约束重跑。
  0.015是这一拟合问题的设计值，不替代原cal验收0.02门槛，不声称它是必要条件。
- 仅全部拟合约束核验通过才回放同一内部54条。继续到全234拟合的条件仍为：峰值P95比不变和同条件无约束臂各低≥10%、
  global ESR及最差fuzz-blend不比不变差，配对文件峰值误差均值差的固定bootstrap95%上界<0。
  不可行只能约束这个固定特征/训练条件问题，不能证明DFZ本身不可建模。
- 内存只保留当前角的完整冻结特征，约1GB，不存磁盘缓存；无模型通过则不建runtime schema或导出包。

首轮在六个角（120条完整拟合录音）通过后，于100,0角的HiGHS可行性检查得到Unknown状态；
已按规则停止，未读内部检验音频、未保存候选。Unknown不是不可行证据。
下一步仅做这个20文件角的数值复现，4分钟预算：保持模型、数据、0.015约束和二次目标不变，
以正的行缩放及总能找到可行起点的Phase-I松弛问题检查原约束；不丢弃特征或放宽限制。
同时检查原始尺度约束、对偶平衡及原始/对偶间隙；只有零松弛才表示原问题可行。
正松弛需要对偶支持，求解失败则继续标为数值问题，不能作为模型失败结论。
原始凸求解器线程未单独固定（Torch/BLAS各2线程）；本次数值复现显式固定HiGHS2线程，所有耗时仅为运行预算而非性能基准。

首次数值复现（53.13秒）将Unknown解析为正的最小标准化松弛0.003596059，
对偶平衡残差3.33e-12、原始/对偶间隙约2.18e-16，说明该固定特征的0.015训练约束不相容。
这尚不能推出原0.02独立最大值门槛不可达：同采样点锚定是更强的要求。
数值脚本随后增加辅助LP，只计算相同约束族的最小绝对容差下界，不采用辅助解、不改实际约束。
旧`numerical-reproducer.json`保留为脚本修订前的历史结果；新修订用独立`numerical-bound.json`绑定新代码SHA。

辅助LP得出的已核验下界为**0.01897896145544205**，低于原验收0.02。
因此不能把额外0.015训练设计的失败解释为原目标失败。这里显式修正的是额外实验设计，不是音质验收门槛；
这也纠正了此前“任何训练约束失败都停止”的过强规则。下界只是必要条件，尚不证明整段可行。
仅对同一100,0角做一次原目标核验：0.02减1e-6浮点余量、仍然8轮/4分钟、相同80特征和ridge。
两种容差分别传入锚定与完整扫描，新增测试防止只改一处或默许超过0.02。
不采用辅助LP给出的系数，不按结果剔除文件；只有原目标核验通过才允许恢复其余组合和内部54验证。
此前报告保存历史代码SHA，新核验以`original-gate-corner.json`绑定修订后的实现。

原目标的全段新增约束进一步给出下界0.020811748501164344（对偶平衡4.44e-12）。
它否定的是该固定80维读出下的**每文件、同采样点**要求，不是原P95总体要求，不能据此声称原模型目标不可能。
该角未保存系数，原校准集和内部检验均未被这组约束实验打开。

### 本阶段最后一项对照：总体尾部约束（不是再换网络）

纠正“每文件最大值”与“总体P95”混淆，仍固定180/54内部划分及80维读出。
使用CVaR epigraph约束最大ceil(5%*N)个**对齐峰值误差上界**的平均：总体≤0.02−1e-5，
各fuzz-blend组≤0.03−1e-5。它保守地蕴含原P95约束，但不是原指标本身；所有录音都参与，
不人工挑出或剔除难例。最小化同一完整波形ridge目标，只有训练参数被约束，推理仍是原结构的线性读出修正。

- 稀疏凸QP使用项目虚拟环境新增的`osqp==1.0.5`，未升级其他依赖；实现依据[官方问题定义](https://osqp.org/docs/solver/index.html)。
- 一次45分钟预算、至多10轮全段切平面、每轮QP≤60秒；求解残差、float32系数回放和实际CVaR都必须核验。
  预算在正式拟合前由小批量预检修正：9条暖态7.43秒，完整180条每轮约149秒，不能沿用未测量时的25分钟估计。
  预检MPS分配1.33GB；冻结输出MPS/CPU最大差7.75e-6，只是训练特征实现观察，明确不算通过2e-6部署门槛。
- 原音频按显式路径流式读取；冻结递归编码可在MPS计算，不缓存隐藏状态或音频到磁盘。CPU最终检验不使用MPS结果代替。
  RAM只保留小型正规方程和约束采样点，不保留约8GB的全段特征；MPS分配预算4GB。
- 无约束全段ridge作为同条件对照。仅训练完整约束通过才运行内部54 CPU回放；继续条件仍为峰值P95比不变/无约束各好10%、
  global ESR及最差分组不退步、配对均值差bootstrap95%上界<0。失败后不再追加本阶段变体。
- 内部验证通过才允许全234重拟合及原cal54的完整音质验收；这里不代表可以打开official eval或加载应用。

原坐标OSQP首轮通过独立残差核验（primal3.09e-10），全段扫描补入424点；第二轮达到100,000次迭代上限，
primal3.74e-5 / dual2.40e-6，未获准通过。没有把求解器未收敛解释为音频或模型不可行。
数值修复只做可逆Cholesky变量变换：模型、ridge、尾部阈值、全部文件和目标函数不变，
解恢复原坐标后仍检查相同残差。新增测试验证目标值及约束值在变换前后一致。
同一问题的重跑写`whitened-metrics.json`，历史`metrics.json`保留；若再次数值失败，仅保存紧凑稀疏QP，
避免重复提取全部音频，仍不保存原音频或完整隐藏状态缓存。

Cholesky首轮通过，第二轮仍先碰到100,000次迭代上限（primal1.57e-6），保存约1.06MB稀疏QP。
离线纯数值复现`solver-budget-replay.json`只提高迭代次数上限至2,000,000，**保持每次60秒墙钟预算**及全部阈值不变：
183,525次/55.21秒收敛，原坐标独立primal3.00e-9、dual7.83e-10。没有打开音频或checkpoint。
因此继续同一固定拟合，报告`verified-solver-metrics.json`，总45分钟/10轮保持不变；未获得模型或音质准入。
当前完整离线回归：`python -m unittest discover -s remix -t . -p 'test_*.py'`，259项/8.864秒全部通过。
新增反例测试拒绝未收敛状态、非有限解、原坐标primal或dual超标，即使求解器自报残差为零也不接受。
这是工程测试，不是DFZ音质通过；测试日志中的eval为临时合成fixture，不是官方音频。

`verified-solver-metrics.json`第三轮触及60秒墙钟上限，原坐标primal4.49e-6、dual1.26e-6，按规则结束。
最后一项90秒纯数值自查只读取该轮紧凑QP，Phase-I验证保存的切面约束，检查原尺度primal及对偶平衡/预算/间隙；
不输出拟合系数，不开启音频/模型。可行只证明这些切面不冲突，不代表全部时间点可行或原音质门槛可达；
不可行也仅适用于更强的同点锚定/CVaR约束。无论结果如何，本阶段不再追加拟合或模型变体。

### 本轮结案：未获得DFZ准入，后续不自动续训

- 最终拟合耗时571.972秒，两个完整180文件扫描通过执行但未通过约束；实际保守尾部上界0.260036/0.271338。
  第三轮QP用满60秒，原坐标primal4.4869e-6、dual1.2589e-6，未达到2e-7/1e-6要求。
- `final-feasibility.json`在0.535秒内给出零Phase-I松弛；保存的3616行/1264列问题原尺度最大违反1.4424e-9。
  因而保存切面**可行**，而不是模型不可行；仍未知所有时间点的可行性，也未获得满足原完整音质门槛的模型。
  该报告`numerically_verified=false`沿用QP最优解字段；可行性结论明确记录于`audit.verified_feasible=true`，不是模型准入。
- 按预定停止规则关闭这一阶段，未运行内部54候选验收、全234重拟合、原cal54/official eval及导出；没有新权重。
  本轮全部新实验目录约2.6MB，没有音频或全隐藏状态缓存；保留基线SHA再次匹配`89640c9e…`。
- 工作包2/3的既有开发验收文件再次核实通过，工作包1保持open，不能称为全部完成。
  当前没有运行中或排队的拟合/训练；不凭此次数值失败推荐加数据、再换网络或放宽音质门槛。
- 收尾完整回归260项/7.955秒全部通过，`git diff --check`通过；只是实现检查，不改变DFZ拒绝结论。
- 若以后重开此路线，第一关只能是在小型QP复现上验证更有效的求解/初始化，先冻结数值预算与原尺度残差要求，
  通过之后才能讨论恢复完整拟合；不能直接再开音频提取或长训练，也不能把可行性证据当作泛化改善。

2026-09-01按上述第一关重开一次纯数值A/B：冷启动同一3616行QP已知60秒失败；唯一改动是用已验证的
Phase-I可行点初始化OSQP，目标、约束、Cholesky坐标、1e-8求解容差、原坐标2e-7/1e-6门槛及60秒上限不变。
不读取音频/checkpoint，不保存解或权重。一次运行仍失败即关闭初始化假设；通过也只允许恢复完整拟合，不能算音质改善。

该A/B已完成并**关闭初始化假设**。Phase-I仍在0.594秒得到原尺度违反1.44e-9的可行点，但暖启动QP用满60秒后
原坐标primal5.93e-6、dual1.70e-6，比冷启动的4.49e-6/1.26e-6更差；没有合格解、音频读取或权重输出。
现有记录已覆盖稳定LSTM候选、全核/末层/跳层读出、FIR/短长TCN/NARX、charge/交互、事件损失、有限历史网络、
级联网络和固定读出凸约束；直接增益/归一化/限幅又违反既定音质约束。未发现尚未检验且有独立证据支持的低成本变体。
所以DFZ forward当前正式标为**研究阻塞**：工作包未完成，但没有依据继续随机换结构、损失或下载无配对吉他音频。
解除阻塞需要新的机制证据或真正设备/演奏者隔离的配对Dry/Wet数据；无配对纯吉他只能做稳定性测试，不能训练/验收该映射。

公开数据复核未发现Duality Fuzz的新增配对参数数据；ToneTwisT覆盖多种失真/模糊设备、MorphDrive覆盖27款过载，均不能替代目标设备映射。
厂商手册为两条fuzz电路的固定混合提供机制依据，但这一事实此前已记录。启动前先做一次无训练诊断：仅在原fit内部留出的18条
Fuzz Blend=50录音上比较现模型与两个端点条件输出的50/50固定混合；全54条端点保持不变。CPU、10分钟上限，不读原cal/eval。
只有中点峰值P95改善至少10%、中点/全局ESR不退步、配对bootstrap上界<0且全54峰值不退步，才允许双专家训练。

诊断已完成且五项条件全部失败，因此**不启动双专家训练**。中点18条的峰值P95从0.019252恶化到0.070349，
global ESR从0.009349恶化到0.015060；全54条峰值P95从0.026378恶化到0.052781，global ESR从0.010152
恶化到0.011900，配对bootstrap也未通过。厂商所说的电路混合不能被假定为最终输出的线性混合；该具体实现被数据否决。
报告`dfz-circuit-mixture-phase11/metrics.json`只含逐文件指标/哈希，无权重、音频或隐藏缓存。第一次执行完成回放后暴露
子集汇总要求三组Blend的实现错误；修正为复用同源同哈希CPU54基线，并只重新计算18条中点的两个端点，避免再回放端点。
公开数据搜索不下载：当前ToneTwisT清单有多种失真/fuzz但无目标Duality Fuzz，MorphDrive是27款overdrive而非该fuzz设备。
因此现阶段没有可直接补入的同设备配对集；下载这些数据只能支持通用预训练研究，不能解除目标设备的保真验收缺口。
收尾完整回归265项/8.439秒通过，`git diff --check`通过；新增诊断目录24KB，没有新增模型或音频产物。

Screen the six public long-trained stable 4x64 GFB/MAE checkpoints from the
already verified ASRNN archive. Preserve the previous train-side take-modulo-5
selection partition (234 fit / 54 calibration; this is not session isolation).
Public checkpoints may already have seen all official train data. Calibration
is selection only. No evaluation samples enter training. Require the complete
unchanged `evaluate_asrnn_effect._quality_gate`, including absolute peak P95
<=0.02, plus exact silence, dynamic-control / stream parity <=2e-6, float32
ONNX parity <=2e-6 and CPU RTF <1. Consider fixed convex ensembles only after
single-model screening. No gain patch, normalization or limiter.

## 2. Semantic inverse controls

Retain RAT Volume / identifiable Distortion; resolve Tone and establish CS-3
Attack and DFZ Blend/Filter recovery. Learn labels from paired audio with
explicit spectral-transfer, coherence, and transient-envelope features. Compare
regularized regressors and compact neural models on train-side calibration;
closed-loop reconstruction is a separate test, not a substitute for labels.
RAT retains its existing per-control gates (Volume MAE/P95 <=0.06/0.18,
audible Distortion and Tone <=0.08/0.22; Wet peak threshold 0.001). New CS-3/DFZ
controls use the existing stricter direct-inverse gates MAE/P95 <=0.06/0.18
per control, macro MAE/P95 <=0.05/0.15. Report quiet/unidentifiable samples and
all-sample metrics without silently dropping difficult cases. Retain existing
RAT closed-loop ESR/MAE improvement >=0.90/0.80 versus bypass; new-device
closed-loop improvement >=0.75/0.70. No oracle labels may enter inference.

## 3. Order and chain development improvement

Keep the existing -30 dB counterfactual equivalence mask and exact same expanded
five-domain development audit (320 synthetic examples/domain; 640 real-order
records sampled, of which 320 have eligible relations, exactly as before). The existing search
baseline is fixed in `order-control-paired-public/order-search-metrics.json`.
Add transferable paired time-frequency / envelope interaction features; fit
on training guitars and synthetic replay, select only on calibration. Require
at least +5 percentage points exact order accuracy on BOTH real DAFx and
Pedalboard, no domain exact regression >1 point or pairwise regression >1.5
points, and no higher mean gain-aligned reconstruction error in any domain.
This adds meaningful improvement requirements; it does not weaken an old gate.
Re-run the existing full-chain bypass, silence, streaming, oracle and recovered
control quality gates. Do not reinterpret DAFx's unpublished controls as labels.

These are development gates, not a claim of universal hardware-chain fidelity.
The untouched Tele/final audio stays closed. Any missing true paired evidence
remains explicitly part of work package 4, not fabricated for packages 1–3.

## Live development record

- RAT semantic recovery now passes calibration and all 128 development pairs
  under the original audibility rule: Tone MAE/P95 0.07214/0.21744, Distortion
  0.02665/0.09529, Volume 0.01774/0.05060. Closed-loop global MAE improves 93.84%
  versus bypass. `rat-identifiable-phase10/metrics.json` is the evidence; its
  inference package also passed a raw-audio replay check (maximum normalized
  control difference 2.63e-8, input arrays unchanged). The 22 quiet
  pairs remain disclosed, not silently counted as identifiable.
- DFZ forward: six 4x64 recurrent candidates, linear/conditional readouts,
  recurrent fine-tuning, FIR and causal TCN corrections do not yet pass peak
  fidelity. The nonlinear readout reaches peak P95 0.027221; a signed-waveform
  charge ridge variant reaches 0.026634 but worsens worst-Blend P95 to 0.037780.
  Both miss the unchanged 0.02 / 0.03 gates. Full-prefix final-LSTM adaptation,
  charge interactions and long-context residual training also failed. The
  independent finite-history route completed its fresh two-array raw-input/skip phase and failed (CPU54 peak0.105526); further training is paused by the self-audit above. No rejected model
  is admitted as a faithful DFZ renderer or a replacement for accepted devices.
- CS-3 inverse: label-free causal forward search passes all 352 development
  pairs: Attack MAE/P95 0.04474/0.15000, closed-loop MAE improvement 97.68%.
  Four evenly spaced CPU audio-only replay cases exactly match frozen controls.
  The P95 comparison uses 5e-8 tolerance only for float32 normalized-control
  representation (.15 is stored as .15000000596), never for audio fidelity.
- DFZ inverse: all 288 development pairs pass: Blend MAE/P95 0.02767/0.12500,
  Filter 0.02593/0.09375, closed-loop MAE improvement 91.79%. Four CPU replay
  cases exactly match. Its forward-search teacher remains peak-fidelity rejected;
  inverse semantic admission is explicitly separate from forward admission.
- Order: the final cascade / interaction / temporal head with a bounded
  diagnostic DSP bank passes every frozen five-domain calibration/development
  check against BOTH historical and freshly replayed baselines. Real exact is
  255/320 (79.6875%, +5.3125 points); Pedalboard is 76.2295% (+24.5902 points).
  The immutable package is `runs/order-bankfit-bound-phase11`; all 173 raw-audio
  runtime replays pass, including all 118 bank-triggered development examples.
  Diagnostic fitted controls/gain never alter actual chain controls/output gain.
- `runs/full-chain-phase11-accepted` passes the complete existing chain gates
  with the admitted order package, not candidate mode. All 15 topologies pass
  stream parity (max 5.96e-8); bypass is identical and silence exactly zero.
  Recovered-chain reference/challenge audio improvement versus bypass is
  64.42%/63.04%. A new fixed-control diagnostic finds challenge mean actual
  error 1.49% above the original order, despite passing existing gates; this
  limitation is disclosed, not disguised as universally improved fidelity.
  This is replay of previously observed development/challenge audio, NOT a
  new locked-final result. See `PHASE11_ORDER.md` for inputs and limitations.

## Additional architecture experiments

The public 1x64 infinity-norm and 4x64 spectral-norm pools were also screened
after the original six-model pool failed. Neither passed the unchanged peak
gate. Spectral-norm regularization is explicitly **not** labeled as carrying
the infinity-norm model's formal stability guarantee; see Section 2.1 of the
[updated primary paper](https://arxiv.org/html/2509.15622v2). These are rejected
research candidates, not runtime promotions. The original strict importer
remains the default; spectral imports require an explicit opt-in.

The nonlinear readout reached complete calibration peak P95 0.027221 and was
rejected (gate 0.020000). Further recurrent training replays the **entire causal
prefix** before every gradient window, including low-level source noise, and
projects the input/forget gate symmetry as well as the candidate matrix norm.
Left zero padding only aligns batches and leaves the exact zero state unchanged.
The original short-warmup results are retained as separate rejected evidence.

The complete skip-layer + signed-neighborhood run reached only 0.034949 peak
P95. A predeclared 286-point fixed convex screen of four retained candidates
selected the unchanged best single nonlinear readout (0.027221); no ensemble
was admitted or saved. Averaging is not a demonstrated improvement here.
The remaining long-trained stable 4x8 pool also fails: its six candidates have
calibration peak P95 0.090260–0.196321. These smaller models are not promoted.

The exact frozen-prefix final-LSTM run completed all 600 updates and covered
all 234 fit recordings. Its unchanged step-zero source remained best; final
complete-model CPU calibration reproduces P95 0.027221 / worst Blend 0.036902
exactly. No mutable recurrent state was reused across updates, and the first
three layers stayed byte-identical. This experiment also does not pass.

### DFZ charge-state experiment (not admitted)

Read-only full-prefix diagnostics of nine calibration examples found no global
integer-sample shift in any of them. On seven Blend=0 calibration examples,
the fixed input-derived feature `LP20ms[tanh(100*Dry)^2] -
LP100ms[tanh(100*Dry)^2]` correlated with the low-frequency residual by
0.338–0.637 (median 0.509). Six fit examples showed the same direction
(0.221–0.513, median 0.414); a Blend=100 example reversed sign. These are
selected mechanism diagnostics, not population performance or independent
validation. Target-derived low-pass residuals used for diagnosis are never
features at inference, and their oracle improvement is not a model result.

`stable_charge_residual.py` therefore adds five fixed causal charge states at
1/5/20/100/300 ms to the frozen best nonlinear renderer. Input squared
saturation, adjacent charge differences and the longest state feed a 54-weight
control-conditioned additive readout. This is an input-driven dynamic model,
not per-clip loudness fitting; the residual starts at zero. The fixed ReLU RNN
uses diagonal alpha and complementary input coefficients. Reachable states
are nonnegative, so ReLU is identity and the charge branch is a bounded EMA.
An initial LSTM expression was stopped: three-second Torch/ONNX feature error
reached 6.93e-4 due to slow gate numerical drift. The direct-coefficient bank
reduced this to 7.15e-7 in an independent probe; no old cache is reused. Exact
zero, dynamic stream and long float32 export parity are tested; complete CPU
calibration and all unchanged development/runtime gates remain mandatory.

The 4,000-step charge-readout fit selected unchanged step zero. A separate
predeclared signed-waveform ridge menu slightly improves overall calibration
P95 to 0.026634, but worst-Blend P95 increases to 0.037780; it is rejected.
A further 35-feature quadratic interaction menu (charge states plus original
model output, fit-only RMS scaling) selects unchanged again. Its best new
candidate's sampled P95 is 0.035493 despite lower ESR; no new checkpoint or
runtime schema is created. All failed evidence remains separate.

### Long-context residual experiment (not admitted)

The next fixed experiment retains the best schema-7 renderer unchanged and
adds a zero-initialized, zero-preserving TCN residual driven by Dry. Twelve
dilated blocks of width eight have 5,680 parameters and a finite 8,191-frame
receptive field (170.65 ms), versus 511 frames in the earlier rejected TCN.
Each scored 4,096-frame window includes all 8,190 real preceding samples;
only history before the recording itself is zero padded. This avoids cached
trainable state or shortened warmup. The complete frozen core predictions,
Dry and Wet remain in about 498 MB CPU RAM, not a new disk cache.

The fixed run is 1,200 updates, batch six, LR 3e-4, full calibration every 300
updates, with unchanged step zero retained. Signed waveform, pre-emphasis,
local envelope and signed extrema are optimized together. Complete CPU
calibration is required after selection, before any schema/export/admission.
Synthetic resource smoke and full-context/stream/zero tests pass; these are
pipeline checks, not fidelity results. No public evaluation or locked-final
audio enters this experiment.

The completed Dry-only long-TCN run reaches CPU calibration peak P95
0.026615 / worst Blend 0.037411 and therefore still fails both peak gates.
An output-conditioned width-16/12-block TCN was then trained for 4,000 steps
with the original signed-extrema objective. Every trained checkpoint regressed;
the selected step-zero source again measures 0.027221 / 0.036902 on CPU.

### Aligned waveform and explicit short-history experiments (not admitted)

Read-only diagnostics using the immutable full-prefix prediction cache reproduce
fit234/cal54 peak P95 0.026743/0.027221. There is not evidence here of a large
fit-to-calibration gap or an irreducible recording-noise floor. In 56/234 fit and
12/54 calibration files, the prediction's global peak is too high while its
amplitude at the actual Wet peak is too low. Independent-maximum losses can
therefore request corrections at different events. This is a diagnosis, not a
license to change timing, normalize clips, discard failures, or tune on eval.

The same width-16 long TCN was trained for another fixed 4,000 updates using
same-time signed weighted MSE plus 0.1 pre-emphasis MSE, with no independent
maximum/extrema loss. Its last checkpoint improves global ESR by about 8.8%,
but peak P95 rises to 0.042323 and worst Blend to 0.057349. Every trained
checkpoint fails; no source model is replaced.

`dfz_dynamic_readout.py` next implements an explicit short-history nonlinear
readout: current/past Dry and frozen core output at lags 0/1/2/4/8/16/32/64,
nine independent 32x32 MLPs, fixed fit-only feature RMS and convex bilinear
knob interpolation. The zero-origin expression `f(z)-f(0)` and a zero output
initialization preserve exact silence. Its 14,688 trainable parameters use
only a 64-frame Dry/core FIFO (512 bytes), not cached learned state. The fixed
1,500-step signed-MSE run also selects unchanged step zero: its last trained
peak P95 is 0.047159, despite improved ESR. Complete CPU54 validation again
rejects it. Tests cover actual full-prefix windows, fast/grid paths, dynamic
zero, arbitrary streaming and future-input isolation. These tests prove
implementation properties, not audio fidelity.

All three later experiments share one prediction-only cache,
`runs/dfz-frozen-predictions-phase9.pt` (165,937,193 bytes), bound to source
weights, extractor code, all fit/calibration audio bytes and tensor content.
No Dry/Wet copy or trainable recurrent state is persisted in it. Individual
failed runs retain only a small residual checkpoint and JSON evidence.

The next fixed experiment emphasizes aligned transient events rather than
independent output maxima: each corner contributes an active-uniform window
and separate windows around fixed Wet/core signed extrema. Target supervision
stays at the same sample, with full-fit-recording energy denominators; the
uniform contribution remains half of the loss. A separate nine-file fit-only
capacity probe checks whether the unchanged architecture can learn even one
recording per corner. Neither experiment opens official eval or changes gates.

The 4,000-step fixed-event run has now also completed: all 234 fit recordings
were visited 153–154 times, with 468/468 Wet signed events and 468/468 core
signed events covered. Last-checkpoint global ESR is 0.008515, but peak P95
is 0.052729 / worst Blend 0.062273. All trained checkpoints are rejected;
the complete CPU54 check retains the unchanged source. This rules out omitted
original event labels as an explanation for this run, not all possible sampling
or generalization problems. The following bounded experiment updates current
prediction events on fit audio only, while retaining the same 50/25/25 loss
weights and frozen calibration/admission rules.

That current-event experiment also completed all 4,000 steps and 17 full fit
audits. Fit234 peak P95 rises from 0.026743 to 0.055097 and cal54 from 0.027221
to 0.056158; mean ESR improves on both. All 7,488 refresh-generation/file/sign
events were covered. The final twelve worst fit cases all underestimate peaks.
The issue is not just missing newly oversized peaks or calibration-only
generalization. The selected unchanged source remains rejected for DFZ forward;
no existing runtime bundle was changed.

The completed fit-only capacity probe uses the first fit recording at each
corner (nine take-1 files). Peak-focused training reaches full-fit-nine peak
P95 0.008758 at step 500 from 0.063770 initially. At step 2,000, the event
neighborhood RMSE is 80% lower, but full-file peak P95 worsens to 0.032978 and
ESR to 0.02951. This demonstrates local fitting capacity and damage outside
the oversampled events; it is not calibration/development performance. No
weights are retained or admitted from this diagnostic.

The primary research on [time-varying feature modulation](https://arxiv.org/abs/2211.00497)
and [nonlinear black-box/gray-box effects](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)
motivates checking time-dependent behavior and objective choice rather than
assuming that a wider/deeper convolution alone solves fuzz dynamics. This is
architecture guidance, not proof that this DFZ residual follows a particular
unobserved circuit mechanism or that a proposed architecture will pass.

### Export numerical boundary (not admitted)

The independently tested short-history residual exports correctly in float32:
three-second static/dynamic and irregular-batch/block checks have maximum audio
difference 2.98e-8, exact zero, and residual-only ORT RTF about 0.04. The original
four-layer LSTM core remains a separate issue: a complete-model synthetic
dynamic probe has Torch/ORT difference 1.11e-5, above the unchanged 2e-6 gate.
Eight ORT option combinations produce identical outputs; primitive recurrence
reduces but does not eliminate the error. A mixed Torch/ORT diagnostic reaches
2.98e-8 but is not the required single-ONNX artifact and has almost no runtime
margin. None of these diagnostics establishes deployment readiness or replaces
the required complete trained-model parity and performance evidence.

### Standalone two-rate core (architecture precheck passed; not admitted)

`multirate_fuzz.py` is a genuinely independent forward model, not another
correction attached to the old LSTM. A 48-kHz C16/10-block gated convolutional
network has 2,047-frame audio history. A 750-Hz C8/10-block finite controller
reads descriptors of **previously completed** 64-frame blocks, with 2,047-block
history. Its output modulates audio gates; bias never adds directly to audio.
Current sample controls retain an immediate path. There are 26,160 parameters.

The integer sample clock, true pending samples and held controller values are
explicit streaming state. Incomplete blocks are never flushed with invented
audio. An ignored shadow descriptor keeps the exported convolution valid when
no complete block arrives, without advancing any state or affecting returned
audio. Both paths are finite convolutions, avoiding the old long LSTM numerical
recurrence. The complete design exports as **one** float32 ONNX graph, with an
explicit int64 clock rather than a float-valued hidden-clock convention.

`runs/dfz-multirate-precheck-phase11/metrics.json` records random nonzero-weight
CPU/ORT tests at three and thirty seconds, static/dynamic controls, batch1/2/3,
and irregular chunks down to one frame. Maximum audio difference is 5.96e-8,
ORT whole/stream difference is zero, exact silence/future-prefix checks pass,
and CPU single-thread RTF is about 0.049 whole / 0.052 at1024 frames. The graph
is286,366 bytes; logical mono state tensors use196,712 bytes. These are
architecture/synthetic measurements, not trained DFZ fidelity or production
latency guarantees.

The first fixed training run uses fit234/cal54, 1,500 steps, cal every150,
one fit recording per corner, LR3e-4, and same-time signed waveform supervision.
Every update recomputes the entire low-rate history with current weights; the
fast scored window includes all2,046 true preceding samples. No learned states
are cached, no old core/prediction cache is used, and all original audio remains
unchanged. Synthetic GPU checks passed with roughly568MB driver allocation.
The first1,500 updates completed with CPU54 globalESR0.20057994,
peakP950.27093106 and worst-Blend0.30575427: rejected, despite a decreasing
learning curve. `continue_multirate_fuzz.py` starts from those weights with a
fresh AdamW (not an optimizer resume), then runs a fixed13,500 additional
updates:150-step LR ramp3e-4→1e-3, constant through7,500, cosine to3e-5.
Complete cal54 is checked every1,500 updates; only one gate-first selected
checkpoint is retained. The training recipe and source/audio hashes remain
fixed throughout this15,000-total-update phase.

`audit_multirate_fuzz.py` separately reloads completed, hash-bound trained
weights, exports a fresh graph marked trained-but-not-admitted, and compares
all54 complete calibration recordings on CPU/ORT. Dynamic3/30-second and
irregular-callback probes, exact silence, quiet inputs, future-prefix checks,
state/input validation and single-thread timing are included. This audit does
not register a runtime or replace the full development/quality admission.

The completed15,000-update phase selects its last checkpoint. Full CPU54
globalESR is0.04689439, peakP950.22054231, worst-Blend0.30254709: **rejected**.
Fresh trained-weight real54 plus dynamic3/30-second ONNX audit passes its
runtime checks: maximum real-audio parity1.49757e-6, exact ONNX whole/stream,
exact silence and quiet-input checks. The initial1,500-update weights were
separately audited too. Neither audit's runtime pass is a fidelity admission.

A fixed6,000-update same-time worst16-error supplement was then run with
cosine LR3e-4→1e-5. Every trained checkpoint fails; unchanged step zero remains
selected. Last globalESR0.05004282, peakP950.22103572 and worst-Blend0.30282290
do not improve the required joint result. Full CPU54 reload reproduces the
retained source. This is not evidence that all transient losses are ineffective.

### Wider independent-core candidate (not admitted)

`widen_multirate_fuzz.py` duplicates audio16→32 and controller8→16 channels,
retaining both ten-block histories, sample/control clocks and zero-preserving
signal paths. Unequal outgoing weight splits3/8 and5/8 preserve the mathematical
function while breaking gradient symmetry. There are102,368 parameters. Actual
float32 narrow/wide real-audio difference reaches2.08989e-6, so the extra2e-6
cross-model initialization-equivalence check **fails**. Its audit is retained
as overall-failed, not relabeled as an equivalent model replacement.

The wider candidate's own real54 CPU/ORT parity1.48267e-6, dynamic3/30-second
probes, exact zero/finite-history tail/future isolation, quiet-input peak0.000231,
and single-thread CPU/ORT timing all pass. ORT RTF is0.14–0.17 including boundary
validation. This permits an **approximate warm start for a new training
candidate**, not promotion. `train_wide_multirate_fuzz.py` explicitly retains
the failed cross-model test and independently requires the original within-model
2e-6 parity/safety/timing checks. Its fixed8,000-update fit234/cal54 phase uses
the original signed loss,250-step LR ramp1e-4→1e-3, constant through4,000,
cosine to3e-5; cal every1,000. No fidelity/export/admission gate is loosened.

The completed8,000-update run selects step7,000. Full CPU54 gives globalESR
0.03395953, peakP950.15308363 and worst-Blend0.20135819: rejected. Its own
trained real54 CPU/ORT maximum1.60933e-6, long dynamic/stream, exact silence,
finite-history tail and single-thread real-time checks pass. Complete original
calibration quality formulas also fail on both backends; no official eval is
opened and no package is promoted.

### Temporal optimization coordinates and conditional thresholds

`dfz_temporal_basis.py` changes only the first four fast convolutions' training
coordinates: fixed source weights plus zero-initialized coefficients in a
full-rank DC/first-difference/second-difference tap basis. Derivative directions
are scaled8. Saved weights are ordinary float32 convolutions; there is no new
audio filter, normalization, inference operator, receptive field or state.
Initial CPU/MPS outputs and materialized source weights are exactly unchanged.

The fixed20,000-update run `dfz-wide-basis-20000-phase11` selects step18,000.
Complete selected-model CPU54 gives globalESR0.0261753644,
peakP950.0620387718 and worst-Blend0.0751939952. This substantially improves
the independent route but still fails0.02/0.03 and is less faithful than the
retained old nonlinear source. The full trained audit in
`dfz-wide-basis-20000-audit-phase11` also **fails** real54 CPU/ORT parity:
2.66730785e-6 versus2e-6. Dynamic30-second error is5.24335e-7, ONNX stream
error0, silence/expired-tail/future-prefix0 and quiet peak0.000238. Single-thread
CPU full/stream RTF is0.27–0.28/0.215–0.216, ORT0.178–0.203/0.166–0.169.
Those passing subchecks do not override either failed gate.

The next fixed12,000-update phase is `dfz-centered-core-phase11`, initialized
from the selected18,000-step weights. `centered_multirate_fuzz.py` adds
conditional threshold heads inside each fast nonlinearity:
`tanh(activation+threshold)-tanh(threshold)`. All6,080 new parameters start0;
initial output/source weights are exact. The first1,000 updates train only
these heads; the remaining11,000 jointly train with the same signed loss and
real causal history. This is an experimental architecture, not an output-gain
patch or a claim that thresholds already solve the transient error. Its
108,448 parameters keep393,160 bytes mono state and the2,046-frame audio tail.
Synthetic nonzero-head CPU/ONNX preflight passes (max1.78814e-7), but trained
weights must undergo a separate audit. The real-arithmetic fast-update bound
is3 rather than the plain architecture's1.5; no LSTM norm is fabricated.

`evaluate_multirate_fuzz.py`, `multirate_admission.py` and
`promote_multirate_fuzz.py` keep the complete original quality, provenance,
float32 parity and real-time gates for both explicitly distinguished schemas.
Mixed-architecture evidence is rejected. Official development288 cannot be
opened through this evaluator until same-checkpoint full CPU calibration54
passes; no new locked-final audio is permitted. A model card is activated
last in a new bundle only after all evidence passes. These entrypoints are
implemented/tested, **not evidence that a candidate has been admitted**.
Earlier failed8,000-step quality reports retain their historical evaluator
hashes; source-closure changes require fresh evidence before any future use.

Additional fixed-source diagnostics show fit234 peakP950.09419317 and ESR
0.01898985: the independent model's fit peaks also fail, not just calibration.
Selected layer instrumentation preserves the original ONNX output exactly and
shows numerical differences accumulating downstream. Native/exp/rational
activation-expression probes all miss2e-6 on the selected three difficult
calibration files; none is installed. A noncausal target-derived low-frequency
residual oracle also worsens peakP95 at100/375/1000/4000Hz cutoffs. This is
explicitly not a deployable model or evidence to feed targets into inference.
Compact results are in the20,000-step audit's `diagnostics.json`; four disposable
diagnostic ONNX graphs were removed (2,799,491 bytes), reproducible from their
retained temporary scripts. Existing model/audit graphs and source audio stay.

The complete-fit `fit_multirate_readout.py` screen compares twelve anchored
ridge alternatives: six strengths for a shared32-channel projection and the
same six for nine separate control-corner projections, with no bias or
clip-specific gain. All234 fit recordings contribute full causal features and
signed/pre-emphasized normal equations; no features are cached to disk.
All alternatives worsen calibration peaks (P950.065453–0.069487 versus
0.062039); unchanged wins. `dfz-wide-corner-readout-phase11/metrics.json`
retains the screen, **no new readout weights are saved**.

The conditional-threshold phase completed all12,000 updates, selecting10,000.
Its materialized full CPU54 globalESR0.0249881257, peakP950.0601945966 and
worst-Blend0.0686149687 still fail. Its completed trained audit also fails:
real54 CPU/ONNX maximum2.75671482e-6. Dynamic30-second error6.03497e-7,
ONNX stream0, exact zero/tail/causality and quiet/real-time subchecks pass.
Both full original CPU/ORT calibration-quality reports reject it. No eval
audio is opened and its synthetic preflight is not reused as admission evidence.

The next `train_multirate_l1.py` phase retains this exact architecture and
selected weights, fitting normalized signed-weighted absolute waveform error
plus0.1 absolute pre-emphasis error and0.25 of the preceding signed squared
objective. It keeps the same target-only bounded frame weights, balanced234-fit
sampling and complete current causal history. The fixed8,000-update schedule
ramps1e-5→1e-4 for500 updates, then cosines to1e-5; fullcal54 every1,000,
unchanged source retained as step0. No output gain/normalization/limiter is
introduced. The three-step MPS synthetic smoke has finite gradients and exact
materialization (274MB driver memory); this is a training implementation check,
not audio fidelity. `dfz-centered-absolute-phase11` completed all8,000 updates
without improving the selected source. Last calibration ESR0.02438897,
peakP950.06400401 and worst-Blend0.07743135 fail; unchanged step0 wins.
Final CPU54 and the checkpoint SHA reproduce the source exactly. Its runtime
audit need not be duplicated for identical bytes; the source audit still fails.

Primary-source architecture follow-up: the [NAM WaveNet computation guide](https://neuralampmodelercore.readthedocs.io/en/latest/wavenet_walkthrough.html)
distinguishes layer-to-layer residual features from independently accumulated
skip-to-head features. The [DAFx25 activation study](https://dafx.de/paper-archive/2025/DAFx25_paper_50.pdf)
also uses a summed-layer linear audio head and emphasizes aliasing/fidelity
tradeoffs; it does not validate our DFZ or conditioned-control implementation.
This motivates a separate frozen-core diagnostic, not a claim to have cloned
NAM or a proprietary Neural DSP engine. `fit_multirate_skip_readout.py` tests
eight zero-anchored projections of all ten32-channel pre-projection updates,
using fit-only stride16 plus explicit Wet-peak-local supervision and all54 full
calibration files. Only a passing complete-model replay could activate such a
new path; no feature cache or per-clip output-gain parameter is created.

The all-layer screen selects a shared ridge1e-4 projection: provisional
calibration peakP950.05766425, worst-Blend0.06697099, ESR0.02491719.
This is a modest improvement over its source but still fails both peak gates;
it is **not a full-model/stateful/export result**. Only a13,402-byte readout and
compact metrics are retained in `dfz-centered-skip-screen-phase11`; no runtime
schema, accepted model or application path is changed for this failed screen.

A fit-only window-weight diagnostic (seed1008,64 uniform plus64 peak-local
windows per fit file) measures29,952 windows. Relative to fixed complete-file
inverse-energy weighting,141 windows have a denominator multiplier>10,
61>100 and6>1000 (max1813.15). The largest1% account for80.61% of the sum of
these denominator factors, **not a measured share of gradients or loss**.
Typical uniform/peak-local median factors are0.3400/0.3303. Thus rare low-energy
windows can be strongly reweighted; this does not alone establish causation.
The [NAM training implementation](https://raw.githubusercontent.com/sdatkinson/neural-amp-modeler/main/nam/train/lightning_module.py)
also explicitly distinguishes averaged per-example ESR from pooled ESR and
uses MSE by default; this is contextual guidance, not a claim our task is equal.

`train_multirate_fixed_energy.py` prepares a fixed6,000-update comparison from
the same retained centered source. The original signed squared waveform and
pre95 loss use fit-only complete-file energy denominators, retaining the1e-5
floor and bounded target-frame weights. This changes training weights only:
no Dry/Wet normalization, learned-state cache, output gain or limiter. LR ramps
3e-5→3e-4 over250 steps, holds through3,000, then cosines to1e-5; completecal54
every1,000 and final selected-model CPU replay remain mandatory.
`dfz-centered-fixed-energy-phase11` completed all6,000 updates. Last calibration
ESR0.02382536, peakP950.07485898 and worst-Blend0.08041036 fail; unchanged
step0 wins. Final CPU54 and source checkpoint SHA213ca8a3 reproduce the original
ESR0.02498813 / peak0.06019460 / worst-Blend0.06861497 exactly. Changing the
loss denominator did not produce an acceptable improvement, despite the
weight-imbalance diagnostic. No source model/runtime was replaced.

Follow-up analytical **output-waveform** loss gradients on the same29,952 fit
windows quantify that imbalance: the largest1% account for95.90% of summed
gradient squared norms under window denominators, versus12.47% with fixed-file
denominators. These are not model-parameter gradients or proof of the cause of
failed fidelity. Full centered fit234 ESR0.01735683 and peakP950.07537398 also
fail; the peak problem is not limited to calibration generalization.
The output-gradient script SHA is714cef0193243a57b67885006fc70bf5e62d2e39dbe5bf73772725887451f7b2.
A separate54-file clock diagnostic found phase0/other-phase residual MSE ratio
1.00325; maximum mean error was at phase10, not the block boundary. There is
no strong evidence here that64-frame controller-boundary clicks dominate the
peak failures, so speculative smoothing has not been inserted.

### Fresh two-array raw-input/skip core

`cascaded_multirate_fuzz.py` is a separate architecture, informed by the NAM
residual/skip topology above but implemented independently with our existing
strict completed-block controller. Two arrays have widths16 then8, each ten
causal dilated convolutions. Every layer receives raw Dry and has distinct
residual and8-channel skip projections; the20 skip outputs feed a linear head.
Centered tanh modulation preserves exact zero even with a nonzero slow state.
There is no final gain, audio normalization, teacher, pretrained weights or
learned recurrent state. Audio receptive field4093 frames; controller remains
750Hz with2047-block finite history. Mono explicit state262,184 bytes and
25,080 learned parameters.

`dfz-cascaded-precheck-phase11` passes synthetic CPU/ONNX irregular callbacks,
dynamic controls through30seconds, silence, quiet, expired tail, causality and
single-thread real-time checks: maximum CPU/ORT error5.96046e-8, ORT stream0,
graph382,928bytes. These are untrained implementation checks, not fidelity.
The eight-update MPS training smoke passes (stream3.72529e-9, zero0, driver
954MB, stable steps approximately0.117seconds). Tests cover exact prefix
training, zero-history semantics and typed float32 payload/graph rejection.

`train_cascaded_multirate.py` completed the fixed20,000-update fresh phase
in `dfz-cascaded-core-phase11`. It uses complete-fit pooled Wet/pre95 energies
as fixed loss denominators, the same signed-waveform objective and50/50
active-uniform/peak-local fit234 sampling. LR ramps1e-4→1e-3 over500, holds
through10,000, then cosines to3e-5. Every update recomputes the full current
slow prefix and real4092-frame fast prefix; no mutable-state or feature cache.
Fullcal54 every1,000 and final selected-model CPU54 remain mandatory. Only
one selected checkpoint is retained. The separate trained audit reexports
the actual checkpoint and applies the unchanged full CPU/ORT quality gates;
neither a training curve nor the synthetic preflight can admit the model.
Final CPU54 peak0.105526 / worst-Blend0.112238 rejects this phase. The prepared
trained audit is deliberately not run: passing export could not change that
fidelity rejection. No continuation is queued; see the self-audit above.

Two bounded training diagnostics are complete without saving audio or weights.
The synthetic batch benchmark found only12.48% more scored frames/second at
batch18 (1.781GB process driver allocation) versus9; timings include concurrent
GPU training. Batch36 hit its separate4GB diagnostic budget and stopped; that
larger batch was not used in the actual phase. The subsequent9/18 report is
`dfz-cascaded-training-smoke-phase11/batch-throughput.json`.
The single-fit-file coordinate comparison uses only `0,0,1.wav`, identical
seeded fresh weights,1,000 updates, starts and LR3e-4. Both variants fail even
in sample; derivative coordinates improve its peak error from approximately
0.19103 to0.14359 and ESR0.10771 to0.09727. This is insufficient to authorize
a long optimization follow-up, a generalization claim or a passing model. The full CPU results and source
hashes are retained in `fit-coordinate-diagnostic.json`; no candidate is saved.
The training-only coordinate module materializes back to the same ordinary
float32 architecture and passes exact initial/changed CPU equivalence tests.

The manufacturer's [Duality Fuzz manual](https://www.darkglass.com/en-int/pages/duality-fuzz-manual)
describes separate fuzz circuits, clean/fuzz Blend and a high-frequency Filter.
Crucially, the [dataset paper section3.2](https://arxiv.org/html/2509.15622v2#S3.SS2)
clarifies that the recorded controls are **fuzz blend** and Filter, while
Level is maximum and the separate **dry blend is fully wet**. The legacy
dataset/runtime key `blend` denotes the former, not a dry/wet mix. Labels,
trained tensors and accepted semantic evidence are unchanged; no dry-baseline
crossfade constraint should be inferred from the first control being zero.
These published fixed settings still do not prove the charge-state hypothesis
or fidelity at unrecorded knob values.

## Retention policy

Use fresh run directories, keep compact failed metrics/provenance, delete only
this work's rejected weights and disposable feature caches. Keep one accepted
runtime/model bundle per task and the original verified archives. No commits or
pushes without a separate request.

Cleanup completed: discarded feature-inverse weights/caches for RAT, CS-3 and
DFZ; removed the redundant RAT training Tone head after its accepted self-contained
semantic bundle passed replay; removed rejected FIR, full-core, TCN, shallow
and spectral DFZ weights. JSON evidence remains, and discarded outputs can be
regenerated from retained source datasets/archives and scripts. After the
unsuccessful convex screen, rejected transient/full-prefix/skip weights and the
duplicate step-zero neighborhood checkpoint were also removed. The best DFZ
nonlinear readout and public source core remain as comparison/provenance dependencies.

A later proposed cleanup of 22 obsolete order weights/caches (386.42 MiB)
was blocked by the approval reviewer. **Those files have not been deleted.**
The precise list is in `/private/tmp/muspector-order-cleanup.urKzpk/order-cleanup-audit.json`;
source audio, JSON evidence, the accepted order bundle and its provenance
dependencies are excluded. Cleanup awaits explicit approval and does not
block offline training.
