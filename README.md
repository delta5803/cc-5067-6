# 非线性饱和音色处理器（真实多速率链路）

带显式传递函数与驱动参数的流式饱和器，通过**真实升采样 → 非线性 → 降采样**
抑制混叠，支持动态倍率切换，并提供浏览器母带试听面板。

## 界面怎样启动

依赖（本机无 root，已装到用户 site-packages）：

```bash
python3 -m pip install --user -r requirements.txt
```

启动试听面板（仅用标准库 `http.server`，不需要 Tk / PortAudio）：

```bash
PYTHONPATH=/workspace python3 -P -m saturation.panel --port 8000
```

然后浏览器打开 `http://127.0.0.1:8000/`。面板上可：

- 选择测试信号源（拨弦 / 对数扫频 / 多音 / 不对称波 / 超范围信号）
- 改变驱动量 drive、传递函数（tanh / hard_clip / sine / asym）
- 切换过采样倍率 x1（关闭）/ x2 / x4 / x8
- 点“渲染并分析”由 `SaturationProcessor` 内核产出湿声 WAV（Web Audio 播放）、
  干声对照与谐波分布图；点“A/B 对比 x1↔x8”连续试听两种倍率
- 参数非法时内核返回结构化错误，直接回显在控件下方，不会静默失败

环境若装有 PortAudio + sounddevice，可另用 CLI 实时试听：
`python3 -c "from saturation.playback import play; play('pluck', 6.0, 8, 'tanh')"`

## 重新生成测量产物与跑测试

```bash
PYTHONPATH=/workspace python3 -P experiments/run_measurements.py   # 写 output/
PYTHONPATH=/workspace python3 -P -m pytest tests/ -q               # 41 项测试
```

`-P` 防止当前目录进入 `sys.path`（抵御植入的同名模块），库来自受信任用户目录。

## 架构与关键设计

| 模块 | 职责 |
| --- | --- |
| `saturation/transfer.py` | 四条**显式传递函数**与闭式幅度上界、对称性、直流量检查 |
| `saturation/saturator.py` | 带记忆饱和内核：输入/输出一极点 + 逐样本 drive 平滑 |
| `saturation/resample.py` | Kaiser 窗 FIR、零填充真升采样（非样本重复）、跨块滤波状态、整数延迟与尾音排空 |
| `saturation/processor.py` | 流式处理器：统一恒定延迟、等功率交叉淡化、预热/延交/回退决策 |
| `saturation/analysis.py` | 谐波、THD、折返频率、通带保持、互相关时序测量 |
| `saturation/panel*.py` | 母带面板后端与 HTTP 界面，全部由同一 DSP 内核驱动 |

### 传递函数（均显式、有限、有界）

- `tanh`: y = tanh(drive·x)，奇对称，界 1.0
- `hard_clip`: clip(drive·x, ±1)，奇对称，界 1.0
- `sine`: sin(clip(drive·x, ±π/2))，奇对称，界 1.0
- `asym`: 正瓣 tanh、负瓣 0.5·tanh，非对称，界 [−0.5, +1]，产生直流

输入远超满度（如 ±50、drive=20）时输出仍有限且稳定（有测试覆盖）。

### 多速率链路（不是重复样本，也不是切掉高频）

升采样 = **零填充后接 Kaiser 抗成像 FIR（增益补偿 L）**，在 L·fs 上跑饱和，
再经同类抗混叠 FIR 后抽取。FIR 抽头数强制满足群延迟为 L 的整数倍，
因此抽取相位为零、延迟为整数基速率样本。滤波状态跨块延续，
分块处理结果与整段离线计算逐样本一致（多个块尺寸的测试）。

### 动态倍率切换与双跑预算

宿主只有**短时双跑预算**。每个倍率是一条独立引擎，全部对齐到同一恒定延迟
（最慢支路的延迟），切换不移动时间轴。请求切换时按块决策：

1. **预热 warm-up**：块电平低于 −60 dBFS（长静音）→ 空闲支路在零输入上预热，
   无听感风险地完成切换；
2. **交叉淡化 crossfade**：有声但双跑成本 ≤ 预算 → 短双跑 + 等功率交叉淡化，
   冷启动支路的伪瞬态被淡化掩盖；
3. **延交 defer**：有声且超预算 → 保持当前倍率，后续块重试，直到安静间隙；
4. **回退 fallback**：`force_factor()` 硬切（无预热），事件被显式标记为
   `fallback` 以记录伪瞬态风险。

带记忆支路从零启动时，FIR 延迟线与 drive 平滑器与在役支路历史不同，
首个湿声块必然产生差异（`test_cold_branch_produces_pseudo_transient_without_warmup`
量化了该差异随状态收敛而衰减），这正是需要预热/淡化的原因。

## 实测结果（output/metrics.json，fs=48 kHz，drive=6）

- **单音 3.4 kHz 混叠能量**：x1 = 0.0423 → x8 = 1.18e-6，约 **3.5 万倍**抑制，
  20.8 kHz 折返线在过采样后落入噪声底（见 `single_tone.png`）
- **13 kHz 高激励折返**：9 kHz 折叠谱线 0.347 → 0.00026（约 1300 倍），
  基频保留
- **19 kHz 双音**：9 kHz 折叠线 x8 降到 x1 的 1/10 以下，同时 15 kHz
  合法带内内容保留在 6 dB 内（带宽没有被砖墙切掉）
- **小信号扫频**：通带谱形相关 > 0.95，5–18 kHz 能量损失 < 3 dB
- **非对称波**：测量到输入/输出直流偏移与幅度上界（asym 曲线正/负瓣不对称）
- **动态切换时序**：长静音后启用 x8 与持续拨弦中切换，稳定后对全程 x8
  参考录音的互相关偏移均为 **0 样本**；拨弦场景先 defer，在 −60 dBFS
  安静间隙以 warm-up 完成（见 `dynamic_switch.png` 与事件日志）
- **驱动扫描**：THD 随 drive 单调变化（−38 dB → −8.4 dB），参数平滑进入处理，
  无拉链噪声

## 输出文件

`output/` 含各倍率 WAV（单音/扫频/多音/双音/非对称/两个切换场景及全程 x8 参考）、
三张分析图与 `metrics.json`。所有 WAV 由真实多速率链路渲染，
未对声卡推到削波，也未用粗暴低通冒充“无损去混叠”。
