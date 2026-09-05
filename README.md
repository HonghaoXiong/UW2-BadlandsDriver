# UW2-BadlandsDriver — 在 Underworld2 里原生直调 Badlands

> 单文件 [`badlands_driver.py`](badlands_driver.py)：把 Badlands 作为**同进程 Python 库**
> 直接挂进 Underworld2（UW2）模型，不经 UWGeodynamics 的 `surfaceProcesses.Badlands`
> 封装。零第三方依赖（underworld2 环境自带 numpy/scipy）。
>
> 血统：UWG `surfaceProcesses.Badlands`（underworld2 2.17.3 源 `surfaceProcesses.py`
> L48-537）语义逐行平移，仅两处故意偏差（§一.2）；等价性经完整 2D 模型双路径
> A/B 数值验收（84 帧全数据集逐位一致）。

---

## 一、它是什么、调用链长什么样

### 1. 一句话

`BadlandsDriver` 是一个鸭子类型（duck typing）的表面过程驱动类：把它挂到
`GEO.Model.surfaceProcesses` 上，UW2 主循环每个时间步会自动调用它的 `solve(dt)`——采样
UW 表面速度注入 badlands 构造位移、推进 badlands、把新表面读回并转换 UW 粒子材料
（air↔sediment）。badlands 全程作为**同进程 Python 库**运行（无子进程、无文件交换）。
鸭子类型指：调用方只检查对象"有哪些方法/属性"（行为），不检查它继承自哪个类
（"走起来像鸭子、叫起来像鸭子，那就是鸭子"）——所以驱动不需要继承任何 UWG 类。

```
GEO.Model.run_for(...)
 └── Model._update()（每步，平流/popcontrol 之后）
      └── surfaceProcesses.solve(dt)          ← 挂载点：鸭子类型，只要求 solve(dt) 方法
           └── BadlandsDriver.solve(dt)
                ├─ help_gen_sample_point()    动态表面采样点（bcast）
                ├─ velocityField.evaluate_global(nd_coords)   全体 rank 集体采样
                ├─ _inject_badlands_displacement()  写 force.T_disp 时间窗 + force.injected_disps
                ├─ bdm.force.next_display = t+dt    检查点与 UW 步对齐
                ├─ bdm.run_to_time(t+dt)            ← badlands.Model 仅 rank 0 存在
                └─ _update_material_types()   TIN 高程 bcast → 各 rank 本地掩码转换
```

UWG 挂载点为什么能接受外部类：`_model.py` 的 surfaceProcesses setter 只做
`obj.timeField = ...; obj.Model = ...`（触发初始化），主循环只调 `solve(dt)`——
不校验继承关系。

### 2. 与 UWG 封装的两处故意偏差（其余逐行等价）

| 偏差 | 内容 | 原因 |
| --- | --- | --- |
| 去 ABC | 不继承 `SurfaceProcesses` 抽象基类（py2 兼容遗留），改为普通类 + `Model` property setter 触发 `_init_model` | 挂载契约不变、少一层上游依赖 |
| 工厂方法（factory method） | badlands 实例化收敛到 `_create_badlands_model()`，全模块唯一 `from badlands.model import Model` 位置 | 单测注入 stub 的接缝（test seam，Feathers 术语：不改动代码即可替换行为的切入点） |

---

## 二、前置条件

### 1. 软件版本

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| underworld2 | 2.17.x（含 UWGeodynamics，同包分发） | 驱动只用 underworld 核心（mpi/scaling/function），挂载需要 UWG 的 `GEO.Model` |
| badlands | 2.3.x，须含 UW 耦合钩子 | 钩子（XML `<udw>`、`force.injected_disps`、`getSea(udw)`）随官方 badlands 2.3.1 本体分发；见下方指纹自检 |
| numpy / scipy | 随上述依赖 | `scipy.interpolate.griddata/interp1d/CloughTocher2DInterpolator`、`scipy.ndimage.filters.gaussian_filter` |

### 2. badlands 钩子指纹自检（30 秒）

对你的安装跑一遍，三行全 True 即具备直调前提：

```python
import inspect
from badlands.forcing import forceSim, xmlParser
from badlands.model import Model
print("injected_disps:", "injected_disps" in inspect.getsource(forceSim))      # 位移内存注入
print("udw xml tag  :", "udw" in inspect.getsource(xmlParser))                 # 耦合门
print("udw tEnd gate:", "udw" in inspect.getsource(Model.run_to_time))         # run_to_time 可反复调
```

### 3. badlands 稳健性补丁（可选，但推荐）

驱动本体不依赖它们；它们修的是**长耦合运行**的正确性问题。补丁版源文件
（model.py / simulation/buildFlux.py / underland/strataMesh.py）见
[Badlands_coupling_fixedStrata](https://github.com/HonghaoXiong/Badlands_coupling_fixedStrata)：

| 补丁 | 不打的后果 | 何时必须 |
| --- | --- | --- |
| strata 可选（`strata_enabled`） | 无 `<strata>` 节点时 xmlParser 行为异常 | 不开地层的用户 |
| fluxTarget 修复 | `run_to_time` 输沙目标时刻错误 | 所有耦合运行 |
| tNow 对齐 + minDT 钳制 | 反复调用 `run_to_time` 时 tNow 漂移/CFL 过冲 | 所有耦合运行（`run_to_time` 逐段推进依赖它） |

### 4. MPI 线程纪律（运行前环境变量）

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

OpenBLAS 多线程与 MPI 混跑会"加核更慢"并干扰计时。

---

## 三、模块放在哪里（位置 → import 语句对照）

**一句话：把 `badlands_driver.py` 拷到你运行脚本所在目录，直接
`from badlands_driver import BadlandsDriver`——零配置。**
想一劳永逸就拷进 site-packages；clone 本仓库加 PYTHONPATH 则不用拷。

| 放法 | 具体位置 | import 语句 | 适用 |
| --- | --- | --- | --- |
| ① 脚本同目录（**推荐起步**） | `badlands_driver.py` 与你的主脚本同目录（Python 自动把脚本目录加入搜索路径） | `from badlands_driver import BadlandsDriver` | 零配置，单模型 |
| ② site-packages（一劳永逸） | `cp badlands_driver.py $(python -c "import site; print(site.getsitepackages()[0])")` | `from badlands_driver import BadlandsDriver`（任意目录可 import） | 多模型/笔记本长期用 |
| ③ 独立 lib 目录 | 如 `~/libs/badlands_driver.py`，运行前 `export PYTHONPATH=~/libs` | `from badlands_driver import BadlandsDriver` | 多环境共享一份 |
| ④ clone 本仓库 | `git clone https://github.com/HonghaoXiong/UW2-BadlandsDriver.git`，`export PYTHONPATH=<克隆目录>` | `from badlands_driver import BadlandsDriver` | 跟随仓库更新（`git pull` 即最新） |

放好后的 30 秒验证（在任意与模块无关的目录执行）：

```bash
python -c "from badlands_driver import BadlandsDriver; print('OK')"
```

注意：放法 ①②③ 是**副本**，本仓库更新后须重新拷贝，否则拿到的是旧驱动；
放法 ④ 保持单一来源。

---

## 四、最小接入（五步）

### 步骤 1：材料系统准备

UW 侧必须有**空气相**与**沉积相**两个材料索引（驱动按掩码在两者间转换粒子）：

```python
from underworld import UWGeodynamics as GEO
from underworld.scaling import units as u

air = GEO.Material(name="Air", density=1.)          # sticky air（黏性空气层），密度≈0
sediment = GEO.Material(name="Sediment", density=2400.)
# airIndex 可以是列表（多种空气类材料都会被转换）；sedimentIndex 是单一索引
```

陷阱（UWG 通用坑，与驱动无关但必踩）：自定义初始地温时
`Model.init_model(temperature=None)` 必须显式传 `None`，默认 `"steady-state"` 会
**静默覆盖**你的初始地温。

### 步骤 2：初始地形

两种来源：

- **常数/函数**：`BadlandsDriver(..., surfElevation=<UW function 或常数>)`——驱动把它
  求值到规则网格生成 DEM（米制落盘系统临时目录 `tempfile.gettempdir()/dem.csv`
  后交给 badlands）；
- **解析地形**（推荐）：子类覆写 `_generate_dem()` 返回 `(nx*ny, 3)` 数组（米制），
  返回数组的 z 列就是你的初始高程函数。

### 步骤 3：准备 badlands XML

最小模板：

```xml
<badlands>
  <grid>
    <resfactor>1</resfactor>
    <boundary>wall</boundary>   <!-- 2D 耦合惯例：侧壁封闭 -->
    <udw>1</udw>                <!-- ★ UW 耦合门：tEnd 让位于 run_to_time 传入值、
                                     允许 run_to_time 反复调用、终态强制写地层 -->
    <nopit>0</nopit>
  </grid>
  <time>
    <start>0.</start>
    <end>200000.</end>          <!-- ★ ≥ 2× 实际耦合时长（strata 层数组按 tEnd 预分配） -->
    <display>250.</display>     <!-- 会被驱动的 checkpoint_interval 覆写，保持一致即可 -->
  </time>
  <precipitation>...</precipitation>   <!-- 按需 -->
  <sp_law>...</sp_law>                 <!-- 按需：erodibility/m/n/fillmax 等 -->
  <outfolder>outbdls</outfolder>       <!-- 实际会被驱动的 outputDir 参数覆写 -->
</badlands>
```

`<demfile>` **不要写**：DEM 由驱动从 UW 表面生成并经 `_build_mesh` 注入（XML 无
demfile 时 badlands 明确支持后置建网格，model.py:133-136）。开 `<strata>` 时的要求
见 §五.6。

### 步骤 4：挂载（在模型求解开始前）

```python
from badlands_driver import BadlandsDriver

Model.surfaceProcesses = BadlandsDriver(
    airIndex=[air.index],                # 列表：所有会被视作"空气"的材料号
    sedimentIndex=sediment.index,
    XML="badlands.xml",                  # 你的参数文件
    resolution=1000.0 * u.meter,         # badlands DEM 分辨率（带单位）
    checkpoint_interval=250.0 * u.year,  # 覆写 tDisplay：badlands 帧与 UW 步对齐
    outputDir="/abs/path/to/outbdls",    # 建议绝对路径（相对路径语义见 §七坑 2）
    aspectRatio2d=0.25,                  # 仅 2D：伪造第二水平维的纵横比
)
```

赋值瞬间即完成初始化（setter 触发 `_init_model`：load_xml → 生成 DEM → 建网格 →
属性覆写 → 初始材料转换）。

### 步骤 5：运行

```python
Model.run_for(duration=10.0 * u.kiloyear, dt=250.0 * u.year, checkpoint=...)
```

主循环每步自动 `solve(dt)`。**不要自己循环调 solve**——除非你在写 ALE-IB 自由表面
那类特殊路径（见 §五.5）。

---

## 五、设计 / 模型设置要求（正确调用的前提）

| # | 设置 | 要求与原因 |
| --- | --- | --- |
| 1 | **分辨率关系** | badlands DEM 分辨率通常**细于** UW 网格（如 1 km vs 5-10 km）。材料判定用 `griddata(nearest)`——badlands 粗于 UW 时精度会退化 |
| 2 | **2D 第三维** | badlands 内部是 3D TIN；2D 耦合用 `aspectRatio2d` 伪造第二水平维，注入位移沿 `recGrid.rny` tile、y 向位移恒 0。选值使伪维足够窄即可（典型 0.25） |
| 3 | **minCoord/maxCoord** | 缺省取 UW 网格范围（nd 化）；2D 下两维都由 x 范围×`aspectRatio2d` 推出。显式传参可覆盖（带单位） |
| 4 | **时间节奏** | `checkpoint_interval` = UW 步长 → badlands 每步 1 帧对齐；`<laytime> ≤ tDisplay`（整除）；udw 语义 = **每耦合步强制 1 帧 + 1 层**（地层记录契约） |
| 5 | **自由表面 / ALE-IB** | 走 `FreeSurfaceProcessor` 路径时，`solve(dt)` 的**返回值契约**：`Model._freeSurface_ALEIB` 为真时返回 badlands 表面插值函数（2D=`interp1d`，3D=`CloughTocher2DInterpolator`），否则返回 None。普通耦合忽略返回值 |
| 6 | **strata 开启时** | `strataMesh` 在 `_build_mesh` 时捕获 **xmlParser 的 CWD 相对 outDir**，早于驱动覆写 `input.outDir`——不修则 sed 文件写杂散目录。挂载后补一段（写法天然 rank 安全，非 rank 0 上 `badlands_model` 属性不存在）：`bdm = getattr(Model.surfaceProcesses, "badlands_model", None)` → `if bdm is not None and getattr(bdm, "strata", None) is not None: bdm.strata.folder = <你的 outputDir>` |
| 7 | **restart** | 走**构造期**：`BadlandsDriver(..., restartFolder=<run>/outbdls, restartStep=<帧号>)`，驱动会解析 `xmf/tin.time{N}.xmf` 恢复 `time_years` 并开 badlands 内部 restart。注意：UWG `Model.restart()` 的 badlands 重建路径按 `isinstance(obj, surfaceProcesses.Badlands)` 分发，对本驱动**不触发**——先 `Model.restart()` 再挂载驱动即可 |
| 8 | **磁盘规划** | 开 strata 时 sed 全量帧随层数线性增长（O(层数)/帧；40 Myr 量级长时程可达 TB 级），先估算 层数×帧大小 |

---

## 六、单位口径（三套并存，历史上出过 bug）

| 侧 | 口径 | 记住 |
| --- | --- | --- |
| UW2 场量 | 无量纲 nd | 温度 ×KT(=700) 转 K；粒子坐标 nd 数值 = km（当 `[length]` 缩放=1000 m） |
| badlands | **米 / 年** | DEM、位移、高程、tNow 全部米制年制 |
| 接口换算点（驱动内部三处） | — | ① `_generate_dem` 末尾 `dimensionise(..., u.meter)` 写出米；② `solve` 里速度×dt `dimensionise(..., u.meter)` 注入米；③ 读回 TIN 高程 `/fact` 转 nd。跨系统写自己的插值/比较时，先写换算注释再写代码 |

---

## 七、注意事项与坑清单

| # | 坑 | 症状 / 防线 |
| --- | --- | --- |
| 1 | **MPI 集合通信（collective）纪律** | `solve`/`_init_model` 是全体 rank 集体调用点：所有 `comm.bcast`/`Barrier`/`evaluate_global` 必须各 rank 一致到达。绝不要把 `solve` 包进 `if rank==0`（死锁）；badlands 调用本身驱动已限 rank 0 |
| 2 | **输出目录相对路径** | `outputDir` 建议绝对路径；相对路径按**解析 XML 时的 CWD** 解释（xmlParser 构造即建目录）。strata 另有坑 6（§五.6） |
| 3 | **XML `<end>` 预分配** | `<end>` ≥ 2× 耦合时长，否则 strata 层数组越界/截断 |
| 4 | **临时 DEM 竞态** | DEM 固定写 `tempfile.gettempdir()/dem.csv`——同一机器并发多个耦合 run 会互相覆盖。多 run 并行时改子类覆写 `_demfile` 命名 |
| 5 | **pint 浮点噪声** | `dimensionise` 年换算带 1e-14 级噪声（250→249.999...97），驱动已在 `dt_years` 处 `np.round(...,6)` 消除；你自己做时间比较时同样要 round |
| 6 | **airIndex 是列表语义** | 材料掩码用 `np.in1d`：airIndex 里任一材料号的粒子低于表面都会转成 `sedimentIndex`（单值）；侵蚀方向统一转回 `airIndex[0]` |
| 7 | **速度采样风格** | `solve(dt, uw_sample_style=0)` 默认用 badlands 当前动态表面插值采样；`1`=旧法（静态初始 recGrid 高程）。`sigma>0` 对注入位移做高斯平滑（步长单位） |
| 8 | **badlands 串行瓶颈** | badlands 只在 rank 0 跑：UW 并行核数收益递减（Amdahl 定律）。性能问题先归因 badlands 段占比再谈加核 |
| 9 | **verbose 输出** | 默认每步打印紫色 "Processing surface with Badlands..."；生产日志嫌噪传 `verbose=False` |
| 10 | **swarm 粒子身份** | 跨 checkpoint 追踪粒子用 `globalIndex` 变量（`rank*10**10+arange`），行号在 popcontrol/MPI 下会断 |

---

## 八、扩展点与测试方法

### 1. 子类覆写点

| 方法 | 覆写目的 |
| --- | --- |
| `_generate_dem()` | 解析初始地形（米制数组返回） |
| `_update_material_types()` | 多沉积相分类（原生只支持单一 sedimentIndex；可按水深/沉积速率/盆地位置分派多种沉积材料） |
| `_init_model()` | `super()` 后追加属性覆写（strata folder、随机种子等） |
| `_create_badlands_model()` | 测试注入 stub（下条） |

### 2. 单测 stub 模式

写一个记录调用、不演化状态的假 badlands（具备 `input`/`force`/`recGrid` 命名空间与
`load_xml`/`_build_mesh`/`run_to_time` 三个方法）+ 假 UW Model（mesh/velocityField/
swarm/materialField 四件套），在测试子类里覆写 `_create_badlands_model` 返回 stub，
即可离线测 DEM/注入簿记/掩码方向/覆写序列，不需要真实 badlands。注意：若测试依赖
默认缩放（1 nd=1 m），必须先重置 `GEO.scaling_coefficients`（`underworld.scaling`
的全局字典可能被同进程其他模块污染），用例后恢复。

### 3. 等价性 / 回归验证方法论

改驱动或换 badlands 版本后：同随机种子双跑（老/新路径）→ 逐帧 diff
`outbdls/h5/tin.time*.hdf5`（elevation/cumdiff/lake）+ `sed.time*.hdf5` + 终态材料场，
判据 max|diff|=0。

---

## 九、与其它路径的关系

- **UWG 原生封装**（`underworld.UWGeodynamics.surfaceProcesses.Badlands`）：语义与
  本驱动等价（除 §一.2 两处偏差），仍是有效的参照实现。
- **进程外耦合**（子进程+文件交换）：本驱动不涉及；仅在需要 badlands 版本解耦时评估
  （会失去 `injected_disps` 内存注入与 `run_to_time` 逐段推进两个核心钩子）。
- **其它地表过程后端**：驱动把 badlands API 面收敛到一个文件（1 类 + 3 方法 +
  ~20 属性），换后端时只需重写该文件内的属性访问点——这正是直调的设计动机之一。

---

## 许可证

- 本仓库按 **LGPL-3.0** 发布（见 [LICENSE](LICENSE)）：`badlands_driver.py` 是
  [underworld2/UWGeodynamics](https://github.com/underworldcode/underworld2)
  `surfaceProcesses.py` 的语义平移，属其衍生作品。
- [badlands](https://github.com/badlands-model) 本体是独立上游依赖，遵循其自身许可证；
  稳健性补丁版源文件另见
  [Badlands_coupling_fixedStrata](https://github.com/HonghaoXiong/Badlands_coupling_fixedStrata)。

---

## 附：接入前 10 项自检

1. 模块已按 §三放置，`python -c "from badlands_driver import BadlandsDriver"` 与 `python -c "import underworld, badlands"` 全绿？
2. §二.2 指纹三断言全 True？
3. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` 已 export？
4. XML 有 `<udw>1</udw>`、无 `<demfile>`、`<end>` ≥ 2× 耦合时长？
5. `<laytime>`（若开 strata）≤ 且整除 `checkpoint_interval`？
6. 材料系统有 air（可多）与 sediment（单一）索引？air 粒子真的分布在 swarm 里？
7. `init_model(temperature=None)`（若用自定义地温）？
8. `outputDir` 绝对路径且已建 `outbdls/h5`、`outbdls/xmf` 子目录？
9. 开 strata：挂载后同步了 `badlands_model.strata.folder`（§五.6）？
10. 单机多 run 并行：处理了临时目录 dem.csv 竞态（§七坑 4）？
