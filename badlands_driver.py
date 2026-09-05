#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
badlands_driver.py — BadlandsDriver：badlands 同进程直调驱动（不经 UWG 封装）
=====================================================================================
License: LGPL-3.0（本文件是 underworld2/UWGeodynamics surfaceProcesses.py 的语义
平移，属其衍生作品，见 LICENSE；上游 https://github.com/underworldcode/underworld2）
用法指南见本仓库 README.md（放置位置、最小接入五步、模型设置要求、坑清单与
badlands 耦合钩子指纹自检）。

血统与等价性：
  本类是 UWGeodynamics surfaceProcesses.Badlands（uw 2.17.3，
  src/underworld/UWGeodynamics/surfaceProcesses.py L48-537）的语义逐行平移，
  目的：让用户直接持有 badlands API 面（1 类 + load_xml/_build_mesh/run_to_time
  + ~20 个 input/force/recGrid 属性），摆脱对 UWG 封装层的继承依赖。
  验收：完整 2D 模型双路径 A/B 数值等价（84 帧全数据集逐位一致）。

与 UWG 原类的故意偏差（仅两处，其余逐行等价）：
  1. 不再继承 UWG SurfaceProcesses ABC（py2 兼容遗留）；改为普通类 +
     Model property setter 触发 _init_model——UWG Model.surfaceProcesses
     挂载点是鸭子类型（duck typing：只看对象有哪些方法/属性，不查继承树；
     _model.py:537 setter + :1849 solve(dt) 调用），
     不校验继承关系，挂载契约不变。
  2. badlands Model 的实例化收敛到 _create_badlands_model() 工厂方法
     （factory method），是全模块唯一 `from badlands.model import Model`
     位置——单测注入 stub 的接缝（test seam）。

MPI 安全（uw2-mpi-safety 规则，与 UWG 原实现一致）：
  - badlands 仅在 rank 0 实例化与推进；本类所有方法均为全体 rank 集体
    调用点（_init_model / solve），跨 rank 数据一律 comm.bcast + Barrier，
    UW 速度采样用 velocityField.evaluate_global（collective）
  - restart 语义：UWG `Model.restart→restart_badlands` 走 isinstance 分发，
    对本类不触发；推荐时序=先 Model.restart() 再挂载本驱动，
    badlands 重启经构造参数 restartFolder/restartStep 在 _init_model 内处理。

坑清单（平移时逐条保留）：
  - strataMesh CWD 捕获：input.outDir 覆写晚于 _build_mesh，
    sed.time*.hdf5 目录须在挂载后同步 strata.folder（README §五.6）
  - XML <end> ≥ 2× 耦合时长（strata 层数组预分配）
  - 2D 耦合用 aspectRatio2d 伪造第二水平维；位移 tile 到 rny 列
"""
import sys
from tempfile import gettempdir

import numpy as np
import underworld as uw
from scipy.interpolate import CloughTocher2DInterpolator, griddata, interp1d
from scipy.ndimage.filters import gaussian_filter
from underworld.scaling import dimensionalise
from underworld.scaling import non_dimensionalise as nd
from underworld.scaling import units as u

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

_tempdir = gettempdir()


class BadlandsDriver:
    """badlands 同进程直调驱动（鸭子类型挂 UWG Model.surfaceProcesses）。

    挂载契约（与 UWG Badlands 一致）：
      Model.surfaceProcesses = driver_obj 后，UWG 依次赋 timeField、Model；
      Model setter 触发 _init_model()；主循环每步（平流/popcontrol 之后）
      调 solve(dt)。
    """

    def __init__(self, airIndex,
                 sedimentIndex, XML, resolution, checkpoint_interval,
                 surfElevation=0., verbose=True, Model=None, outputDir="outbdls",
                 restartFolder=None, restartStep=None, timeField=None,
                 minCoord=None, maxCoord=None, aspectRatio2d=1.):
        """
        参数与 UWG Badlands 完全同名同义（子类 BadlandsFacies 的
        super().__init__ 调用零改动）：

            airIndex / sedimentIndex : UW 材料号（air 可为列表）
            XML        : badlands 参数文件路径（构造后属性覆写优先于 XML 值）
            resolution : badlands DEM 分辨率（带单位量，内部 nd 化）
            checkpoint_interval : 覆写 badlands tDisplay（带单位量）
            surfElevation : 初始 DEM 高程（UW function 或常数，默认 0）
            outputDir / restartFolder / restartStep : 输出与重启目录/步号
            timeField / minCoord / maxCoord / aspectRatio2d : 同 UWG

        badlands 仅存活于 rank 0；本对象全体 rank 一致构造。
        """
        try:
            import badlands  # noqa: F401  可用性检查，与 UWG 原实现一致

        except ImportError as e:
            raise ImportError("""badlands import as failed. Please check your
                              installation, PYTHONPATH and PATH environment
                              variables""") from e

        self.verbose = verbose
        self.outputDir = outputDir
        self.restartStep = restartStep
        self.restartFolder = restartFolder

        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.resolution = nd(resolution)
        self.surfElevation = uw.function.Function.convert(nd(surfElevation))
        self.checkpoint_interval = nd(checkpoint_interval)
        self.timeField = timeField
        self.XML = XML
        self.time_years = 0.
        self.minCoord = minCoord
        self.maxCoord = maxCoord
        self.aspectRatio2d = aspectRatio2d
        self.Model = Model

    # ------------------------------------------------------------ 挂载契约 --
    @property
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, value):
        self._Model = value
        if value:
            self._init_model()

    # --------------------------------------------------------- badlands 接缝 --
    def _create_badlands_model(self):
        """全模块唯一 badlands 实例化点（rank 0 内调用）。

        单测经子类覆写本方法注入 stub（不触真实 badlands）。
        """
        from badlands.model import Model as BadlandsModel
        return BadlandsModel()

    # ------------------------------------------------------------- 初始化 --
    def _init_model(self):

        if self.minCoord:
            self.minCoord = tuple([nd(val) for val in self.minCoord])
        else:
            self.minCoord = self.Model.mesh.minCoord

        if self.maxCoord:
            self.maxCoord = tuple([nd(val) for val in self.maxCoord])
        else:
            self.maxCoord = self.Model.mesh.maxCoord

        if self.Model.mesh.dim == 2:
            self.minCoord = (self.minCoord[0], self.aspectRatio2d*self.minCoord[0])
            self.maxCoord = (self.maxCoord[0], self.aspectRatio2d*self.maxCoord[0])

        if rank == 0:
            self.badlands_model = self._create_badlands_model()
            self.badlands_model.load_xml(self.XML)

            if self.restartStep:
                # this will kick off internal restart code in Badlands that overwrites the _demfile
                # made below. See Badlands documentation for details.
                self.badlands_model.input.restart = True
                self.badlands_model.input.rstep = self.restartStep
                self.badlands_model.input.rfolder = self.restartFolder
                self.badlands_model.input.outDir = self.restartFolder
                self.badlands_model.outputStep = self.restartStep

                # Parse xmf for the last timestep time
                import xml.etree.ElementTree as etree
                xmf = (self.restartFolder +
                       "/xmf/tin.time" +
                       str(self.restartStep) + ".xmf")
                tree = etree.parse(xmf)
                root = tree.getroot()
                self.time_years = float(root[0][0][0].attrib["Value"])

            # Create Initial DEM
            self._demfile = _tempdir + "/dem.csv"
            self.dem = self._generate_dem()
            np.savetxt(self._demfile, self.dem)

            # Build Mesh
            self.badlands_model._build_mesh(self._demfile, verbose=False)

            self.badlands_model.input.outDir = self.outputDir
            self.badlands_model.input.disp3d = True  # enable 3D displacements
            self.badlands_model.input.region = 0  # TODO: check what this does
            self.badlands_model.input.tStart = self.time_years
            self.badlands_model.tNow = self.time_years

            # Override the checkpoint/display interval in the Badlands xml file.
            self.badlands_model.input.tDisplay = (
                dimensionalise(self.checkpoint_interval, u.years).magnitude)

            # Set Badlands minimal distance between nodes before regridding
            self.badlands_model.force.merge3d = (
                self.badlands_model.input.Afactor *
                self.badlands_model.recGrid.resEdges * 0.5)

            # Bodge Badlands to perform an initial checkpoint
            # FIXME: we need to run the model for at least one
            # iteration before this is generated.
            # It would be nice if this wasn't the case.
            self.badlands_model.force.next_display = 0

        comm.Barrier()

        self._disp_inserted = False

        # a cache for badlands information that can be reused in UW coupling.
        # The surface grid information from badlands is used to:
        #   1. designate material type, _update_material_types.
        #   2. used for sampling the velocity field (via help_gen_sample_point).
        # store this in bdl_cache
        self._bdl_cache = None

        # Transfer the initial DEM state to Underworld
        self._update_material_types()
        comm.Barrier()

    def _generate_dem(self):
        """
        Generate a badlands DEM. This can be used as the initial Badlands state.

        """

        # Calculate number of nodes from required resolution.
        nx = np.int32((self.maxCoord[0] - self.minCoord[0]) / self.resolution)
        ny = np.int32((self.maxCoord[1] - self.minCoord[1]) / self.resolution)
        nx += 1
        ny += 1

        x = np.linspace(self.minCoord[0], self.maxCoord[0], nx)
        y = np.linspace(self.minCoord[1], self.maxCoord[1], ny)

        coordsX, coordsY = np.meshgrid(x, y)

        dem = np.zeros((nx * ny, 3))
        dem[:, 0] = coordsX.flatten()
        dem[:, 1] = coordsY.flatten()

        coordsZ = self.surfElevation.evaluate(dem[:, :2])

        dem[:, 2] = coordsZ.flatten()
        return dimensionalise(dem, u.meter).magnitude

    # ------------------------------------------------------------ 采样点 --
    def help_gen_sample_point(self, uw_sample_style=0):
        '''
        Collective routine. Return sample points.

        Return:
        -------
            np.array : shape (dim)
            Locations to sample UW velocity.

        uw_sample_style : int, 0 default
            Style for UW velocity sampling. Possible values 0 or 1.
            0 - Use dynamic elevation from Badlands' model elevation.
                Interpolated to grid locations to sample the UW velocity.
            1 - Use OLD method to sample at Badlands' initial recGrid elevations.

        Take locations in Badlands' recGrid find the UW velocity field.
        This helper function abstracts the different algorithm, based on dim.
        This uses cached data, _bdl_cache, from routine _update_material_types().
        '''

        if uw_sample_style:
            # The previous LIMITED implementation.
            # LIMITED as it use the static recGrid data from Badlands to sample the UW velocity.
            np_surface = None
            if rank == 0:
                rg = self.badlands_model.recGrid
                if self.Model.mesh.dim == 2:
                    zVals = rg.regZ.mean(axis=1)
                    np_surface = np.column_stack((rg.regX, zVals))

                if self.Model.mesh.dim == 3:
                    np_surface = np.column_stack((rg.rectX, rg.rectY, rg.rectZ))

            np_surface = comm.bcast(np_surface, root=0)
            comm.Barrier()

            return nd(np_surface * u.meters)  # non dimensionalise

        ## SHOULD only get here is uw_sample_style == 0

        fact = dimensionalise(1.0, u.meter).magnitude
        if self.Model.mesh.dim == 2:
            xs = None
            ys = None
            if rank == 0:
                xs = self.badlands_model.recGrid.regX / fact  # 1D simple
                ys = self.badlands_model.recGrid.regY / fact

            (known_xy, known_z) = (self._bdl_cache[0], self._bdl_cache[1])

            xs = comm.bcast(xs, root=0)
            ys = comm.bcast(ys, root=0)

            comm.Barrier()

            grid_x, grid_y = np.meshgrid(xs, ys)
            interpolate_z = griddata(points=known_xy,
                                     values=known_z,
                                     xi=(grid_x, grid_y),
                                     method='nearest').T
            interpolate_z = interpolate_z.mean(axis=1)
            return np.column_stack((xs, interpolate_z))

        if self.Model.mesh.dim == 3:
            rect_x = None
            rect_y = None
            if rank == 0:
                rect_x = self.badlands_model.recGrid.rectX / fact  # 1D, every point
                rect_y = self.badlands_model.recGrid.rectY / fact

            (known_xy, known_z) = (self._bdl_cache[0], self._bdl_cache[1])
            rect_x = comm.bcast(rect_x, root=0)
            rect_y = comm.bcast(rect_y, root=0)

            comm.Barrier()
            interpolate_z = griddata(points=known_xy,
                                     values=known_z,
                                     xi=(rect_x, rect_y),
                                     method='nearest')
            return np.column_stack((rect_x, rect_y, interpolate_z))

    # --------------------------------------------------------------- 步进 --
    def solve(self, dt, sigma=0, uw_sample_style=0):
        """
        Collective routine

        Execute Badlands a badlands solve in the Underworld coupling.

        Parameters
        ----------
        dt : float,
            Non dimensional time to advance badlands forward.
        sigma : float, 0 default
            Apply a gaussian_filter, as per scipy.ndimage.filters, to the
            injected velocity displacements from UW to badlands
        uw_sample_style : int, 0 default
            Style for UW velocity sampling. Possible values 0 or 1.
            0 - Use dynamic elevation from Badlands' model elevation.
                Interpolated to grid locations to sample the UW velocity.
            1 - Use OLD method to sample at Badlands' initial recGrid elevations.

        Information of function.
        Execute Badlands a badlands solve in the Underworld coupling.
            1. Collect Badland's recGrid and broadcast to all procs.
            2. Inerpolate Underworld velocity field on recGrid
            3. Calculate overall displacement in meters by muliplying velocity (m/yr) with input dt (yr).
            4. Using the displacement run badlands from currect time `t` to time `t+dt`.
            5. TODO: Ensure final stratigraphic field from badlands at `t+dt`.
            6. Update Underworld particles depending on Badland tin. Another interpolation.
        """

        dt_years = np.round(dimensionalise(dt, u.years).magnitude, 6)  # fix pint scaling issue

        if rank == 0 and self.verbose:
            purple = "\033[0;35m"
            endcol = "\033[00m"
            msg = (f"Processing surface with Badlands {dt_years}:\n\t"
                   f" from {self.time_years} -> {self.time_years+dt_years}")
            print(purple + msg + endcol)
            sys.stdout.flush()

        # call the helper function to generate sample point for velocity evaluation
        nd_coords = self.help_gen_sample_point(uw_sample_style)

        tracer_velocity = self.Model.velocityField.evaluate_global(nd_coords)

        if rank == 0:
            tracer_disp = dimensionalise(tracer_velocity * dt, u.meter).magnitude

            self._inject_badlands_displacement(self.time_years,
                                               dt_years,        # in years
                                               tracer_disp,     # displacement in m/y
                                               sigma)           # controls gaussian filter smoothing

            # get badlands
            bdm = self.badlands_model

            # force badlands checkpoint to align with UW
            #bdm.force.tDisplay = dt_years # this or the following
            run_until = self.time_years + dt_years
            bdm.force.next_display = run_until

            # Run the Badlands model to the same time point
            bdm.run_to_time(run_until)

        self.time_years += dt_years

        # TODO: Improve the performance of this function
        surf_fn_badlands = self._update_material_types()
        comm.Barrier()

        if rank == 0 and self.verbose:
            purple = "\033[0;35m"
            endcol = "\033[00m"
            print(purple + "Processing surface with Badlands...Done" + endcol)
            sys.stdout.flush()

        return surf_fn_badlands if self.Model._freeSurface_ALEIB else None

    # ------------------------------------------------------- 表面读回判定 --
    def _determine_particle_state_2D(self):

        known_xy = None
        known_z = None
        xs = None
        ys = None
        fact = dimensionalise(1.0, u.meter).magnitude
        if rank == 0:
            # points that we have known elevation for
            known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact
            # elevation for those points
            known_z = self.badlands_model.elevation / fact
            xs = self.badlands_model.recGrid.regX / fact
            ys = self.badlands_model.recGrid.regY / fact

        known_xy = comm.bcast(known_xy, root=0)
        known_z = comm.bcast(known_z, root=0)
        xs = comm.bcast(xs, root=0)
        ys = comm.bcast(ys, root=0)

        # all procs
        self._bdl_cache = (known_xy, known_z)

        comm.Barrier()

        grid_x, grid_y = np.meshgrid(xs, ys)
        interpolate_z = griddata(known_xy,
                                 known_z,
                                 (grid_x, grid_y),
                                 method='nearest').T
        interpolate_z = interpolate_z.mean(axis=1)

        f = interp1d(xs, interpolate_z)

        uw_surface = self.Model.swarm.particleCoordinates.data
        bdl_surface = f(uw_surface[:, 0])

        flags = uw_surface[:, 1] < bdl_surface

        return flags, f

    def _determine_particle_state(self):
        # Given Badlands' mesh, determine if each particle in 'volume' is above
        # (False) or below (True) it.

        # To do this, for each X/Y pair in 'volume', we interpolate its Z value
        # relative to the mesh in blModel. Then, if the interpolated Z is
        # greater than the supplied Z (i.e. Badlands mesh is above particle
        # elevation) it's sediment (True). Else, it's air (False).

        # TODO: we only support air/sediment layers right now; erodibility
        # layers are not implemented

        known_xy = None
        known_z = None
        fact = dimensionalise(1.0, u.meter).magnitude
        if rank == 0:
            # points that we have known elevation for
            known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact
            known_z = self.badlands_model.elevation / fact

        known_xy = comm.bcast(known_xy, root=0)
        known_z = comm.bcast(known_z, root=0)

        # all procs
        self._bdl_cache = (known_xy, known_z)

        comm.Barrier()

        volume = self.Model.swarm.particleCoordinates.data

        interpolate_xy = volume[:, [0, 1]]

        # NOTE: we're using nearest neighbour interpolation. This should be
        # sufficient as Badlands will normally run at a much higher resolution
        # than Underworld. 'linear' interpolation is much, much slower.
        interpolate_z = griddata(points=known_xy,
                                 values=known_z,
                                 xi=interpolate_xy,
                                 method='nearest')

        # True for sediment, False for air
        flags = volume[:, 2] < interpolate_z
        f = CloughTocher2DInterpolator((known_xy[:, 0], known_xy[:, 1]), known_z)

        return flags, f

    def _update_material_types(self):
        # What do the materials (in air/sediment terms) look like now?
        if self.Model.mesh.dim == 3:
            under_bd_surface, surf_fn_badlands = self._determine_particle_state()
        if self.Model.mesh.dim == 2:
            under_bd_surface, surf_fn_badlands = self._determine_particle_state_2D()

        # If any materials changed state, update the Underworld material types
        mi = self.Model.materialField.data

        # convert air to sediment
        for air_material in self.airIndex:
            # if material air, and we're below surface, make it sediment
            sedimented_mask = np.logical_and(np.in1d(mi, air_material), under_bd_surface)
            mi[sedimented_mask] = self.sedimentIndex

        # convert sediment to air
        for air_material in self.airIndex:
            # if material is not air, and above surface, make it air
            eroded_mask = np.logical_and(~np.in1d(mi, air_material), ~under_bd_surface)
            mi[eroded_mask] = self.airIndex[0]
        return surf_fn_badlands

    # --------------------------------------------------------- 位移注入 --
    def _inject_badlands_displacement(self, time, dt, disp, sigma):
        """
        Takes a plane of tracer points and their DISPLACEMENTS in 3D over time
        period dt applies a gaussian filter on it. Injects it into Badlands as 3D
        tectonic movement.
        """

        # The Badlands 3D interpolation map is the displacement of each DEM
        # node at the end of the time period relative to its starting position.
        # If you start a new displacement file, it is treated as starting at the
        # DEM starting points (and interpolated onto the TIN as it was at
        # that tNow).

        # kludge; don't keep adding new entries
        if self._disp_inserted:
            self.badlands_model.force.T_disp[0, 0] = time
            self.badlands_model.force.T_disp[0, 1] = (time + dt)
        else:
            self.badlands_model.force.T_disp = np.vstack(([time, time + dt], self.badlands_model.force.T_disp))
            self._disp_inserted = True

        # Extent the velocity field in the third dimension
        if self.Model.mesh.dim == 2:
            dispX = np.tile(disp[:, 0], self.badlands_model.recGrid.rny)
            dispY = np.zeros((self.badlands_model.recGrid.rnx * self.badlands_model.recGrid.rny,))
            dispZ = np.tile(disp[:, 1], self.badlands_model.recGrid.rny)

            disp = np.zeros((self.badlands_model.recGrid.rnx * self.badlands_model.recGrid.rny, 3))
            disp[:, 0] = dispX
            disp[:, 1] = dispY
            disp[:, 2] = dispZ

        # Gaussian smoothing
        if sigma > 0:
            dispX = np.copy(disp[:, 0]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            dispY = np.copy(disp[:, 1]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            dispZ = np.copy(disp[:, 2]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            smoothX = gaussian_filter(dispX, sigma)
            smoothY = gaussian_filter(dispY, sigma)
            smoothZ = gaussian_filter(dispZ, sigma)
            disp[:, 0] = smoothX.flatten()
            disp[:, 1] = smoothY.flatten()
            disp[:, 2] = smoothZ.flatten()

        self.badlands_model.force.injected_disps = disp
