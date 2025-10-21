import active_adaptation


if active_adaptation.get_backend() == "isaac":
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg
    import builtins
    import numpy as np

    def add_skin_by_tiles(scene, scene_cfg,
                        z_plane=0.0,              # 想贴合的全局物理高度
                        thickness=0.02,           # 薄板厚度
                        prim_root="/World/terrain_skins",
                        col_axis="x",             # 'x' 表示列沿 X 方向，'y' 表示列沿 Y
                        ):
        gen = scene_cfg.terrain.terrain_generator
        num_rows = int(getattr(gen, "num_rows"))
        num_cols = int(getattr(gen, "num_cols"))

        # 单块尺寸——以 pillar 的 size 为准（你这套是固定 8×8）
        tile_sx, tile_sy = gen.sub_terrains["hussar_pillar"].size
        half_h = thickness * 0.5

        # 1) 按 key 顺序 + proportion 划分列范围
        keys = list(gen.sub_terrains.keys())
        props = [float(gen.sub_terrains[k].proportion) for k in keys]
        s = sum(props) or 1.0
        props = [p / s for p in props]

        boundaries = [0]
        cum = 0.0
        for p in props[:-1]:
            cum += p
            boundaries.append(int(round(cum * num_cols)))
        boundaries.append(num_cols)
        key_cols = {k: (boundaries[i], boundaries[i+1]) for i, k in enumerate(keys)}
        c0, c1 = key_cols["hussar_pillar"]                 # [c0, c1) 是 pillar 的列段
        target_rows = range(0, 3)              # 前半行
        target_cols = range(c0, c1)

        # 2) 推断地形网格左下角 (x0, y0) 的“块左下角坐标”（不是中心）
            # 用当前 env 起点的最小 x,y 当作“左下角块中心”，退半块得到左下角角点
            # origins = np.asarray(scene.env_origins.cpu().detach(), dtype=float) if getattr(scene, "env_origins", None) is not None else np.zeros((1,3))
        x0 = -0.5 * num_rows * tile_sy#-0.75 * num_cols * tile_sx
        y0 = -0.5 * num_cols * tile_sx
            # if col_axis == "x":
            #     x0 = origins[:, 0].min() - 0.5 * tile_sx
            #     y0 = origins[:, 1].min() - 0.5 * tile_sy
            # else:
            #     # 列沿 Y 时，仍然用同样的“最小中心退半块”法
            #     x0 = origins[:, 0].min() - 0.5 * tile_sx
            #     y0 = origins[:, 1].min() - 0.5 * tile_sy


        # 3) 生成不可见可碰撞的薄板 cfg
        skin_cfg = CuboidCfg(
            size=(tile_sx, tile_sy, thickness),
            visible=False,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),   # 静态蒙皮
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True, rest_offset=0.0, contact_offset=0.01
            ),
        )

        # 4) 按 “列∈pillar段 且 行在前半” 的格子铺板（不看 num_envs）
        placed = 0
        for r in target_rows:
            for c in target_cols:
                # 块中心坐标
                if col_axis == "x":
                    cx = x0 + (c + 0.5) * tile_sx
                    cy = y0 + (r + 0.5) * tile_sy
                else:  # 列沿 Y
                    cx = x0 + (r + 0.5) * tile_sx
                    cy = y0 + (c + 0.5) * tile_sy
                cz = float(z_plane) + half_h

                prim_path = f"{prim_root}/pillar_skin_r{r:02d}_c{c:02d}"
                spawn_cuboid(prim_path=prim_path, cfg=skin_cfg, translation=(cx, cy, cz))
                placed += 1

        print(f"[skin-by-tiles] ✓ 覆盖 hussar_pillar 列 {c0}-{c1-1}，行 0-{(num_rows//2)-1}，共放置 {placed} 块。"
            f" 原点(x0,y0)=({x0:.3f},{y0:.3f})，z_plane={z_plane:.3f}, axis={col_axis}")
        
    def create_isaaclab_sim_and_scene(
        sim_cfg: SimulationCfg,
        scene_cfg: InteractiveSceneCfg
    ):
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            sim = SimulationContext(sim_cfg)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        scene = InteractiveScene(scene_cfg)
        add_skin_by_tiles(scene, scene_cfg, z_plane=-0.02, thickness=0.02, col_axis='y')
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            sim.reset()
        sim.step(render=sim.has_gui())
        return sim, scene

elif active_adaptation.get_backend() == "mujoco":
    pass
else:
    raise NotImplementedError


