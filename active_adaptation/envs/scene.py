import active_adaptation


if active_adaptation.get_backend() == "isaac":
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import builtins

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
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            sim.reset()
        sim.step(render=sim.has_gui())
        return sim, scene

elif active_adaptation.get_backend() == "mujoco":
    pass
else:
    raise NotImplementedError


