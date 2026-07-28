"""dm_control 1.0.41 + mujoco 3.9.0: drop fields missing from MjModel (e.g. flex_bandwidth)."""


def apply_dm_control_mujoco_compat() -> None:
    try:
        import dm_control.mujoco.wrapper.mjbindings.sizes as sizes
    except ImportError:
        return

    mjmodel = sizes.array_sizes.get("mjmodel")
    if not mjmodel:
        return

    import mujoco

    probe = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='1 1 0.1'/></worldbody></mujoco>"
    )
    for field in list(mjmodel.keys()):
        if not hasattr(probe, field):
            del mjmodel[field]
