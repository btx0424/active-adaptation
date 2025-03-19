from typing import List, Union, Callable
import numpy as np
import torch
import omni.usd
import re
import functools

from omni.physx import get_physx_replicator_interface, get_physx_simulation_interface
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdUtils, Vt, Semantics

from isaacsim.core.cloner import Cloner
from isaaclab.sim import schemas, SpawnerCfg, find_matching_prim_paths

class MyCloner(Cloner):

    def clone(
        self,
        source_prim_path: str,
        prim_paths: List[str],
        positions: Union[np.ndarray, torch.Tensor] = None,
        orientations: Union[np.ndarray, torch.Tensor] = None,
        scales: Union[np.ndarray, torch.Tensor] = None,
        replicate_physics: bool = False,
        base_env_path: str = None,
        root_path: str = None,
        copy_from_source: bool = False,
    ):

        """Clones a source prim at user-specified destination paths.
            Clones will be placed at user-specified positions and orientations.

        Args:
            source_prim_path (str): Path of source object.
            prim_paths (List[str]): List of destination paths.
            positions (Union[np.ndarray, torch.Tensor]): An array containing target positions of clones. Dimension must equal length of prim_paths.
                                    Defaults to None. Clones will be placed at (0, 0, 0) if not specified.
            orientations (Union[np.ndarray, torch.Tensor]): An array containing target orientations of clones. Dimension must equal length of prim_paths.
                                    Defaults to None. Clones will have identity orientation (1, 0, 0, 0) if not specified.
            replicate_physics (bool): Uses omni.physics replication. This will replicate physics properties directly for paths beginning with root_path and skip physics parsing for anything under the base_env_path.
            base_env_path (str): Path to namespace for all environments. Required if replicate_physics=True and define_base_env() not called.
            root_path (str): Prefix path for each environment. Required if replicate_physics=True and generate_paths() not called.
            copy_from_source: (bool): Setting this to False will inherit all clones from the source prim; any changes made to the source prim will be reflected in the clones.
                         Setting this to True will make copies of the source prim when creating new clones; changes to the source prim will not be reflected in clones. Defaults to False. Note that setting this to True will take longer to execute.
        Raises:
            Exception: Raises exception if source prim path is not valid.

        """
        assert not replicate_physics

        # check if inputs are valid
        if positions is not None:
            if len(positions) != len(prim_paths):
                raise ValueError("Dimension mismatch between positions and prim_paths!")
            # convert to numpy array
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            elif not isinstance(positions, np.ndarray):
                positions = np.asarray(positions)
            # convert to pxr gf
            positions = Vt.Vec3fArray.FromNumpy(positions)
        if orientations is not None:
            if len(orientations) != len(prim_paths):
                raise ValueError("Dimension mismatch between orientations and prim_paths!")
            # convert to numpy array
            if isinstance(orientations, torch.Tensor):
                orientations = orientations.detach().cpu().numpy()
            elif not isinstance(orientations, np.ndarray):
                orientations = np.asarray(orientations)
            # convert to pxr gf -- wxyz to xyzw
            orientations = np.roll(orientations, -1, -1)
            orientations = Vt.QuatdArray.FromNumpy(orientations)
        if scales is not None:
            if len(scales) != len(prim_paths):
                raise ValueError("Dimension mismatch between scales and prim_paths!")
            if isinstance(scales, torch.Tensor):
                scales = scales.detach().cpu().numpy()
            elif not isinstance(scales, np.ndarray):
                scales = np.asarray(scales)
            scales = Vt.Vec3fArray.FromNumpy(scales)

        # make sure source prim has valid xform properties
        stage = omni.usd.get_context().get_stage()
        source_prim = stage.GetPrimAtPath(source_prim_path)
        if not source_prim:
            raise Exception("Source prim does not exist")
        properties = source_prim.GetPropertyNames()
        xformable = UsdGeom.Xformable(source_prim)
        # get current position and orientation
        T_p_w = xformable.ComputeParentToWorldTransform(Usd.TimeCode.Default())
        T_l_w = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        T_l_p = Gf.Transform()
        T_l_p.SetMatrix(Gf.Matrix4d(np.matmul(T_l_w, np.linalg.inv(T_p_w)).tolist()))
        current_translation = T_l_p.GetTranslation()
        current_orientation = T_l_p.GetRotation().GetQuat()
        # get current scale
        current_scale = Gf.Vec3d(1, 1, 1)
        if "xformOp:scale" in properties:
            current_scale = Gf.Vec3d(source_prim.GetAttribute("xformOp:scale").Get())

        # remove all xform ops except for translate, orient, and scale
        properties_to_remove = [
            "xformOp:rotateX",
            "xformOp:rotateXZY",
            "xformOp:rotateY",
            "xformOp:rotateYXZ",
            "xformOp:rotateYZX",
            "xformOp:rotateZ",
            "xformOp:rotateZYX",
            "xformOp:rotateZXY",
            "xformOp:rotateXYZ",
            "xformOp:transform",
            "xformOp:scale",
        ]
        xformable.ClearXformOpOrder()
        for prop_name in properties:
            if prop_name in properties_to_remove:
                source_prim.RemoveProperty(prop_name)

        properties = source_prim.GetPropertyNames()
        # add xform ops if they don't exist
        if "xformOp:translate" not in properties:
            xform_op_translate = xformable.AddXformOp(
                UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
            )
        else:
            xform_op_translate = UsdGeom.XformOp(source_prim.GetAttribute("xformOp:translate"))
        xform_op_translate.Set(current_translation)

        if "xformOp:orient" not in properties:
            xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
        else:
            xform_op_rot = UsdGeom.XformOp(source_prim.GetAttribute("xformOp:orient"))
        xform_op_rot.Set(current_orientation)

        if "xformOp:scale" not in properties:
            xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
        else:
            xform_op_scale = UsdGeom.XformOp(source_prim.GetAttribute("xformOp:scale"))
        xform_op_scale.Set(current_scale)
        # set xform op order
        xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

        # set source actor transform
        if source_prim_path in prim_paths:
            idx = prim_paths.index(source_prim_path)
            prim = UsdGeom.Xform(stage.GetPrimAtPath(source_prim_path))

            if positions is not None:
                translation = positions[idx]
            else:
                translation = current_translation

            if orientations is not None:
                orientation = orientations[idx]
            else:
                orientation = current_orientation

            # overwrite translation and orientation to values specified
            prim.GetPrim().GetAttribute("xformOp:translate").Set(translation)
            prim.GetPrim().GetAttribute("xformOp:orient").Set(orientation)

        has_clones = False
        with Sdf.ChangeBlock():
            for i, prim_path in enumerate(prim_paths):
                if prim_path != source_prim_path:
                    has_clones = True
                    env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                    stack = UsdGeom.Xform(stage.GetPrimAtPath(source_prim_path)).GetPrim().GetPrimStack()

                    if copy_from_source:
                        Sdf.CopySpec(env_spec.layer, Sdf.Path(source_prim_path), env_spec.layer, Sdf.Path(prim_path))
                    else:
                        env_spec.inheritPathList.Prepend(source_prim_path)

                    if positions is not None:
                        translation = positions[i]  # use specified translation
                    else:
                        translation = current_translation  # use the same translation as source

                    if orientations is not None:
                        orientation = orientations[i]  # use specified orientation
                    else:
                        orientation = current_orientation  # use the same orientation as source
                    
                    if scales is not None:
                        scale = scales[i]
                    else:
                        scale = current_scale

                    translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
                    if translate_spec is None:
                        translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                    translate_spec.default = translation

                    orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
                    if orient_spec is None:
                        orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                    orient_spec.default = orientation

                    scale_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                    if scale_spec is None:
                        scale_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
                    scale_spec.default = scale

                    op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

        if replicate_physics and has_clones:
            self.replicate_physics(source_prim_path, prim_paths, base_env_path, root_path)
        else:
            get_physx_replicator_interface().unregister_replicator(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())

def clone(func: Callable) -> Callable:
    """Decorator for cloning a prim based on matching prim paths of the prim's parent.

    The decorator checks if the parent prim path matches any prim paths in the stage. If so, it clones the
    spawned prim at each matching prim path. For example, if the input prim path is: ``/World/Table_[0-9]/Bottle``,
    the decorator will clone the prim at each matching prim path of the parent prim: ``/World/Table_0/Bottle``,
    ``/World/Table_1/Bottle``, etc.

    Note:
        For matching prim paths, the decorator assumes that valid prims exist for all matching prim paths.
        In case no matching prim paths are found, the decorator raises a ``RuntimeError``.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function that spawns the prim and clones it at each matching prim path.
        It returns the spawned source prim, i.e., the first prim in the list of matching prim paths.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str | Sdf.Path, cfg: SpawnerCfg, *args, **kwargs):
        # cast prim_path to str type in case its an Sdf.Path
        prim_path = str(prim_path)
        # check prim path is global
        if not prim_path.startswith("/"):
            raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
        # resolve: {SPAWN_NS}/AssetName
        # note: this assumes that the spawn namespace already exists in the stage
        root_path, asset_path = prim_path.rsplit("/", 1)
        # check if input is a regex expression
        # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
        is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

        # resolve matching prims for source prim path expression
        if is_regex_expression and root_path != "":
            source_prim_paths = find_matching_prim_paths(root_path)
            # if no matching prims are found, raise an error
            if len(source_prim_paths) == 0:
                raise RuntimeError(
                    f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
                )
        else:
            source_prim_paths = [root_path]

        # resolve prim paths for spawning and cloning
        prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
        # spawn single instance
        prim = func(prim_paths[0], cfg, *args, **kwargs)
        # set the prim visibility
        if hasattr(cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(cfg, "activate_contact_sensors") and cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_paths[0], cfg.activate_contact_sensors)
        # clone asset using cloner API
        if len(prim_paths) > 1:
            cloner = MyCloner()
            # clone the prim
            low, high = getattr(cfg, "scale_range", (1., 1.))
            scales = torch.ones(len(prim_paths), 1)
            scales[1:].uniform_(low, high)
            cfg.scale = scales
            cloner.clone(
                prim_paths[0], 
                prim_paths[1:],
                scales=scales[1:].expand(-1, 3),
                replicate_physics=False, 
                copy_from_source=cfg.copy_from_source
            )
        # return the source prim
        return prim

    return wrapper