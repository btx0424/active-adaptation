import numpy as np
import trimesh
import random

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from .regular import MeshPlaneTerrainCfg


def hussar_3d_terrain_ver2(
    difficulty: float, cfg: "Hussar3DTerrainVer2Cfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a 3D terrain in project hussar.
    """
    difficulty = 0.0
    has_tree = cfg.has_tree
    has_ceil = cfg.has_ceil
    has_maze = cfg.has_maze

    ceil_min_height = cfg.ceil_min_height
    ceil_max_height = cfg.ceil_max_height
    min_gap = cfg.min_gap
    max_gap = cfg.max_gap
    
    # 计算地形位置
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.3)
    size_x, size_y = cfg.size
    
    x0 = [size_x, size_y, 0]
    x1 = [size_x, 0.0, 0]
    x2 = [0.0, size_y, 0]
    x3 = [0.0, 0.0, 0]
    # generate the tri-mesh with two triangles
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combined_meshes = [plane_mesh]
    
    # difficulty ∈ [0, 1]
    if has_ceil:
        # 根据难度计算天花板高度
        ceiling_height = ceil_max_height - difficulty * (ceil_max_height - ceil_min_height)
        
        # 创建天花板障碍物
        num_ceiling_parts = int(3 + difficulty * 2)  # 随难度增加天花板数量
        
        for _ in range(num_ceiling_parts):
            # 随机确定天花板大小
            width = random.uniform(1.0, 2.0)
            depth = random.uniform(1.0, 2.0)
            height = random.uniform(0.1, 0.3)
            
            # 随机位置，确保在场景范围内
            x = random.uniform(-size_x/2 + width/2, size_x/2 - width/2)
            y = random.uniform(-size_y/2 + depth/2, size_y/2 - depth/2)
            
            # 高度在计算的范围内随机浮动一点
            z = ceiling_height + random.uniform(0.0, 0.5) + height/2
            
            # 创建天花板块
            ceiling_part = trimesh.creation.box(
                extents=[width, depth, height]
            )
            
            # 应用平移
            translation = trimesh.transformations.translation_matrix([x, y, z])
            ceiling_part.apply_transform(translation)
            
            combined_meshes.append(ceiling_part)
    
    if has_tree:
        # 根据难度计算树干之间的间隙
        current_gap = max_gap - difficulty * (max_gap - min_gap)
        
        # 计算树干的数量和分布
        tree_density = 0.2 + difficulty * 0.4  # 随难度增加树的密度
        area = size_x * size_y
        num_trees = int(area * tree_density / (current_gap * current_gap))
        
        # 限制最大树数量，避免过度密集
        num_trees = min(num_trees, 200)
        
        # 跟踪已放置的树位置
        tree_positions = []
        
        for _ in range(num_trees):
            # 生成候选位置
            attempt = 0
            valid_position = False
            
            while attempt < 50 and not valid_position:  # 最多尝试50次找位置
                x = random.uniform(-size_x/2 + 0.5, size_x/2 - 0.5)
                y = random.uniform(-size_y/2 + 0.5, size_y/2 - 0.5)
                
                # 检查与其他树的距离
                valid_position = True
                for pos in tree_positions:
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < current_gap:
                        valid_position = False
                        break
                        
                attempt += 1
            
            if valid_position:
                tree_positions.append((x, y))
                
                # 树干参数
                trunk_height = random.uniform(1.5, 4.0)
                trunk_radius = random.uniform(0.1, 0.3)
                
                # 创建树干
                trunk = trimesh.creation.cylinder(
                    radius=trunk_radius,
                    height=trunk_height,
                    sections=8
                )
                
                # 随机倾斜角度 (0-20度)
                tilt_angle = random.uniform(0, 20) * (np.pi / 180)
                tilt_direction = random.uniform(0, 2 * np.pi)
                
                # 计算倾斜轴和矩阵
                tilt_axis = [np.cos(tilt_direction), np.sin(tilt_direction), 0]
                tilt_matrix = trimesh.transformations.rotation_matrix(tilt_angle, tilt_axis)
                
                # 应用变换
                bottom_to_origin = trimesh.transformations.translation_matrix([0, 0, trunk_height/2])
                translation = trimesh.transformations.translation_matrix([x, y, 0])
                
                trunk.apply_transform(bottom_to_origin)
                trunk.apply_transform(tilt_matrix)
                trunk.apply_transform(translation)
                
                combined_meshes.append(trunk)
    
    if has_maze:
        # 根据难度调整迷宫通道宽度
        passage_width = max_gap - difficulty * (max_gap - min_gap)
        
        # 墙体参数
        wall_height = 2.0
        wall_thickness = 0.3
        
        # 创建随机墙体迷宫，而不是使用DFS算法
        # 生成水平和垂直的墙体，确保它们之间有足够的通道宽度
        
        # 确定墙体数量，随难度增加
        num_walls = int(5 + difficulty * 15)
        
        # 已放置墙体的位置记录
        wall_positions = []
        
        # 墙体的最小和最大长度
        min_wall_length = 2.0
        max_wall_length = min(size_x, size_y) * 0.4
        
        # 生成墙体
        for _ in range(num_walls):
            # 随机决定墙体方向 (水平或垂直)
            is_horizontal = random.choice([True, False])
            
            # 随机墙体长度
            wall_length = random.uniform(min_wall_length, max_wall_length)
            
            # 决定是否在墙体上挖洞（70%的概率挖洞）
            create_hole = random.random() < 0.7
            
            # 墙体尺寸
            if is_horizontal:
                wall_size = [wall_length, wall_thickness, wall_height]
            else:
                wall_size = [wall_thickness, wall_length, wall_height]
            
            # 尝试放置墙体
            max_attempts = 50
            for attempt in range(max_attempts):
                # 随机位置
                x = random.uniform(-size_x/2 + wall_length/2, size_x/2 - wall_length/2) if is_horizontal else random.uniform(-size_x/2 + wall_thickness/2, size_x/2 - wall_thickness/2)
                y = random.uniform(-size_y/2 + wall_thickness/2, size_y/2 - wall_thickness/2) if is_horizontal else random.uniform(-size_y/2 + wall_length/2, size_y/2 - wall_length/2)
                
                # 检查是否与已有墙体太近
                too_close = False
                for wall_pos, wall_is_horizontal, wall_len in wall_positions:
                    # 如果方向相同且平行很近
                    if is_horizontal == wall_is_horizontal:
                        if is_horizontal:
                            # 水平墙体之间的垂直距离
                            if abs(y - wall_pos[1]) < passage_width:
                                # 检查水平重叠
                                min_x = max(x - wall_length/2, wall_pos[0] - wall_len/2)
                                max_x = min(x + wall_length/2, wall_pos[0] + wall_len/2)
                                if max_x > min_x:  # 存在重叠
                                    too_close = True
                                    break
                        else:
                            # 垂直墙体之间的水平距离
                            if abs(x - wall_pos[0]) < passage_width:
                                # 检查垂直重叠
                                min_y = max(y - wall_length/2, wall_pos[1] - wall_len/2)
                                max_y = min(y + wall_length/2, wall_pos[1] + wall_len/2)
                                if max_y > min_y:  # 存在重叠
                                    too_close = True
                                    break
                    # 如果方向不同，检查交叉点附近是否有足够空间
                    else:
                        if is_horizontal:
                            # 水平墙与垂直墙
                            h_x, h_y = x, y
                            v_x, v_y = wall_pos
                            h_len = wall_length
                            v_len = wall_len
                        else:
                            # 垂直墙与水平墙
                            v_x, v_y = x, y
                            h_x, h_y = wall_pos
                            v_len = wall_length
                            h_len = wall_len
                        
                        # 检查是否在彼此范围内
                        if (abs(h_y - v_y) < passage_width and 
                            abs(v_x - h_x) < passage_width and
                            v_x - passage_width < h_x + h_len/2 and 
                            v_x + passage_width > h_x - h_len/2 and
                            h_y - passage_width < v_y + v_len/2 and
                            h_y + passage_width > v_y - v_len/2):
                            too_close = True
                            break
                
                if not too_close:
                    # 找到有效位置，创建墙体
                    
                    # 决定是否挖洞及洞的位置
                    if create_hole and wall_length > passage_width * 2 + 1.0:
                        # 确定洞的位置（避免太靠近墙体边缘）
                        hole_margin = 0.5  # 洞距离墙体边缘的最小距离
                        hole_position = random.uniform(hole_margin, wall_length - passage_width - hole_margin)
                        
                        # 计算墙体左右/上下两段
                        if is_horizontal:
                            # 水平墙体，左段
                            left_length = hole_position
                            left_x = x - wall_length/2 + left_length/2
                            left_y = y
                            left_size = [left_length, wall_thickness, wall_height]
                            
                            # 水平墙体，右段
                            right_length = wall_length - hole_position - passage_width
                            right_x = x + wall_length/2 - right_length/2
                            right_y = y
                            right_size = [right_length, wall_thickness, wall_height]
                            
                            # 创建左段墙体
                            if left_length > 0.2:  # 长度足够才创建
                                left_wall = trimesh.creation.box(extents=left_size)
                                translation = trimesh.transformations.translation_matrix([left_x, left_y, wall_height/2])
                                left_wall.apply_transform(translation)
                                combined_meshes.append(left_wall)
                            
                            # 创建右段墙体
                            if right_length > 0.2:  # 长度足够才创建
                                right_wall = trimesh.creation.box(extents=right_size)
                                translation = trimesh.transformations.translation_matrix([right_x, right_y, wall_height/2])
                                right_wall.apply_transform(translation)
                                combined_meshes.append(right_wall)
                        else:
                            # 垂直墙体，下段
                            bottom_length = hole_position
                            bottom_x = x
                            bottom_y = y - wall_length/2 + bottom_length/2
                            bottom_size = [wall_thickness, bottom_length, wall_height]
                            
                            # 垂直墙体，上段
                            top_length = wall_length - hole_position - passage_width
                            top_x = x
                            top_y = y + wall_length/2 - top_length/2
                            top_size = [wall_thickness, top_length, wall_height]
                            
                            # 创建下段墙体
                            if bottom_length > 0.2:  # 长度足够才创建
                                bottom_wall = trimesh.creation.box(extents=bottom_size)
                                translation = trimesh.transformations.translation_matrix([bottom_x, bottom_y, wall_height/2])
                                bottom_wall.apply_transform(translation)
                                combined_meshes.append(bottom_wall)
                            
                            # 创建上段墙体
                            if top_length > 0.2:  # 长度足够才创建
                                top_wall = trimesh.creation.box(extents=top_size)
                                translation = trimesh.transformations.translation_matrix([top_x, top_y, wall_height/2])
                                top_wall.apply_transform(translation)
                                combined_meshes.append(top_wall)
                    else:
                        # 创建完整墙体（无洞）
                        wall = trimesh.creation.box(extents=wall_size)
                        translation = trimesh.transformations.translation_matrix([x, y, wall_height/2])
                        wall.apply_transform(translation)
                        combined_meshes.append(wall)
                    
                    # 记录墙体位置
                    wall_positions.append(([x, y], is_horizontal, wall_length))
                    break
    
    # 合并所有网格
    if len(combined_meshes) > 1:
        combined_mesh = trimesh.util.concatenate(combined_meshes)
    else:
        combined_mesh = combined_meshes[0]
    
    return [combined_mesh], origin
    


@configclass
class Hussar3DTerrainVer2Cfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = hussar_3d_terrain_ver2

    """whether to add a stair plane to the terrain"""
    has_tree: bool = False
    has_ceil: bool = True
    has_maze: bool = True

    ceil_min_height: float = 0.7
    ceil_max_height: float = 1.4
    min_gap: float = 0.5
    max_gap: float = 2.0


NEW_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.5,
        ),
        "hussar_3d": Hussar3DTerrainVer2Cfg(
            proportion  = 0.5,
            has_tree    = False,
            has_ceil    = False,
            has_maze    = True,
            ceil_min_height = 0.7,
            ceil_max_height = 1.4,
            min_gap = 0.5,
            max_gap = 2.0
        ),
    },
)


ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=None,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)

TERRAINS = {
    "hussar_3d": ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=NEW_ROUGH_TERRAINS_CFG),
}