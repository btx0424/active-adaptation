# 保存为 show_voxel_timeseries.py 后运行：python show_voxel_timeseries.py path_to.npy
import sys, numpy as np, plotly.graph_objects as go

def make_voxel_timeseries_figure(vol4d, threshold=0.5):
    assert vol4d.ndim == 4 and vol4d.shape[1:] == (32, 32, 32)
    T, nx, ny, nz = vol4d.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    x = x.astype(np.float32).ravel()
    y = y.astype(np.float32).ravel()
    z = z.astype(np.float32).ravel()
    v0 = (vol4d[0] > threshold).astype(np.float32).ravel()
    colors = [[0.0, "rgba(241,86,66,1.0)"], [1.0, "rgba(241,86,66,1.0)"]]
    fig = go.Figure(
        data=[go.Volume(x=x, y=y, z=z, value=v0, isomin=0.0, isomax=1.0, surface_count=1, opacity=0.9,
                        caps=dict(x_show=False, y_show=False, z_show=False),colorscale=colors)],
        layout=go.Layout(
            title=f"Voxel t=0",
            scene=dict(aspectmode="cube"),
            updatemenus=[dict(type="buttons", showactive=False, buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate")]),
            ])],
            sliders=[dict(steps=[], currentvalue=dict(prefix="t = "))]
        ),
        frames=[]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )

    steps = []
    frames = []
    for k in range(T):
        vk = (vol4d[k] > threshold).astype(np.float32).ravel()
        frames.append(go.Frame(
            name=f"t{k}",
            data=[go.Volume(x=x, y=y, z=z, value=vk, isomin=0.0, isomax=1.0, surface_count=1, opacity=0.9,
                            caps=dict(x_show=False, y_show=False, z_show=False))],
            layout=go.Layout(title=f"Voxel t={k}")
        ))
        steps.append(dict(method="animate", args=[[f"t{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                          label=str(k)))

    fig.frames = frames
    fig.layout.sliders = [dict(steps=steps, currentvalue=dict(prefix="t = "))]

    return fig


def make_points_timeseries_figure(points_ts):
    """
    points_ts: 
      - ndarray with shape (T,) and dtype=object, each element (Ni,3), or
      - ndarray with shape (T, N, 3)
    """
    if isinstance(points_ts, np.ndarray) and points_ts.dtype == object:
        T = len(points_ts)
        p0 = points_ts[0]
    else:
        assert points_ts.ndim == 3 and points_ts.shape[-1] == 3
        T = points_ts.shape[0]
        p0 = points_ts[0]

    x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
    fig = go.Figure(
        data=[go.Scatter3d(x=x0, y=y0, z=z0, mode="markers",
                            marker=dict(size=3, color="rgba(30,144,255,1.0)"))],
        layout=go.Layout(
            title=f"Ray points t=0",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            updatemenus=[dict(type="buttons", showactive=False, buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate")]),
            ])],
            sliders=[dict(steps=[], currentvalue=dict(prefix="t = "))],
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
        ),
        frames=[]
    )

    steps = []
    frames = []
    for k in range(T):
        pk = points_ts[k] if (isinstance(points_ts, np.ndarray) and points_ts.dtype == object) else points_ts[k]
        if pk.size == 0:
            xk = yk = zk = []
        else:
            xk, yk, zk = pk[:, 0], pk[:, 1], pk[:, 2]
        frames.append(go.Frame(
            name=f"t{k}",
            data=[go.Scatter3d(x=xk, y=yk, z=zk, mode="markers",
                                marker=dict(size=3, color="rgba(30,144,255,1.0)"))],
            layout=go.Layout(title=f"Ray points t={k}")
        ))
        steps.append(dict(method="animate", args=[[f"t{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                          label=str(k)))

    fig.frames = frames
    fig.layout.sliders = [dict(steps=steps, currentvalue=dict(prefix="t = "))]
    return fig

# import numpy as np
# import plotly.graph_objects as go
# from skimage import measure

# def _binary_to_mesh3d(vol_bin):
#     """
#     vol_bin: (32,32,32) 的 {0,1} 数组
#     返回: Mesh3d 需要的 (verts_x, verts_y, verts_z, i, j, k)
#     """
#     # marching_cubes 在体素坐标系中：体素中心在整数点，面落在 0.5 处
#     verts, faces, normals, values = measure.marching_cubes(vol_bin.astype(np.float32), level=0.5)
#     # faces 是 (N,3) 的三角面索引
#     i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
#     x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
#     return x, y, z, i, j, k

# def make_voxel_timeseries_figure(vol4d, threshold=0.5):
#     assert vol4d.ndim == 4 and vol4d.shape[1:] == (32, 32, 32)
#     T = vol4d.shape[0]

#     # 首帧 mesh
#     vol0_bin = (vol4d[0] > threshold)
#     x, y, z, i, j, k = _binary_to_mesh3d(vol0_bin)

#     fig = go.Figure(
#         data=[go.Mesh3d(
#             x=x, y=y, z=z, i=i, j=j, k=k,
#             color="lightgray",        # 单色块面，更接近“方块”外观
#             flatshading=True,         # 平面着色，避免光滑过渡
#             opacity=1.0,
#             lighting=dict(ambient=0.5, diffuse=0.6, specular=0.1, roughness=1.0),
#             showscale=False,
#         )],
#         layout=go.Layout(
#             title=f"Voxel t=0",
#             scene=dict(
#                 aspectmode="cube",
#                 xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
#                 xaxis=dict(nticks=9, range=[-0.5, 32.5]),
#                 yaxis=dict(nticks=9, range=[-0.5, 32.5]),
#                 zaxis=dict(nticks=9, range=[-0.5, 32.5]),
#             ),
#             updatemenus=[dict(type="buttons", showactive=False, buttons=[
#                 dict(label="▶ Play", method="animate",
#                      args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
#                 dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate")]),
#             ])],
#             sliders=[dict(steps=[], currentvalue=dict(prefix="t = "))]
#         ),
#         frames=[]
#     )

#     steps = []
#     frames = []
#     for k_idx in range(T):
#         vol_bin = (vol4d[k_idx] > threshold)
#         xk, yk, zk, ik, jk, kk = _binary_to_mesh3d(vol_bin)
#         frames.append(go.Frame(
#             name=f"t{k_idx}",
#             data=[go.Mesh3d(
#                 x=xk, y=yk, z=zk, i=ik, j=jk, k=kk,
#                 color="lightgray",
#                 flatshading=True,
#                 opacity=1.0,
#                 lighting=dict(ambient=0.5, diffuse=0.6, specular=0.1, roughness=1.0),
#                 showscale=False,
#             )],
#             layout=go.Layout(title=f"Voxel t={k_idx}")
#         ))
#         steps.append(dict(
#             method="animate",
#             args=[[f"t{k_idx}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
#             label=str(k_idx)
#         ))

#     fig.frames = frames
#     fig.layout.sliders = [dict(steps=steps, currentvalue=dict(prefix="t = "))]
#     return fig

def main():
    if len(sys.argv) == 2:
        path = sys.argv[1]
        arr = np.load(path, allow_pickle=True)
    else:
        path = None
        T = 20
        arr = (np.random.rand(T, 32, 32, 32) > 0.7).astype(np.uint8)

    # 自动分流：体素 or 点云
    if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[1:] == (32, 32, 32):
        fig = make_voxel_timeseries_figure(arr, threshold=0.5)
        out = "voxel_timeseries.html"
    else:
        fig = make_points_timeseries_figure(arr)
        out = "ray_points_timeseries.html"

    fig.write_html(out, include_plotlyjs="cdn", auto_open=True)
    print(f"已输出 {out}")

if __name__ == "__main__":
    main()