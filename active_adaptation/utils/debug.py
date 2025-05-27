import torch

class DebugDraw:
    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
    
    def clear(self):
        self._draw.clear_lines()
        self._draw.clear_points()

    def plot(self, x: torch.Tensor, size=2.0, color=(1., 1., 1., 1.)):
        if not (x.ndim == 2) and (x.shape[1] == 3):
            raise ValueError("x must be a tensor of shape (N, 3).")
        x = x.cpu()
        point_list_0 = x[:-1].tolist()
        point_list_1 = x[1:].tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
        
    def vector(self, x: torch.Tensor, v: torch.Tensor, size=2.0, color=(0., 1., 1., 1.)):
        x = x.cpu().reshape(-1, 3)
        v = v.cpu().reshape(-1, 3)
        if not (x.shape == v.shape):
            raise ValueError("x and v must have the same shape, got {} and {}.".format(x.shape, v.shape))
        point_list_0 = x.tolist()
        point_list_1 = (x + v).tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
    
    def point(self, x: torch.Tensor, color=(1., 0., 0., 1.), size=10.0):
        point_list = x.cpu().reshape(-1, 3).tolist()
        sizes = [size] * len(point_list)
        colors = [color] * len(point_list)
        self._draw.draw_points(point_list, colors, sizes)