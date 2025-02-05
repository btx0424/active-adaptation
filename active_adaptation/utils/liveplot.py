import zmq
import time
import numpy as np
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys


class LivePlotClient:
    def __init__(self, zmq_addr="tcp://localhost:5555", send_interval=0.01):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(zmq_addr)
        self.send_interval = send_interval

    def test(self):
        """
        示例：持续发送随机数列表。
        """
        while True:
            # 生成随机数据
            data = np.random.rand(3).tolist()  # 假设发送长度为3的浮点数列表
            self.socket.send_pyobj(data)
            time.sleep(self.send_interval)  # 控制发送速度

    def send(self, data):
        """
        对外提供的发送接口，可以自由调用。
        data: 可以是 list, dict 等可序列化对象
        """
        self.socket.send_pyobj(data)


class LivePlotServer(QtCore.QObject):
    data_received = QtCore.pyqtSignal(
        list
    )  # Signal to communicate with the main thread

    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
        self.win.resize(800, 600)  # 可选：调整窗口大小

        # plots[i] 表示第 i 个子图
        self.plots = []
        # curves[i] 表示第 i 个子图上的所有曲线 (可能 1 条或 2 条)
        self.curves = []
        # data[i] 表示第 i 个子图所有曲线的历史数据
        # data[i][0] 第 i 个子图中曲线 1 的数据
        # data[i][1] 第 i 个子图中曲线 2 的数据 (如果有的话)
        self.data = []

        # Set up ZeroMQ and the thread to receive data
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:5555")
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        # Connect the data_received signal to the update_plots slot
        self.data_received.connect(self.update_plots)

        # Set up a timer to regularly update the plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50 milliseconds

    def receive_data(self):
        """Receive data in a separate thread and emit it to the main thread."""
        while True:
            data = self.socket.recv_pyobj()  # Receive data from ZeroMQ
            # data 形如 [x1, x2, ..., xN]
            # 其中 x_i 要么是一个单独的数，要么是一个长度为 2 的 list
            self.data_received.emit(data)

    def update_plots(self, data_list):
        """Update the plots in the main thread based on the received data."""
        # data_list 的长度就是子图个数
        n = len(data_list)

        # 如果子图个数发生了变化，或第一次进来，需要重新创建
        if n != len(self.plots):
            self.win.clear()
            self.plots = []
            self.curves = []
            self.data = []

            for i in range(n):
                p = self.win.addPlot(row=i, col=0)
                p.showGrid(x=True, y=True)  # 启用网格线

                # 创建一个空列表来容纳子图里的多条曲线
                # 注意：可能是 1 条或 2 条曲线
                curve_list = []
                data_storage = []

                # 如果 data_list[i] 是单通道 (int/float)，就创建 1 条曲线
                # 如果是双通道 ([val1, val2])，就创建 2 条曲线
                # 下面我们先统一判断，实际上只要判断是不是可迭代并且长度为 2 即可
                vals = data_list[i]
                if not isinstance(vals, list):
                    vals = [vals]
                colors = ["r", "g", "b"]
                for j in range(len(vals)):
                    c = p.plot(pen=pg.mkPen(color=colors[j], width=2))
                    curve_list.append(c)
                    data_storage.append([])

                # 在 y=0 位置画一条水平线
                hline = pg.InfiniteLine(
                    pos=0, angle=0, pen=pg.mkPen(color="r", width=1)
                )
                p.addItem(hline)

                self.plots.append(p)
                self.curves.append(curve_list)
                self.data.append(data_storage)

        # 现在根据 data_list 的实际内容来 append 数据
        for i in range(n):
            vals = data_list[i]
            if not isinstance(vals, list):
                vals = [vals]
            for j in range(len(vals)):
                self.data[i][j].append(vals[j])
                self.data[i][j] = self.data[i][j][-500:]

    def update(self):
        """Update the curves with the latest data."""
        for i, curve_list in enumerate(self.curves):
            # 计算所有曲线的数据的最值，便于自适应 Y 轴
            all_values = []
            for j, curve in enumerate(curve_list):
                data_array = self.data[i][j]
                curve.setData(data_array)
                all_values.extend(data_array)

            if all_values:
                min_y = min(all_values)
                max_y = max(all_values)
                # 添加一些填充以避免数据紧贴边界
                padding = (max_y - min_y) * 0.1 if max_y != min_y else 1
                self.plots[i].setYRange(min_y - padding, max_y + padding)

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    plotter = LivePlotServer()
    plotter.run()
