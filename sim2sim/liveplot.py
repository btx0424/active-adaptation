import sys
import numpy as np
import pyqtgraph as pg
import time
import zmq
import argparse
import threading

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

SDK_JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]

MJC_JOINT_ORDER = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]

RANGES = [
    [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, -0.],
    [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, -0.],
    [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, -0.],
    [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, -0.]
]

class LivePlot():
    def __init__(
        self,
        jorder: np.ndarray,
        jrange: np.ndarray,
        time_span: int = 200
    ):
        self.time_span = time_span
        self.app = QApplication(sys.argv)

        # Create a GraphicsLayoutWidget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('4x3 Grid of Plots')
        self.win.resize(1200, 900)
        self.win.setBackground('lightgray')

        # List to store the plot items so we can update them later
        self.plot_items = []

        m, n = jorder.shape

        self.foot_force_plots = []
        for row in range(m):
            for col in range(n):
                plot = self.win.addPlot(row=row, col=col)
                plot.setTitle(jorder[row, col])  # Set title for each plot
                plot.showGrid(x=True, y=True)
                plot.setYRange(jrange[row, col][0], jrange[row, col][1])

                # Create a plot item (line plot) and add it to the plot
                curve_0 = plot.plot(pen=pg.mkPen(color='b', width=2))  # Blue line plot
                curve_1 = plot.plot(pen=pg.mkPen(color='r', width=2))  # Red line plot
                curve_2 = plot.plot(pen=pg.mkPen(color='g', width=2))  # Green line plot
                self.plot_items.append((curve_0, curve_1, curve_2))
            plot = self.win.addPlot(row=row, col=col+1)
            plot.setTitle("Foot Force")
            curve = plot.plot(pen=pg.mkPen(color='b', width=2))  # Blue line plot
            self.foot_force_plots.append(curve)
        
        self.rpy_plots = []
        self.gyro_plots = []
        for i, axis in enumerate(["Roll", "Pitch", "Yaw"]):
            plot = self.win.addPlot(row=m+1, col=i, title=axis)
            plot.showGrid(x=True, y=True)
            # if i < 2:
            #     plot.setYRange(-np.pi / 2, np.pi / 2)
            curve_rpy = plot.plot(pen='b')
            curve_gyro = plot.plot(pen='r')
            self.rpy_plots.append(curve_rpy)
            self.gyro_plots.append(curve_gyro)
        
        self.acc_plots = []
        for i, axis in enumerate(["X", "Y", "Z"]):
            plot = self.win.addPlot(row=m+2, col=i, title=axis)
            plot.showGrid(x=True, y=True)
            curve_acc = plot.plot(pen='b')
            self.acc_plots.append(curve_acc)

        self.jpos = [[] for _ in range(12)]
        self.jpos_des = [[] for _ in range(12)]
        self.tau = [[] for _ in range(12)]
        self.rpy = [[] for _ in range(3)]
        self.gyro = [[] for _ in range(3)]
        self.foot_force = [[] for _ in range(4)]
        self.acc = [[] for _ in range(3)]

        # Function to update the data    
        self.t0 = time.time()

    def run(self, addr: str = "tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.start_time = time.time()
        self.update_cnt = 0

        timer = QTimer(self.app)
        timer.timeout.connect(self.update_plot)
        timer.start(10)

        threading.Thread(target=self.update_data).start()
        # Show the window
        self.win.show()

        # Start the Qt event loop
        sys.exit(self.app.exec_())

    def update_data(self):
        while True:
            data = self.socket.recv_pyobj()

            if "jpos" in data:
                jpos = data["jpos"]
                jpos_des = data["jpos_des"]

                for i, (curve_0, curve_1, curve_2) in enumerate(self.plot_items):
                    self.jpos[i] = self.jpos[i][-self.time_span:]
                    self.jpos[i].append(jpos[i])

                    self.jpos_des[i] = self.jpos_des[i][-self.time_span:]
                    self.jpos_des[i].append(jpos_des[i])
            
            if "rpy" in data:
                rpy = data["rpy"]
                for i, curve in enumerate(self.rpy_plots):
                    self.rpy[i] = self.rpy[i][-self.time_span:]
                    self.rpy[i].append(rpy[i])
            
            if "gyro" in data:
                gyro = data["gyro"]
                for i, curve in enumerate(self.gyro_plots):
                    self.gyro[i] = self.gyro[i][-self.time_span:]
                    self.gyro[i].append(gyro[i])
            
            if "foot_force" in data:
                foot_force = data["foot_force"]
                for i, curve in enumerate(self.foot_force_plots):
                    self.foot_force[i] = self.foot_force[i][-self.time_span:]
                    self.foot_force[i].append(foot_force[i])
            
            if "acc" in data:
                acc = data["acc"]
                for i, curve in enumerate(self.acc_plots):
                    self.acc[i] = self.acc[i][-self.time_span:]
                    self.acc[i].append(acc[i])
            
            self.update_cnt += 1
            if self.update_cnt % self.time_span == 0:
                update_freq = self.time_span / (time.time() - self.t0)
                print(f"Update freq: {update_freq:.2f}")
                self.t0 = time.time()
    
    def update_plot(self):

        for i, (curve_0, curve_1, curve_2) in enumerate(self.plot_items):
            curve_0.setData(self.jpos[i])
            curve_1.setData(self.jpos_des[i])
            curve_2.setData(self.tau[i])
        
        for i, curve in enumerate(self.rpy_plots):
            curve.setData(self.rpy[i])

        for i, curve in enumerate(self.gyro_plots):
            curve.setData(self.gyro[i])

        for i, curve in enumerate(self.foot_force_plots):
            curve.setData(self.foot_force[i])
        
        for i, curve in enumerate(self.acc_plots):
            curve.setData(self.acc[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jorder', type=str, default="mjc")
    parser.add_argument("--ip", type=str, default="localhost")
    args = parser.parse_args()

    if args.jorder == "mjc":
        jorder = np.array(MJC_JOINT_ORDER).reshape(4, 3)
    elif args.jorder == "sdk":
        jorder = np.array(SDK_JOINT_ORDER).reshape(4, 3)
    
    jrange = np.array(RANGES).reshape(4, 3, 2)

    liveplot = LivePlot(jorder, jrange, 400)
    liveplot.run(addr=f"tcp://{args.ip}:5555")