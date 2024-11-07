import sys
import numpy as np
import pyqtgraph as pg
import time
import zmq
import argparse

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

class LivePlot():
    def __init__(self, jorder: np.ndarray):
        
        self.app = QApplication(sys.argv)

        # Create a GraphicsLayoutWidget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('4x3 Grid of Plots')
        self.win.resize(1200, 900)
        self.win.setBackground('lightgray')

        # List to store the plot items so we can update them later
        self.plot_items = []

        m, n = jorder.shape

        for row in range(m):
            for col in range(n):
                plot = self.win.addPlot(row=row, col=col)
                plot.setTitle(jorder[row, col])  # Set title for each plot
                plot.showGrid(x=True, y=True)

                # Create a plot item (line plot) and add it to the plot
                curve_0 = plot.plot(pen='b')  # Blue line plot
                curve_1 = plot.plot(pen='r')  # Red line plot
                self.plot_items.append((curve_0, curve_1))
        
        self.jpos = [[] for _ in range(12)]
        self.jpos_des = [[] for _ in range(12)]

        # Function to update the data    
        self.t0 = time.time()

    def run(self, addr: str = "tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.start_time = time.time()
        self.update_cnt = 0

        # Set up a timer to update the plots every 50ms
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(5)  # Update every 50 milliseconds

        # Show the window
        self.win.show()

        # Start the Qt event loop
        sys.exit(self.app.exec_())

    def update(self):
        data = self.socket.recv_pyobj()
        
        jpos = data["jpos"]
        jpos_des = data["jpos_des"]

        for i, (curve_0, curve_1) in enumerate(self.plot_items):
            # Generate some example data (sinusoidal wave for demo purposes)
            self.jpos[i] = self.jpos[i][-100:]
            self.jpos[i].append(jpos[i])
            curve_0.setData(self.jpos[i])

            self.jpos_des[i] = self.jpos_des[i][-100:]
            self.jpos_des[i].append(jpos_des[i])
            curve_1.setData(self.jpos_des[i])
        self.update_cnt += 1
        if self.update_cnt % 20 == 0:
            print(f"Update freq: {self.update_cnt / (time.time() - self.t0)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jorder', type=str, default="mjc")
    args = parser.parse_args()

    if args.jorder == "mjc":
        jorder = np.array(MJC_JOINT_ORDER).reshape(4, 3)
    elif args.jorder == "sdk":
        jorder = np.array(SDK_JOINT_ORDER).reshape(4, 3)

    liveplot = LivePlot(jorder)
    liveplot.run()