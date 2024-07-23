#include <string>
#include <thread>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include "unitree/idl/go2/LowState_.hpp"
#include <unitree/idl/go2/SportModeState_.hpp>
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

#include "robot_interface.hpp"
#include "gamepad.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/chrono.h"
#include "pybind11/numpy.h"

using namespace unitree::common;
using namespace unitree::robot;
namespace py = pybind11;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_HIGHSTATE "rt/sportmodestate"

class RobotIface
{
public:
    RobotIface() {
        // sport_client.SetTimeout(10.0f);
        // sport_client.Init();
        rsc.SetTimeout(10.0f);
        rsc.Init();

        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher->InitChannel();

        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber->InitChannel(std::bind(&RobotIface::LowStateMessageHandler, this, std::placeholders::_1), 1);
        
        highstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        highstate_subscriber->InitChannel(std::bind(&RobotIface::HighStateMessageHandler, this, std::placeholders::_1), 1);
        
        std::cout << "Init complete!" << std::endl;
    }

    void StartControl() {
        // waiting for gamepad command to start the control thread
        std::chrono::milliseconds duration(100);
        // listen to gamepad command

        // while (true)
        // {
        //     std::cout << "Press R2 to start!" << std::endl;
        //     std::this_thread::sleep_for(duration);

        //     if (gamepad.R2.on_press)
        //     {
        //         break;
        //     }
        // }

        rsc.ServiceSwitch("sport_mode", 1);
        // UserControlCallback();

        main_thread_ptr = CreateRecurrentThreadEx("main", UT_CPU_ID_NONE, 500, &RobotIface::MainThreadStep, this);
        
        std::cout << "Start control!" << std::endl;
    }

    void SetCommand(const py::array_t<float> &input_cmd) {
        std::lock_guard<std::mutex> lock(cmd_mutex);
        py::buffer_info buf = input_cmd.request();
        if (buf.size != robot_interface.jpos_des.size()) {
            throw std::runtime_error("Command size must be 12");
        }
        std::copy(input_cmd.data(), input_cmd.data() + input_cmd.size(), robot_interface.jpos_des.begin());
    }

    py::array_t<float> GetJointPos() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jpos.size(), robot_interface.jpos.data());
    }

    py::array_t<float> GetJointPosTarget() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jpos_des.size(), robot_interface.jpos_des.data());
    }

    py::array_t<float> GetJointVel() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jvel.size(), robot_interface.jvel.data());
    }

    py::array_t<float> GetQuat() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.quat.size(), robot_interface.quat.data());
    }

    py::array_t<float> GetRPY() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.rpy.size(), robot_interface.rpy.data());
    }


    py::array_t<float> GetProjectedGravity() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(
            robot_interface.projected_gravity.size(), 
            robot_interface.projected_gravity.data()
        );
    }

    py::array_t<float> lxy() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 2> lxy = {gamepad.lx, gamepad.ly};
        return py::array_t<float>(lxy.size(), lxy.data());
    }

    py::array_t<float> rxy() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 2> rxy = {gamepad.rx, gamepad.ry};
        return py::array_t<float>(rxy.size(), rxy.data());
    }

    py::array_t<float> GetKp() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.kp.size(), robot_interface.kp.data());
    }

    py::array_t<float> GetKd() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.kd.size(), robot_interface.kd.data());
    }

    py::array_t<float> GetVelocity() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto velocity = high_state.velocity();
        return py::array_t<float>(velocity.size(), velocity.data());
    }

    py::array_t<float> GetFootPos() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto feet_pos = high_state.foot_position_body();
        return py::array_t<float>(feet_pos.size(), feet_pos.data());
    }

    py::array_t<float> GetFullState() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 12 + 12 + 12 + 3 + 3> state;
        std::copy(
            robot_interface.jpos.begin(), 
            robot_interface.jpos.begin() + robot_interface.jpos.size(), 
            state.begin()
        );
        std::copy(
            robot_interface.jvel.begin(),
            robot_interface.jvel.begin() + robot_interface.jvel.size(),
            state.begin() + 12
        );
        std::copy(
            robot_interface.tau.begin(),
            robot_interface.tau.begin() + robot_interface.tau.size(),
            state.begin() + 24
        );
        std::copy(
            robot_interface.rpy.begin(),
            robot_interface.rpy.begin() + 3, 
            state.begin() + 36
        );
        std::copy(
            robot_interface.gyro.begin(),
            robot_interface.gyro.begin() + 3,
            state.begin() + 39
        );
        return py::array_t<float>(state.size(), state.data());
    }

    py::float_ GetYawSpeed() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto yaw_speed = high_state.yaw_speed();
        return yaw_speed;
    }

    void SetKp(const float value) {
        robot_interface.kp.fill(value);
    }

    void SetKd(const float value) {
        robot_interface.kd.fill(value);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    double interval;

private:
    void LowStateMessageHandler(const void *message)
    {
        low_state = *(unitree_go::msg::dds_::LowState_ *)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            robot_interface.GetState(low_state);
            // update gamepad
            memcpy(rx.buff, &low_state.wireless_remote()[0], 40);
            gamepad.update(rx.RF_RX);
        }
        auto now = std::chrono::high_resolution_clock::now();
        interval = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp).count();
        timestamp = std::chrono::high_resolution_clock::now();
    }

    void HighStateMessageHandler(const void *message)
    {
        high_state = *(unitree_go::msg::dds_::SportModeState_ *)message;
    }

    void LowCmdwriteHandler()
    {
        // write low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            lowcmd_publisher->Write(cmd);
        }
    }
    
    void MainThreadStep()
    {
        if (gamepad.L1.on_press && control_mode==0) {
            control_mode = 1;
            rsc.ServiceSwitch("sport_mode", 0);
            std::cout << "L1 pressed, switch mode to user control" << std::endl;
            UserControlCallback();
        } else if (gamepad.L1.on_press && control_mode==1) {
            control_mode = 0;
            rsc.ServiceSwitch("sport_mode", 1);
            std::cout << "L1 pressed, switch mode to sport mode" << std::endl;
        }

        // write low-level command
        if (control_mode == 1) {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            robot_interface.SetCommand(cmd);
            lowcmd_publisher->Write(cmd);

            if (robot_interface.projected_gravity.at(2) > -0.3) {
                std::cout << "Falling detected, damping" << std::endl;
                Damping();
            }
        }
    }

    void Damping() {
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(0.);
        robot_interface.kd.fill(2.5);
        robot_interface.tau_ff.fill(0.);
    }

    void UserControlCallback() {
        // robot_interface.jpos_des = {0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8};
        robot_interface.jvel_des.fill(0.);
        // robot_interface.kp.fill(30.);
        // robot_interface.kd.fill(1.0);
        robot_interface.tau_ff.fill(0.);
    }

protected:
    RobotInterface robot_interface;

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> highstate_subscriber;

    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr, main_thread_ptr;

    unitree_go::msg::dds_::LowCmd_ cmd;
    unitree_go::msg::dds_::LowState_ low_state;
    unitree_go::msg::dds_::SportModeState_ high_state;

    // high-level
    go2::SportClient sport_client;
    go2::RobotStateClient rsc;
    std::array<float, 3> velocity;

    Gamepad gamepad;
    REMOTE_DATA_RX rx;

    std::mutex state_mutex, cmd_mutex;

    int control_mode=0; // 0: sportmode, 1: usermode
};

void InitChannel(const std::string &networkinterface) {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkinterface);
}

PYBIND11_MODULE(go2py, m)
{
    py::class_<RobotIface>(m, "RobotIface")
        .def(py::init<>())
        .def("start_control", &RobotIface::StartControl)
        .def("get_joint_pos", &RobotIface::GetJointPos)
        .def("get_joint_pos_target", &RobotIface::GetJointPosTarget)
        .def("get_joint_vel", &RobotIface::GetJointVel)
        .def("get_quat", &RobotIface::GetQuat)
        .def("get_rpy", &RobotIface::GetRPY)
        .def("get_projected_gravity", &RobotIface::GetProjectedGravity)
        .def("lxy", &RobotIface::lxy)
        .def("rxy", &RobotIface::rxy)
        .def("set_command", &RobotIface::SetCommand)
        .def("get_kd", &RobotIface::GetKd)
        .def("get_kp", &RobotIface::GetKp)
        .def("set_kd", &RobotIface::SetKd)
        .def("set_kp", &RobotIface::SetKp)
        .def("get_velocity", &RobotIface::GetVelocity)
        .def("get_feet_pos", &RobotIface::GetFootPos)
        .def("get_yaw_speed", &RobotIface::GetYawSpeed)
        .def("get_full_state", &RobotIface::GetFullState)
        .def_readonly("timestamp", &RobotIface::timestamp)
        .def_readonly("interval", &RobotIface::interval)
        ;
    
    m.def("init_channel", &InitChannel, py::arg("networkinterface"));
}
