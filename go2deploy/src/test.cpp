#include <string>
#include <thread>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include "unitree/idl/go2/LowState_.hpp"
#include <unitree/idl/go2/SportModeState_.hpp>
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

#include "robot_interface.hpp"
#include "gamepad.hpp"

// #include "pybind11/pybind11.h"
// #include "pybind11/numpy.h"

using namespace unitree::common;
using namespace unitree::robot;
// namespace py = pybind11;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_HIGHSTATE "rt/sportmodestate"

class RobotIface
{
public:
    RobotIface() {}

    void InitDdsModel() {
        sport_client.SetTimeout(10.0f);
        sport_client.Init();
        
        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher->InitChannel();

        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber->InitChannel(std::bind(&RobotIface::LowStateMessageHandler, this, std::placeholders::_1), 1);
        
        std::cout << "init complete" << std::endl;
    }
    
    void InitDdsModel(const std::string &networkInterface) {
        std::cout << networkInterface << std::endl;
        ChannelFactory::Instance()->Init(0, networkInterface);
        sport_client.SetTimeout(10.0f);
        sport_client.Init();
        
        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher->InitChannel();

        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber->InitChannel(std::bind(&RobotIface::LowStateMessageHandler, this, std::placeholders::_1), 1);
        
        std::cout << "init complete" << std::endl;
    }

    void StartControl()
    {
        std::chrono::milliseconds duration(100);

        // init dds
        main_thread_ptr = CreateRecurrentThreadEx("main", UT_CPU_ID_NONE, 2000, &RobotIface::MainThreadStep, this);

        // keep the main thread alive
        while (true)
        {
            std::this_thread::sleep_for(duration);
        }
    }

    // void SetCommand(const py::array_t<float> &input_cmd) {
    //     std::lock_guard<std::mutex> lock(cmd_mutex);
    //     py::buffer_info buf = input_cmd.request();
    //     if (buf.size != robot_interface.jpos_des.size()) {
    //         throw std::runtime_error("Command size must be 12");
    //     }
    //     std::copy(input_cmd.data(), input_cmd.data() + input_cmd.size(), robot_interface.jpos_des.begin());
        
    //     robot_interface.SetCommand(cmd);
    // }

    // py::array_t<float> GetJointPos() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     return py::array_t<float>(robot_interface.jpos.size(), robot_interface.jpos.data());
    // }

    // py::array_t<float> GetJointPosTarget() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     return py::array_t<float>(robot_interface.jpos_des.size(), robot_interface.jpos_des.data());
    // }

    // py::array_t<float> GetJointVel() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     return py::array_t<float>(robot_interface.jvel.size(), robot_interface.jvel.data());
    // }

    // py::array_t<float> GetQuat() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     return py::array_t<float>(robot_interface.quat.size(), robot_interface.quat.data());
    // }

    // py::array_t<float> lxy() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     std::array<float, 2> lxy = {gamepad.lx, gamepad.ly};
    //     return py::array_t<float>(lxy.size(), lxy.data());
    // }

    // py::array_t<float> rxy() {
    //     std::lock_guard<std::mutex> lock(state_mutex);
    //     std::array<float, 2> rxy = {gamepad.rx, gamepad.ry};
    //     return py::array_t<float>(rxy.size(), rxy.data());
    // }


private:
    void LowStateMessageHandler(const void *message)
    {
        state = *(unitree_go::msg::dds_::LowState_ *)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            robot_interface.GetState(state);
            // update gamepad
            memcpy(rx.buff, &state.wireless_remote()[0], 40);
            gamepad.update(rx.RF_RX);
        }
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
        if (gamepad.R1.on_press) {
            std::cout << "R1 pressed!" << std::endl;
            sport_client.StandUp();
        }
        else if (gamepad.R2.on_press)
        {
            std::cout << "R2 pressed!" << std::endl;
            sport_client.StandDown();
        }
        
    }

protected:
    RobotInterface robot_interface;

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr, main_thread_ptr;

    unitree_go::msg::dds_::LowCmd_ cmd;
    unitree_go::msg::dds_::LowState_ state;

    // high-level
    unitree::robot::go2::SportClient sport_client;

    Gamepad gamepad;
    REMOTE_DATA_RX rx;

    std::mutex state_mutex, cmd_mutex;
};

int main(int argc, char **argv) {
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    RobotIface robot;
    robot.InitDdsModel();
    robot.StartControl();
}