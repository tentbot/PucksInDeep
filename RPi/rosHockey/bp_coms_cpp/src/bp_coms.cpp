#include <chrono>
#include <memory>


#include <stdio.h>
#include <string.h>
#include <fcntl.h> // Contains file controls like O_RDWR
#include <errno.h> // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h> // write(), read(), close()
#include <sys/ioctl.h> //FIONREAD for getting num bytes in buffer

#include "rclcpp/rclcpp.hpp"
#include "hockey_msgs/msg/mallet_pos.hpp"
#include "hockey_msgs/msg/next_path.hpp"
#include "hockey_msgs/msg/motor_status.hpp"


#define TIMER_FREQ 10ms
#define READ_SIZE 28
#define NUM_FLOATS_READ 7

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class BpComm : public rclcpp::Node
{
public:
  BpComm()
  : Node("bp_coms")
  {
    // ros stuff
    mallet_publisher_ = this->create_publisher<hockey_msgs::msg::MalletPos>("MALLET", 1);
    motor_publisher_ = this->create_publisher<hockey_msgs::msg::MotorStatus>("MOTOR", 1);
    timer_ = this->create_wall_timer(
        TIMER_FREQ, std::bind(&BpComm::read_bp, this));
    path_sub_ = this->create_subscription<hockey_msgs::msg::NextPath>(
        "PATH", 1, std::bind(&BpComm::write_bp, this, std::placeholders::_1));
    serial_port = config_tty();
    // clear_serial();
  }
  void close_serial();
  ~BpComm(){
      if (serial_port>0)
      	close_serial();
  }

private:

  int config_tty();
//   void clear_serial();
  void read_bp();
  void write_bp(const hockey_msgs::msg::NextPath::SharedPtr msg);
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<hockey_msgs::msg::MalletPos>::SharedPtr mallet_publisher_;
  rclcpp::Publisher<hockey_msgs::msg::MotorStatus>::SharedPtr motor_publisher_;
  rclcpp::Subscription<hockey_msgs::msg::NextPath>::SharedPtr path_sub_;
  struct termios tty;
  int serial_port;
  char read_buf [28];
  int n_read;
};

// void BpComm::clear_serial(){
//     int count = 0;
//     int bytes;
//     uint8_t init_byte[1];
//     ioctl(serial_port, FIONREAD, &bytes);
//     while (count < 4){
//         if (bytes > 0){
//             read(serial_port, &init_byte, sizeof(init_byte))
//             if (init_byte[0] == 0xff){
//                 count++;
//             }
//             else{
//                 count = 0
//             }
//         }
//         ioctl(serial_port, FIONREAD, &bytes);
//     }
//     int floats_read = 0;
//     printf("Clearing Serial");
//     while(floats_read <= NUM_FLOATS_READ+1){
//         if (bytes>=4){
//             read(serial_port, &byte_buff, __SIZEOF_FLOAT__);
//             memcpy(&sum, &byte_buff[0], __SIZEOF_FLOAT__);
//             if (sum==0xffffffff){
//                 printf("Here");
//                 floats_read++;
//             }

//             else if (floats_read>0){
//                 // read(serial_port, &byte_buff, __SIZEOF_FLOAT__);
//                 floats_read++;
//             }
//         }
//         ioctl(serial_port, FIONREAD, &bytes);
//     }
// }

void BpComm::read_bp(){
    // RCLCPP_INFO(this->get_logger(), "reading");
    int count = 0;
    int bytes;
    uint8_t init_byte[1];
    ioctl(serial_port, FIONREAD, &bytes);
    while (count < 4){
        if (bytes > 0){
            read(serial_port, &init_byte, sizeof(init_byte));
            if (init_byte[0] == 0xff){
                count++;
            }
            else{
                count = 0;
            }
        }
        ioctl(serial_port, FIONREAD, &bytes);
    }

// go fuck yourself

    // int bytes;
    // uint8_t start_bytes[__SIZEOF_FLOAT__];
    // uint32_t start_byte;
    // ioctl(serial_port, FIONREAD, &bytes);
    // if (bytes >= READ_SIZE + __SIZEOF_FLOAT__){
    //     read(serial_port, &start_bytes, __SIZEOF_FLOAT__);
    //     memcpy(&start_byte, &start_bytes[0], __SIZEOF_FLOAT__);

    //     if (start_byte != 0xffffffff){
    //         printf("ERROR: WRONG START BYTE. CHECK COMS\n");
    //         printf("%d", start_byte);
    //         printf("\n");
    //         }
        
    //     else {
    //         printf("OK!");
    //     }
    // if(bytes>READ_SIZE){
        while(bytes<READ_SIZE){
            ioctl(serial_port, FIONREAD, &bytes);
        }
        n_read = read(serial_port, &read_buf, sizeof(read_buf));
        if (n_read == READ_SIZE){
            hockey_msgs::msg::MalletPos mallet_msg = hockey_msgs::msg::MalletPos();
            hockey_msgs::msg::MotorStatus motor_msg = hockey_msgs::msg::MotorStatus();

            float float_values[NUM_FLOATS_READ];
            float f;
              for (int i = 0; i < NUM_FLOATS_READ; i++) {
                memcpy(&f, &(read_buf[4*i]), 4);
                float_values[i] = f;
            }
            mallet_msg.x = float_values[0];
            mallet_msg.y = float_values[1];
            mallet_msg.vx = float_values[2];
            mallet_msg.vy = float_values[3];
            motor_msg.m1effort = float_values[4];
            motor_msg.m2effort = float_values[5];
            mallet_msg.time_on_path = float_values[6];
            motor_msg.time_on_path = float_values[6];
            mallet_publisher_->publish(mallet_msg);
            motor_publisher_->publish(motor_msg);
        }
    // }
    // RCLCPP_INFO(this->get_logger(), "read");
}


void BpComm::write_bp(const hockey_msgs::msg::NextPath::SharedPtr msg_ptr){
    RCLCPP_INFO(this->get_logger(), "sending");
    uint8_t write_buffer[32];
    for (int i = 0; i<4; i++)
        write_buffer[i] = 0xff;
    const hockey_msgs::msg::NextPath msg = *msg_ptr;
    float vals[7] = {msg.x, msg.y, msg.vx, msg.vy, msg.ax, msg.ay, msg.t};
    // short vals[7] = {2,2,3,4,5,6,7};
    for (int i=0;i<7;i++){
        memcpy(&write_buffer[4*i+4], &vals[i], 4);
        // printf("Writing %.2f\n", vals[i]);
    }
    write(serial_port, write_buffer, 32);
        RCLCPP_INFO(this->get_logger(), "sent");
}

int BpComm::config_tty(){

    // serial stuff
    int serial_port = open("/dev/ttyUSB0", O_RDWR);
    if (serial_port < 0) {
        printf("Error %i from open: %s\n", errno, strerror(errno));
        printf("Trying /dev/ttyUSB1\n");
        serial_port = open("/dev/ttyUSB1", O_RDWR);
        if (serial_port<0){
            printf("Error %i from open: %s\n", errno, strerror(errno));
            printf("Trying /dev/ttyUSB2\n");
            serial_port = open("/dev/ttyUSB2", O_RDWR);
            if (serial_port<0) {
                printf("Could not open /dev/ttyUSB2\n");
                exit(1);
                return serial_port;
            }
        }
    }

    // Read in existing settings, and handle any error
    // NOTE: This is important! POSIX states that the struct passed to tcsetattr()
    // must have been initialized with a call to tcgetattr() overwise behaviour
    // is undefined
    if(tcgetattr(serial_port, &tty) != 0) {
        printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
    }
    
    //set 8 bits per byte
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cflag &= ~CRTSCTS; // Disable RTS/CTS hardware flow control (most common)
    tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    // Disable canonicall mode (I think we want this off cause we just looking at raw serial)
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO; // Disable echo
    tty.c_lflag &= ~ECHOE; // Disable erasure
    tty.c_lflag &= ~ECHONL; // Disable new-line echo

    tty.c_lflag &= ~ISIG; // Disable interpretation of INTR, QUIT and SUSP
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes
    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed

    tty.c_cc[VTIME] = 1;    // This sets time out of 0.1 s 
    tty.c_cc[VMIN] = 28;     // Wait for 28 bytes in

    cfsetispeed(&tty, B460800);
    cfsetospeed(&tty, B460800);

    // Save tty settings, also checking for error
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        printf("Error %i from tcsetattr: %s\n", errno, strerror(errno));
    }

    return serial_port;
}

void BpComm::close_serial(){
    close(serial_port);
}



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BpComm>());
  rclcpp::shutdown();
  return 0;
}
