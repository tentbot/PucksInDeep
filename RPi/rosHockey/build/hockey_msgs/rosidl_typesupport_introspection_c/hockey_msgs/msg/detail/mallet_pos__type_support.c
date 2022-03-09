// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from hockey_msgs:msg/MalletPos.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "hockey_msgs/msg/detail/mallet_pos__rosidl_typesupport_introspection_c.h"
#include "hockey_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "hockey_msgs/msg/detail/mallet_pos__functions.h"
#include "hockey_msgs/msg/detail/mallet_pos__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void MalletPos__rosidl_typesupport_introspection_c__MalletPos_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  hockey_msgs__msg__MalletPos__init(message_memory);
}

void MalletPos__rosidl_typesupport_introspection_c__MalletPos_fini_function(void * message_memory)
{
  hockey_msgs__msg__MalletPos__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_member_array[5] = {
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hockey_msgs__msg__MalletPos, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hockey_msgs__msg__MalletPos, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hockey_msgs__msg__MalletPos, vx),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hockey_msgs__msg__MalletPos, vy),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time_on_path",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hockey_msgs__msg__MalletPos, time_on_path),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_members = {
  "hockey_msgs__msg",  // message namespace
  "MalletPos",  // message name
  5,  // number of fields
  sizeof(hockey_msgs__msg__MalletPos),
  MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_member_array,  // message members
  MalletPos__rosidl_typesupport_introspection_c__MalletPos_init_function,  // function to initialize message memory (memory has to be allocated)
  MalletPos__rosidl_typesupport_introspection_c__MalletPos_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_type_support_handle = {
  0,
  &MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_hockey_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hockey_msgs, msg, MalletPos)() {
  if (!MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_type_support_handle.typesupport_identifier) {
    MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &MalletPos__rosidl_typesupport_introspection_c__MalletPos_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
