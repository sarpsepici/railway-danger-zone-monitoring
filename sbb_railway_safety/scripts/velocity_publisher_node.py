#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TwistStamped

latest_cmd = TwistStamped()
last_update_time = None
timeout_seconds = 1.5  # Stop if no new velocity for this duration

def callback(msg):
    global latest_cmd, last_update_time
    latest_cmd = msg
    last_update_time = rospy.Time.now()

if __name__ == '__main__':
    rospy.init_node('velocity_publisher_node')
    rospy.loginfo("Velocity Publisher Node started.")

    rospy.Subscriber('/computed_velocity', TwistStamped, callback)
    pub = rospy.Publisher('/path_planning_and_following/twist', TwistStamped, queue_size=1)

    rate = rospy.Rate(10)  # 10 Hz
    zero_cmd = TwistStamped()

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        if last_update_time is not None and (now - last_update_time).to_sec() < timeout_seconds:
            pub.publish(latest_cmd)
        else:
            pub.publish(zero_cmd)

        rate.sleep()
