import geometry_msgs.msg

def TfMatrix2RosQuatAndVector3(tf_matrix):

  translation_ros=geometry_msgs.msg.Vector3();
  rotation_ros=geometry_msgs.msg.Quaternion();

  translation=tf_matrix.translation();
  translation_ros.x=translation[0];
  translation_ros.y=translation[1];
  translation_ros.z=translation[2];
  # q=tr.quaternion_from_matrix(w_state.w_T_f.T)
  quaternion=tf.transformations.quaternion_from_matrix(tf_matrix.T) #See https://github.com/ros/geometry/issues/64
  rotation_ros.x=quaternion[0] #See order at http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
  rotation_ros.y=quaternion[1]
  rotation_ros.z=quaternion[2]
  rotation_ros.w=quaternion[3]

  return rotation_ros, translation_ros

def TfMatrix2RosPose(tf_matrix):

  rotation_ros, translation_ros=TfMatrix2RosQuatAndVector3(tf_matrix);

  pose_ros=geometry_msgs.msg.Pose();
  pose_ros.position.x=translation_ros.x
  pose_ros.position.y=translation_ros.y
  pose_ros.position.z=translation_ros.z

  pose_ros.orientation=rotation_ros

  return pose_ros
