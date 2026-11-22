uint32 id
geometry_msgs/Pose Pose  # 位置 + 朝向（朝向沿用最近一次观测的 PnP 结果）
geometry_msgs/Vector3 velocity  # 速度；Step 2 恒为 0，Step 4/5 填
bool tracking  # true=有真实观测, false=纯预测
bool valid                       # 速度和预测是否可信；Step 2 恒为 false
float32 lost_time                # 距最后一次观测的时间（秒）
geometry_msgs/Point position_predict  # [改] t+T 时刻的预测位置（相机系，米），T = track_predict_T
