# Per-step diagnostics of the NMPC node.
std_msgs/Header header       # stamp of the state this step answered
float64 solve_time_ms        # acados internal time (time_tot)
float64 step_time_ms         # whole callback: reference, warm start, solve, publish
float64 latency_ms           # state stamp -> command published
int32 status                 # acados status (0 = success)
float64 tcp_error            # distance of the TCP to the reference at this state (m)
float64 max_velocity_ratio   # max |dq_i| / dq_max_i at this state
