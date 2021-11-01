from gym.envs.registration import register

register(
    id='my-environment-v1',
    entry_point='compression.envs.MyEnvironment:MyEnvironment',
)

# register(
#     id='simple-mass-v0',
#     entry_point='compression.envs.SimpleMassEnv:SimpleMassEnv',
# )

# register(
#     id='simple-mass-controlled-v0',
#     entry_point='compression.envs.SimpleMassEnv:SimpleMassWithFeedbackEnv',
# )

# register(
#     id='mav-v0',
#     entry_point='compression.envs.MavEnv:MavEnv',
# )

# register(
#     id='mav-nonlinear-v0',
#     entry_point='compression.envs.MavNonLinEnv:MavNonLinEnv',
# )

# register(
#     id='mav-nonlinear-controlled-v0',
#     entry_point='compression.envs.MavNonLinEnv:MavWithFeedbackNonLinEnv',
# )

# register(
#     id='mav-nonlinear-v1',
#     entry_point='compression.envs.MavNonLinWithPlannerEnv:MavWithPlannerNonLinEnv',
# )

# register(
#     id='mav-nonlinear-controlled-v1',
#     entry_point='compression.envs.MavNonLinWithPlannerEnv:MavWithPlannerWithFeedbackNonLinEnv',
# )