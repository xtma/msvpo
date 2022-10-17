config = dict(
    agent=dict(
        model_kwargs=dict(hidden_sizes=[400, 300]),
        z_model_kwargs=dict(hidden_sizes=[400, 300]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        replay_ratio=256,
        target_update_tau=0.005,
        target_update_interval=2,
        policy_update_interval=2,
        learning_rate=1e-3,
        z_learning_rate=1e-3,
        clip_grad_norm=10,
        clip_reward=10,
    ),
    env=dict(id="Hopper-v3"),
    # eval_env=dict(id="Hopper-v3"),  # Same kwargs as env, in train script.
    optim=dict(),
    runner=dict(
        n_steps=3e6,
        log_interval_steps=5e3,
    ),
    sampler=dict(
        batch_T=10,
        batch_B=10,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(5e4),
        eval_max_trajectories=10,
    ),
)