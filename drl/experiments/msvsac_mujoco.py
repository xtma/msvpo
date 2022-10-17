config = dict(
    env=dict(
        id="Hopper-v3",
        noise_scale=0.,
    ),
    agent=dict(
        model_kwargs=dict(hidden_sizes=[256, 256]),
        q_model_kwargs=dict(hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        min_steps_learn=int(1e4),
        replay_size=int(1e6),
        replay_ratio=256,  # data_consumption / data_generation
        target_update_tau=0.005,  # tau=1 for hard update.
        target_update_interval=1,  # 1000 for hard update, 1 for soft.
        learning_rate=3e-4,
        action_prior="uniform",  # or "gaussian"
        reward_scale=1,
        target_entropy="auto",  # "auto", float, or None
        reparameterize=True,
        clip_grad_norm=10,
        n_step_return=1,
        updates_per_sync=1,  # For async mode only.
        bootstrap_timelimit=True,
        clip_reward=None,
        reward_buffer_size=int(1e4),
        reward_norm=True,
        msd_coef_up=1.,
        msd_coef_down=1.,
        msd_decay=True,
    ),
    runner=dict(
        n_steps=1e6,
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
