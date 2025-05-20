# gymnasium registry는 전역적이므로 고유 이름을 갖도록 해야함

import gymnasium as gym
from . import agents

## 
#  Register Gyn environments.
## 

gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    }   
)

# id: Issac- 접두사 붙이기 / Task의 이름 / 버전은 - v<N>
# entry_point: <module>:<class>
# env_cfg_entry_point: 기본 configuration을 부를 때 사용 (python.conf 혹은 yaml으로 가능)

# Direct-based environment의 경우 환경 등록 시 EntryPoint로 구현체를 넣는다.
# -Direct 접미사를 붙인다.

# __init__.py에 적절히 기입할 것
gym.register(
    id="Isaac-Cartpole-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",    }
)

import isaaclab_task # noqa: F401

# create environment configuration
env_cfg = parse_env_cfg(
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
# create environment
env = gym.make(args_cli.task, cfg=env_cfg)

# 환경을 만들면 그 후는 ressting and stepping
