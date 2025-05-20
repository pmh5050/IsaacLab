
# @configclass
# class RewardsCfg:
#     """Rewards terms for the MDP"""

#     # (1) Constant running reward
#     alive = RewTerm(func=mdp.is_alive, weight=1.0)
#     # (2) Failure penalty
#     terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
#     # (3) Primary task: keep pole upright
#     pole_pos = RewTerm(
#         func=mdp.joint_pos_target_l2,
#         weight=-1.0,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
#     )
#     # (4) Shaping tasks: lower cart velocity
#     cart_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.01,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
#     )
#     # (5) Shaping tasks: lower pole angular velocity
#     pole_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.005,
#         params={"asset_cfg": SceneEntitiyCfg("robot", joint_names=["cart_to_pole"])}
#     )

# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     # (1) Time out
#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     # (2) Cart out of bounds
#     cart_out_of_bounds = DoneTerm(
#         func=mdp.joint_pos_out_of_manual_limit,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
#     )

# # managers.CommandManager를 통해 Goal-Conditioned task/command에 대한 명세를 수행할 수 있다.
# # + command를 agent의 obs로 제공하는 기능도 가능
# # + 이 예제에서는 단순하므로 기본값(None)으로 남김

# # managers.CurriculumManger를 통해 커리큘럼 학습 기능도 지원

# @configclass
# class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
#     """Configuration for the cartpole environment."""

#     # Scene settings
#     scene: CartpoleSceneCfg = CartpoleSCeneCfg(num_envs=4096, env_spacing=4.0)
#     # Basic settings
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     events: EventCfg = EventCfg()
#     # MDP settings
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()


#     # Post initialization
#     def __post_init__(self) -> None:
#         """Post initialization."""
#         # general settings
#         self.decimation = 2 # 아마 60Hz?
#         self.episode_lenght_s = 5
#         # viewer settings
#         self.viewer.eye = (8.0, 0.0, 5.0)
#         # simulation settings
#         self.sim.dt = 1 / 120
#         self.sim.render_interval = self.decimation

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

def main():
    """Main function."""
    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset() # reset 시점에서 obs를 반환하는 지 확인할 것
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update cointer
            count += 1
        
    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()