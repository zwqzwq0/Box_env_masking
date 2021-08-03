import ray
import ray.rllib.agents.ppo as ppo
from box_env import Box_Env
from box_model import Box_Model
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print


def env_creator(env_config):
    return Box_Env(env_config['map'])  # return an env instance

if __name__ == '__main__':
    ray.init()
    register_env("my_env", env_creator)
    ModelCatalog.register_custom_model(
        "box",Box_Model
    )
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["model"]={
        "custom_model":"box",
        "vf_share_layers":True,
    }
    config["framework"]="tf"
    preprocessor_pref = 'rllib'
    trainer = ppo.PPOTrainer(config={"env_config": {"map": [[1, 0, 0], [0, 0, -1]]}}, env='my_env')
    # 加载训练完的节点
    # agent.restore("/Users/4paradigm/ray_results/checkpoint_000100/checkpoint-100")
    for i in range(100):
        result=trainer.train()
        if (i+1)%10==0:
            print(pretty_print(result))

    trainer.save("/Users/4paradigm/ray_results")

    env = Box_Env([[1, 0, 0], [0, 0, -1]])
    obs=env.reset()
    done=False
    total_reward=0
    while not done:
        action=trainer.compute_action(obs)
        print(action)
        obs,reward,done,_=env.step(action)
        total_reward+=reward
        if done:
            print(total_reward)
            break
    env.close()
