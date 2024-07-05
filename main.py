import hydra
import ipdb
import os
from components.datastructure.trace import Trace
from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from components.simulator import TraceSimulator
from components.chiplet import Chiplet
from components.application import Application
# from components.Die import Die
# from components.Tile import Tile


# config_path: 指定 Hydra 在哪个目录中查找配置文件
# config_name: 指定要使用的配置文件的基本名称
# version_base: 指定 Hydra 的版本基础
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))
    print(cfg.keys())
    chiplet = Chiplet.from_config(cfg.chiplet)
    router = None  # init router
    arbiter = None  # init arbitger
    # init application
    application = Application.from_config(cfg,
                                          chiplet=chiplet,
                                          router=router,
                                          arbiter=arbiter)
    # init trace
    trace_path = os.path.join(get_original_cwd(), cfg.trace_path)
    trace = Trace.from_csv(trace_path)
    ipdb.set_trace()
    # add application to components
    # add to router
    # add to arbiter
    sim = TraceSimulator(trace=trace,
                         chiplet=chiplet,
                         applications=application,
                         router=router,
                         arbiter=arbiter,
                         end_time=cfg.end_time)
    start_state(cfg, chiplet, application)
    sim.run()
    # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

def start_state(cfg, chiplet, application):
    # HACK: 直接顺序遍历tp而不是
    # TODO: profile run to determine
    prompt_cfg = dict()
    token_cfg = dict()
    
    # 获取新的参数
    prompt_tp = prompt_cfg.prompt_tp  # 18
    token_tp = token_cfg.token_tp  # 16
    n_prompts = prompt_cfg.n_prompt  # 80 
    n_token = prompt_cfg.n_token  # 90
    tiles = [(die_id, tile_id) for die_id in range(chiplet.num_x * chiplet.num_y) \
                                    for tile_id in range(chiplet.tile_num)]
    prefill_groups = []
    decode_groups = []
    remaining_tiles = set(tiles)
    def create_groups(num_groups, group_size):
        groups = []
        for _ in range(num_groups):
            if len(remaining_tiles) < group_size:
                break
            group = set()
            for tile in list(remaining_tiles):
                group.add(tile)
                remaining_tiles.remove(tile)
                if len(group) == group_size:
                    break
            groups.append(group)
        return groups

    prefill_groups = create_groups(n_prompts, prompt_tp)
    decode_groups = create_groups(n_token, token_tp)
    for prefill_group in prefill_groups:
        chiplet.start_spin_up_instance(instance_cfg=prompt_cfg,
                                        processors=prefill_group,
                                        #parallelism=prompt_parallelism,
                                        pre_start=True,
                                        tag="prompt")
    for decode_group in decode_groups:
         chiplet.start_spin_up_instance(instance_cfg=prompt_cfg,
                                        processors=decode_group,
                                        #parallelism=token_parallelism,
                                        pre_start=True,
                                        tag="token")
    return prefill_groups, decode_groups


   
if __name__ == '__main__':
    #trace_path = os.path.join(get_original_cwd(), 'traces/AzureLLMInferenceTrace_conv.csv')
    # trace = Trace.from_csv('./traces/AzureLLMInferenceTrace_conv.csv')
    # 读取数据 ok！
    run()