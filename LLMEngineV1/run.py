import logging
import os
import random
import sys
import ipdb
import hydra

from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from simulator import TraceSimulator
from initialize import *


# register custom hydra resolver
# 配置文件中使用 "eval" 解析器来动态执行包含 Python 表达式的字符串
OmegaConf.register_new_resolver("eval", eval)


def run_simulation(cfg):
    hardware_repo = init_hardware_repo(cfg)
    model_repo = init_model_repo(cfg)
    orchestrator_repo = init_orchestrator_repo(cfg)
    performance_model = init_performance_model(cfg)
    power_model = init_power_model(cfg)
    # 根据yaml配置文件使用instantiate函数初始化类对象
    chiplet = init_chiplet(cfg)
    #cluster = init_cluster(cfg)  # Cluster is a collection of Servers and interconnected Links.
    router = init_router(cfg, chiplet)  # Router routes Requests to Application Schedulers.
    arbiter = init_arbiter(cfg, chiplet)  # Arbiter allocates Processors to Application Allocators.
    applications = init_applications(cfg, chiplet, router, arbiter)  # An Application is the endpoint that a Request targets.
    trace = init_trace(cfg)  # Analyze and statistic Python program execution flow and call relation
    for application in applications.values():
        router.add_application(application)
        arbiter.add_application(application)
    sim = TraceSimulator(trace=trace,
                         cluster=chiplet,
                         applications=applications,
                         router=router,
                         arbiter=arbiter,
                         end_time=cfg.end_time)
    init_start_state(cfg,
                     cluster=chiplet,
                     applications=applications,
                     router=router,
                     arbiter=arbiter)
    sim.run()


# config_path: 指定 Hydra 在哪个目录中查找配置文件
# config_name: 指定要使用的配置文件的基本名称
# version_base: 指定 Hydra 的版本基础
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

    # initialize random number generator
    random.seed(cfg.seed)

    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):
        os.remove("oom.csv")

    run_simulation(cfg)


if __name__ == "__main__":
    run()
