from collections import defaultdict
from die import Die
from instance import Instance
from itertools import count
import ipdb
class Chiplet:
    """
    Chiplet is a collection of Dies, Interconnected Links & DRAM.
    """
    def __init__(self,
                 dies: list[Die],
                 interconnects,
                 num_x,  # x axis die nums
                 num_y,  # y
                 tile_num,
                 drams,
                 power_budget):
        self.dies = dies  # List[tx8]
        self.interconnects = interconnects
        self.num_x = num_x
        self.num_y = num_y
        self.tile_num = tile_num
        self.tile_id = [(die_id, tile_id) for die_id in range(self.num_x * self.num_y) \
                                    for tile_id in range(self.tile_num)]
        self.tiles = [[tile for tile in die.tiles] for die in self.dies]
        self.drams = drams
        self.power_budget = power_budget
        self.total_power = 0
        self.instances = None  # 用于存储划分tp后的
        for die in self.dies:
            die.chiplet = self
        # 记录每个die被哪些dram分配了多少id
        for i in range(len(self.dies)):
            for dram in self.drams:
                if i in dram.memory_allocation.keys():
                    allocated_memory_value = self.dies[dram.dram_id].dram.memory_allocation.get(i, 0)
                    #print(f"Die {i}, DRAM {dram.dram_id}, allocated_memory_value: {allocated_memory_value}")
                    if dram.dram_id in self.dies[i].allocated_memory.keys():
                        self.dies[i].allocated_memory[dram.dram_id] += allocated_memory_value
                    else:
                        self.dies[i].allocated_memory[dram.dram_id] = allocated_memory_value
                    #print(f"Updated allocated_memory for die {i}, dram {dram.dram_id}: {self.dies[i].allocated_memory[dram.dram_id]}")
        #ipdb.set_trace()
            
    def __str__(self):
        return "Chiplet:" + str(self.dies)

    def add_dies(self, die: Die):
        self.dies.append(die)

    def remove_die(self, die: Die):
        self.dies.remove(die)

    def models(self):
        models = []
        for die in self.dies:
            models.extend(die.models)
        return models

    def run(self):
        """
        Runs dies in the cluster.
        """
        # NOTE: power usage updates not supported
        for die in self.dies:
            die.run()

                
    def start_spin_up_instance(self,
                               instance_cfg,
                               processors,
                               #parallelism,  # tp来划分每个gpu要存的权重 我们不用
                               pre_start=False,
                               tag=None):
        instance = Instance.from_config(instance_cfg=instance_cfg,
                                        instance_id=next(self.total_instances),
                                        application=self.application,
                                        #name=processors[0].name,
                                        tag=tag,
                                        #model=None #model,
                                        processors=processors,
                                        #overheads=self.instance_overheads,
                                        debug=self.debug)
        self.instances.append(instance)
    @classmethod
    def from_config(cls, *args, **kwargs):
        # args processing
        chiplet_cfg = args[0]
        print(chiplet_cfg.keys())
        dies_cfg = chiplet_cfg.dies
        #interconnects_cfg = chiplet_cfg.interconnects
        num_x = chiplet_cfg.num_x
        num_y = chiplet_cfg.num_y
        tile_num = 144#dies_cfg.tile_num
        # instantiate dies
        die_id = count()
        dies = []
        drams = []
        for _ in range(chiplet_cfg.die_num):
            die = Die.from_config(dies_cfg, die_id=next(die_id))  # 调用die的from_config的classmethod
            dies.append(die)
            drams.append(die.dram)
                

        # instantiate interconnects
        # TODO: add better network topology / configuration support
        interconnects = []
        # for interconnect_cfg in interconnects_cfg:
        #     if interconnect_cfg.topology == "p2p":
        #         continue
        #     interconnect = instantiate(interconnect_cfg)
        #     interconnects.append(interconnect)

        return cls(dies=dies,
                   interconnects=interconnects,
                   num_x=num_x,  # x axis die nums
                   num_y=num_y,  # y
                   tile_num=tile_num,
                   drams=drams,
                   power_budget=0)#chiplet_cfg.power_budget)