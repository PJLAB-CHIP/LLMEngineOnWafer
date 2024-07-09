
# used by hydra instantiate
from tile import Tile
from dram import Dram
from itertools import count
from dataclasses import field
import utils
import ipdb
import interconnect



class Die:
    """
    Dies are a collection of Tiles that may be connected by local Interconnects. 
    Dies themselves are also interconnected by Interconnects. 
    dies run Instances (partially or fully).

    Attributes:
        die_id (str): The unique die_id of the die.
        tiles (list): A list of Tiles.
        interconnects (list[Link]): Peers that this Die is
                                    directly connected to.
    """
    dies = {}  # class variable 用于记录所有实例
    # logger for all dies
    logger = None

    def __init__(self,
                 die_id,
                 d2d_bw,
                 dram,
                 num_x,
                 num_y,
                 tiles,
                 interconnects):
        # HACK: 避免Hydra多次运行时的重复实例化问题
        if die_id in Die.dies:
            Die.dies = {}
            Die.logger = None
        self.d2d_bw = d2d_bw
        self.die_id = die_id
        self.store_weight = False  # for the first 
        self.name = 'Die'
        self.dram  = dram
        self.num_x = num_x
        self.num_y = num_y
        self.tiles = tiles
        for tile in self.tiles:
            tile._die = self
        # allocated memory from others' die
        self.allocated_memory: dict[int, int] = {}
        self.interconnects = interconnects
        # for intercon in self.interconnects:
        #     intercon.die = self
        self.chiplet = None
        # Die's Coordination
        self.loc_x = int(die_id) % self.num_x
        self.loc_y = int(die_id) // self.num_y
        Die.dies[die_id] = self  # Add to global variable
        self.instances :list[list[Tile]]= []  # Tensor parallel
        self.power = 0
        #self.update_power(0)
        dram._die = self
        for tile in self.tiles:
            tile.die = self
        #self._instances = []

        # initialize die logger
        if Die.logger is None:
            self.logger = utils.file_logger("die")
            Die.logger = self.logger
            self.logger.info("time,die")
        else:
            self.logger = Die.logger

    def __str__(self):
        return f"Die:{self.die_id}"

    def __repr__(self):
        return self.__str__()

    @property
    def instances(self):
        return self._instances

    @instances.setter
    def instances(self, instances):
        self._instances = instances

    # def update_power(self, power):
    #     old_power = self.power
    #     self.power = get_die_power(self) + \
    #                     sum(processor.power for processor in self.tiles)
    #     if self.cluster:
    #         self.cluster.update_power(self.power - old_power)

    def run(self):
        pass

    @classmethod
    def load(cls):
        pass

    @classmethod
    def from_config(cls, *args, die_id, **kwargs):
        die_cfg = args[0][0]
        processors_cfg = die_cfg.tiles
        dram_cfg = die_cfg.dram
        num_x = die_cfg.num_x
        num_y = die_cfg.num_y
        d2d_bw = die_cfg.d2d_bw
        #interconnects_cfg = die_cfg.interconnects
        tile_id = count()
        dram = Dram.from_config(dram_cfg, dram_id=die_id)
        tiles = []
        tile_id = count()
        for _ in range(die_cfg.tile_num):
            tile = Tile.from_config(processors_cfg, tile_id=next(tile_id))
            tiles.append(tile)
            

        # TODO: add better network topology / configuration support
        # interconnects = []
        # for interconnect_name in interconnects_cfg:
        #     intercon = hardware_repo.get_interconnect(interconnect_name)
        #     interconnects.append(intercon)

        return cls(die_id=die_id,
                   d2d_bw=d2d_bw,
                   #name=die_cfg.name,
                   dram=dram,
                   num_x=num_x,
                   num_y=num_y,
                   tiles=tiles,
                   interconnects=None)#interconnects)
    