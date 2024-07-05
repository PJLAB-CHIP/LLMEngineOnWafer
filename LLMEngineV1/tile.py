from dataclasses import dataclass, field
from enum import IntEnum

from instance import Instance
from simulator import clock, schedule_event, cancel_event, reschedule_event




@dataclass#(kw_only=True)  # 数据类，并且只能用keyword初始化
class Tile():
    """
    Tile is the lowest-level processing unit that can run computations (Tasks).

    Each Tile can belong to only one Die
    Tile could eventually run multiple Instances/Tasks.

    Attributes:
        sram (float): The memory size of the Tile.
        memory_used (float): The memory used by the Tile.
        die (Die): The Die that the tile belongs to.
        instances (list[Instance]): Instances running on this Tile.
        interconnects (list[Link]): Peers that this Tile is directly connected to.
    """
    name: str = "Tile"
    sram: int = 0  # MB
    tile_id: int = 0
    bandwidth: int = 0# GB
    tflops: float = 0.
    tops: float = 0.
    _power: float = 0.  # TODO: if needed
    interconnects: list['Link'] = field(default_factory=list)
    instances: list['Tile'] = field(default_factory=list)
    die = None
    _die = None

    @property
    def die(self):
        return self._die

    @die.setter
    def die(self, server):
        if type(server) is property:
            server = None
        self._server = server

    @property
    def memory_used(self):
        return self._memory_used


    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        if type(power) is property:
            power = 0.
        if power < 0:
            raise ValueError("Power cannot be negative")
        self._power = power

    @classmethod
    def from_config(cls, *args, tile_id, **kwargs):
        tile_cfg = args[0][0]
        sram = tile_cfg.sram
        bandwidth = tile_cfg.bandwidth
        tflops = tile_cfg.tflops
        tops = tile_cfg.tops
        return cls(sram=sram,
                   tile_id = tile_id,
                   bandwidth=bandwidth,
                   tflops=tflops,
                   tops=tops)
        