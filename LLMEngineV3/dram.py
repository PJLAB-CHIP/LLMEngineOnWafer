from dataclasses import dataclass, field
from enum import IntEnum
import os
from instance import Instance
from simulator import clock, schedule_event, cancel_event, reschedule_event

GB = 2**30


@dataclass#(kw_only=True)
class Dram():
    """
    Dram is the unit to store the weight & kv cache

    Each DRAM can belong to only one Die

    Attributes:
        memory_size (float): The memory size of the Processor.
        memory_used (float): The memory used by the Processor.
        die (die): The die that the Processor belongs to.
        instances (list[Instance]): Instances running on this Processor.
        interconnects (list[Link]): Peers that this Processor is directly connected to.
    """
    name: str = 'Dram'
    die = None
    _die = None
    dram_id: int = 0
    memory_size: int = 0  # GB
    memory_allocation: dict[int, int] = field(default_factory=dict) # die_id, size
    memory_used:int = 0
    _memory_used: int = 0
    bandwidth: int = 0
    latency: float = 0.
    
    def allocate_memory(self, die_id: int, size: int) -> None:
        if die_id in self.data:
            print(f"Updating key {die_id} from {self.memory_allocation[die_id]} GB to {size} GB")
        else:
            print(f"Adding key {die_id} with value {size} GB")
        self.data[die_id] = size
        
    @property
    def die(self):
        return self._die

    @die.setter
    def die(self, die):
        if type(die) is property:
            die = None
        self._die = die

    @property
    def memory_used(self):
        return sum(self.memory_allocation.values())

    @memory_used.setter
    def memory_used(self, memory_used):
        if type(memory_used) is property:
            memory_used = 0
        if memory_used < 0:
            raise ValueError("Memory cannot be negative")
        # if OOM, log instance details
        if memory_used > self.memory_size:
            if os.path.exists("oom.csv") is False:
                with open("oom.csv", "w", encoding="UTF-8") as f:
                    fields = ["time",
                              "instance_name",
                              "instance_id",
                              "memory_used",
                              "processor_memory",
                              "pending_queue_length"]
                    f.write(",".join(fields) + "\n")
            with open("oom.csv", "a", encoding="UTF-8") as f:
                instance = self.instances[0]
                csv_entry = []
                csv_entry.append(clock())
                csv_entry.append(instance.name)
                csv_entry.append(instance.instance_id)
                csv_entry.append(memory_used)
                csv_entry.append(self.memory_size)
                csv_entry.append(len(instance.pending_queue))
                f.write(",".join(map(str, csv_entry)) + "\n")
            # raise OOM error
            #raise ValueError("OOM")
        self._memory_used = memory_used

    @property
    def memory_free(self):
        return self.memory_size - self.memory_used

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
    def from_config(cls, *args, dram_id, **kwargs):
        dram_cfg = args[0]
        bandwidth = dram_cfg.bandwidth
        latency = dram_cfg.latency  # us
        memory_size = dram_cfg.memory_size*GB
        memory_used = dram_cfg.memory_used*GB
        return cls(dram_id = dram_id,
                   bandwidth = bandwidth,
                   latency = latency,
                   memory_size = memory_size,
                   memory_allocation = {dram_id: memory_size},
                   memory_used = memory_used)
        
