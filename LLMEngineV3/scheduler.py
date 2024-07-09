import logging
import os
import sys
import time
import ipdb
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import random
import utils
import time
from executor import Executor, ExecutorType
from interconnect import DummyLink
from performance_model import get_duration
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import Task, TaskType
from flow import FlowType
from mapper.static_mapper import *
from instance import SplitwiseInstance
from node import *
from copy import deepcopy


class Scheduler(ABC):
    """
    Scheduler schedules Requests to Instances and spawns Executors to handle them.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        self.application = application
        self.router = router
        self.overheads = overheads
        self.executor_overheads = executor_overheads
        self.debug = debug

        # instances
        self.instances = []

        # request queues
        self.pending_queue = []
        self.executing_queue = []
        self.completed_queue = []

        # executors
        self.executor_type = ExecutorType.CentralExecutor
        self.executors = {}

        # logger for scheduler actions
        logger_name = f"schedulers/{self.application.application_id}"
        level = logging.DEBUG if self.debug else logging.INFO
        os.makedirs("schedulers", exist_ok=True)
        self.scheduler_logger = utils.file_logger(logger_name, level=level)
        self.scheduler_logger.info("time,action,info")


    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, application):
        self._application = application

    def add_instance(self, instance):
        """
        Track instances at the scheduler level.
        Helps maintain the scheduler-specific view of instances. 
        """
        self.instances.append(instance)

    @abstractmethod
    def schedule(self, request, *args, **kwargs):
        """
        Main scheduler logic to assign request to instances.
        Called when a request is run.
        Creates a plan for the request.
        """
        raise NotImplementedError

    def request_arrival(self, request):
        """
        Handles the arrival of a new Request.
        """
        #print(f"Request {request} arrived at scheduler.")
        request.arrive_at_scheduler()
        self.pending_queue.append(request)
        if len(self.pending_queue) == 1:
            self.run_request(request)

    def request_completion(self, request):
        """
        Handles the completion of a Request.
        """
        request.complete_at_scheduler()
        self.executing_queue.remove(request)
        self.completed_queue.append(request)
        self.router.request_completion(request)

    def run_request(self, request):
        """
        Runs the Request by scheduling it and spawning an Executor.
        """
        request.run_on_executor()  # 下一步在executor上执行，记录排队时间
        # measure scheduling overhead
        start = time.time()
        self.schedule(request)
        end = time.time()
        self.scheduler_logger.debug('%s,sched_overhead,%s', clock(), end-start)
        self.spawn_executor(ExecutorType.CentralExecutor,
                            request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    def spawn_executor(self, executor_type, request):
        """
        Spawn an Executor for the request.
        Executors can logically execute anywhere.
        We don't model where they run in simulation.
        """
        executor = Executor.create(executor_type,
                                   request,
                                   self,
                                   self.executor_overheads)
        self.executors[request.request_id] = executor
        executor.run()

    def notify_busy_instance(self, instance):
        """
        Notify to the Scheduler that the instance is busy.
        """

    def notify_free_instance(self, instance):
        """
        Notify to the Scheduler that the instance is free.
        """

    def terminate_executor(self, executor):
        """
        Delete the Executor from the Scheduler.
        """
        del self.executors[executor.request.request_id]

    def save_all_request_metrics(self):
        """
        Saves start and end timestamps for all request nodes.
        Helpful for Gantt charts.
        """
        node_metrics = []
        for request in self.completed_queue:
            node_metrics.extend(request.get_all_node_metrics())
        node_metrics_df = pd.DataFrame(node_metrics)
        node_metrics_df.to_csv("request_nodes.csv", index=False)

    def get_results(self):
        """
        Returns results for all completed requests.   
        """
        array_results = {}

        request_ids = [r.request_id for r in self.completed_queue]
        array_results["request_ids"] = np.array(request_ids)

        response_times = [r.metrics.router_response_time for r in self.completed_queue]
        array_results["response_times"] = np.array(response_times)

        queue_times = [r.metrics.queue_time for r in self.completed_queue]
        array_results["queue_times"] = np.array(queue_times)

        ttft_times = [r.metrics.TTFT for r in self.completed_queue]
        array_results["ttft_times"] = np.array(ttft_times)

        tbt_times = [(r.metrics.router_response_time - r.metrics.TTFT) / (r.token_size)
                     for r in self.completed_queue]
        array_results["tbt_times"] = np.array(tbt_times)

        nth_token_overhead = [r.get_nth_token_overhead() for r in self.completed_queue]
        array_results["nth_token_overheads"] = np.array(nth_token_overhead)

        prompt_sizes = [r.prompt_size for r in self.completed_queue]
        array_results["prompt_sizes"] = np.array(prompt_sizes)

        token_sizes = [r.token_size for r in self.completed_queue]
        array_results["token_sizes"] = np.array(token_sizes)

        return array_results


class KVScheduler(Scheduler):
    """
    KVScheduler is a base class for Schedulers that ship KV caches.
    It does not implement the schedule method.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.prompt_processors = prompt_processors
        self.token_processors = token_processors
        self.prompt_instances = []
        self.token_instances = []

    def add_instance(self, instance):
        """
        Tracks prompt and token instances differently.
        NOTE: assumes instance tags are distinguishers, not h/w itself
        TODO: make this more flexible and robust
        """
        self.instances.append(instance)
        if instance.tag == "prompt":
            self.prompt_instances.append(instance)
        elif instance.tag == "token":
            self.token_instances.append(instance)
        else:
            # alternative way to distinguish instances
            if isinstance(self.prompt_processors, list):
                if instance.name in self.prompt_processors:
                    self.prompt_instances.append(instance)
                elif instance.name in self.token_processors:
                    self.token_instances.append(instance)
                else:
                    raise ValueError(f"Unsupported instance type: \
                                        {instance.processors[0].name}")

    def add_kv_cache_transfer(self, request, src_instance, dest_instance, bandwidth):
        """
        Convert prompt->token request to prompt->kvtransfer->token request
        by adding a flow node to the request graph.
        """
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # create new tasks and flows
        flow_size = request.estimate_kv_cache_size(
                                        num_tokens=prompt_task.prompt_size,
                                        model=src_instance.model)
        kv_transfer_flow = request.create_flow(FlowType.KVCacheTransfer,
                                               size=flow_size,
                                               src=src_instance,
                                               dest=dest_instance)
        kv_transfer_flow.notify = True

        # update request DAG
        request.flow_node = kv_transfer_flow
        request.dag.remove_edge(prompt_task, token_task)
        request.dag.add_edge(prompt_task, kv_transfer_flow)
        request.dag.add_edge(kv_transfer_flow, token_task)

        # assign tasks and flows to instances and links
        prompt_task.instance = src_instance
        token_task.instance = dest_instance
        # NOTE: simulate delay by adding a link of configurable bandwidth
        kv_transfer_flow.link = DummyLink(name="DummyLink",
                                          bandwidth=bandwidth)


class RandomScheduler(Scheduler):
    """
    RandomScheduler schedules Requests to Instances randomly.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to a random instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = np.random.choice(self.instances)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class RoundRobinScheduler(Scheduler):
    """
    RoundRobinScheduler schedules Requests in a round-robin fashion
    across all Instances.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to the next instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = self.instances[self.instance_index]
        self.instance_index = (self.instance_index + 1) % len(self.instances)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class JSQScheduler(Scheduler):
    """
    JSQScheduler schedules Requests to the Instance with smallest Request queue.
    Currently uses an inefficient O(n) search.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to the least loaded instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = min(self.instances,
                       key=lambda instance: len(instance.pending_requests))
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class TokenJSQScheduler(Scheduler):
    """
    JSQScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Currently uses an inefficient O(n) search.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request DAG to the instance with smallest queue
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = min(self.instances,
                       key=lambda instance: instance.sched_pending_tokens)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        # bookkeeping
        instance.sched_pending_tokens += prompt_task.prompt_size + 1


class KVRoundRobinScheduler(KVScheduler):
    """
    Schedules requests on prompt and token instances using round-robin.
    Prompt and token instances are not interchangeable.
    Always ships KV-caches from the prompt to the token instances.
    Does not overlap the KV-cache shipping with prompt computation.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []
        self.prompt_instance_index = 0
        self.token_instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        Assigns the prompt task to the next fast instance, and the token task
        to the next slow instance in a round-robin fashion.
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_instance = self.prompt_instances[self.prompt_instance_index]
        token_instance = self.token_instances[self.token_instance_index]
        self.prompt_instance_index = (self.prompt_instance_index + 1) % \
                                                len(self.prompt_instances)
        self.token_instance_index = (self.token_instance_index + 1) % \
                                                len(self.token_instances)

        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)


class KVJSQScheduler(KVScheduler):
    """
    KVJSQScheduler schedules Requests to the Instance with smallest queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []
        self.prompt_instance_index = 0
        self.token_instance_index = 0


    def pre_select_batch(self, instance, pred_task):
        # cpoy cur instance paras
        instance.add_to_pool(pred_task)
        pending_requests = instance.pending_requests
        old_batch = instance.batch
        memory = deepcopy(instance.memory)
        pending_tokens = deepcopy(instance.pending_tokens)
        max_memory = instance.max_memory
        max_batch_size = instance.max_batch_size
        max_batch_tokens = instance.max_batch_tokens

        new_batch = []
        new_tasks = []
        preempted_tasks = []

        batch_tokens = 0  # NOTE: NEW_ADDED
        for task in old_batch:
            new_batch.append(task)
            batch_tokens += task.tokens_per_iteration  # NOTE: NEW_ADDED
            
        for request in pending_requests:
            if len(new_batch) == max_batch_size:
                print("Reached max batch size")
                break
            # HACK: new_batch只会在>0时判断是否会超出，所以就算单个prompt大于max_batch_tokens也会加入task
            if len(new_batch) > 0 and batch_tokens + task.tokens_per_iteration > max_batch_tokens:      
                #print(f"Exceeded max batch tokens: batch_tokens={batch_tokens}, task.tokens_per_iteration={task.tokens_per_iteration}, max_batch_tokens={self.max_batch_tokens}")
                break
            
            task = instance.request_tasks[request][0]
            if task in old_batch:
                continue
            
            if task.state == NodeState.BLOCKED:
                new_batch.append(task)
                new_tasks.append(task)
                batch_tokens += task.tokens_per_iteration  # NOTE: NEW_ADDED
                continue
            
            if task.memory + memory <= max_memory:
                '''
                TODO: 这种情况会避开前面的检查, 暂时没有看出是为什么
                Added task: 880, batch_tokens now: 880
                Added task: 7373, batch_tokens now: 8253
                Exceeded max batch tokens: batch_tokens=8253, task.tokens_per_iteration=7373, max_batch_tokens=2048
                total batch tokens: 8253, total tasks in batch: 2
                '''
                if batch_tokens + task.tokens_per_iteration > max_batch_tokens and len(new_batch) > 0:  # FIXME: 这里有bug
                    #print(f"{self.tag} Instance {self.instance_id} Exceeded max batch tokens when considering task: batch_tokens={batch_tokens}, task.tokens_per_iteration={task.tokens_per_iteration}, max_batch_tokens={self.max_batch_tokens}, len(batch): {len(new_batch)}")
                    break
                new_batch.append(task)
                new_tasks.append(task)
                memory += task.memory
                batch_tokens += task.tokens_per_iteration  # NOTE: NEW_ADDED
                #print(f"Added task: {task.tokens_per_iteration}, batch_tokens now: {batch_tokens}, len(batch): {len(new_batch)}")
            else:
                print("Exceeded max memory")
                break
        instance.remove_from_pool(pred_task)
        # ORCA Instance 不支持抢占
        assert len(preempted_tasks) == 0
        #print(f'total batch tokens: {batch_tokens}, total tasks in batch: {len(new_batch)}')
        return preempted_tasks, new_tasks  
        
    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by predicted executing time & queue length)
        """
        # if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
        #     raise ValueError("No instances available")

        # prompt_task = request.root_node
        # token_task = next(request.successors(prompt_task))

        # prompt_instance = min(self.prompt_instances,
        #                       key=lambda instance: len(instance.pending_requests))
        # token_instance = min(self.token_instances,
        #                      key=lambda instance: len(instance.pending_requests))

        # # ship KV-cache between instances
        # self.add_kv_cache_transfer(request,
        #                            prompt_instance,
        #                            token_instance,
        #                            self.transfer_bandwidth)
        if (len(self.prompt_instances) == 0 or len(self.token_instances) == 0):
            raise ValueError("No instances available")
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # 调用底层的静态分析器获取预估执行时间最短的 Prompt Instance
        prompt_instance = None
        min_prefill_time = sys.maxsize
        for instance in random.sample(self.prompt_instances, len(self.prompt_instances)):  # 遍历每个 Prompt Instance
            if instance.pending_tokens == 0:
                prompt_instance = instance
                break
            # 首先要把task的instance指定为该instance否则执行不了select_batch()
            prompt_task.instance = instance  # 后面不用取消，因为add_kv_cache_transfer会重新指定
            _, new_tasks = self.pre_select_batch(instance, prompt_task) #super(SplitwiseInstance, instance).select_batch()  # 改成显式调用OCRAInstance的
            if prompt_task in new_tasks:
                exec_time = get_static_mapper_duration(new_tasks, instance)
                end_time = time.time()
                if exec_time < min_prefill_time:
                    prompt_instance = instance
                    min_prefill_time = exec_time
            prompt_task.instance = None
        # 如果下个组的batch里面没有该task，那说明该Instance已经超过了max_batch_tokens, 那执行时间肯定不是最短           
        if prompt_instance == None:  # HACK: 在所有的 Prompt Instance里找不到能在下个batch执行的
             prompt_instance = min(self.prompt_instances,  # 选择待执行token数最短的
                              key=lambda instance: instance.pending_tokens)    
                          
        # decode 同理
        token_instance = None       
        min_decode_time = sys.maxsize
        for instance in random.sample(self.token_instances, len(self.token_instances)):
            if instance.pending_tokens == 0:
                token_instance = instance
                break
            # 首先要把task的instance指定为该instance否则执行不了select_batch()
            token_task.instance = instance  # 后面不用取消，因为add_kv_cache_transfer会重新指定
            _, new_tasks = self.pre_select_batch(instance, token_task)#super(SplitwiseInstance, instance).select_batch()  # 改成显式调用OCRAInstance的
            # #ipdb.set_trace()
            if token_task in new_tasks:
                exec_time = get_static_mapper_duration(new_tasks, instance)
                if exec_time < min_decode_time:
                    token_instance = instance
                    min_decode_time = exec_time
            token_task.instance = None
        # 如果下个组的batch里面没有该task，那说明该Instance已经超过了max_batch_tokens, 那执行时间肯定不是最短
        if token_instance == None:  # HACK: 在所有的 Token Instance 里找不到能在下个batch执行的
            token_instance =  min(self.token_instances,  # 选择待执行token数最短的
                            key=lambda instance: instance.pending_tokens)
        # task的instance在add_kv_cache_transfer里面赋值的
        assert prompt_instance != token_instance
        self.add_kv_cache_transfer(request,
                                       prompt_instance,
                                       token_instance,
                                       self.transfer_bandwidth)
        #bookkeeping
        #ipdb.set_trace()
        prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
        token_instance.sched_memory += prompt_task.max_memory(token_instance) + \
                                        token_task.max_memory(token_instance)
                                        
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1


class OverlapKVJSQScheduler(KVJSQScheduler):
    """
    Same as KVJSQScheduler, but overlaps the KV-shipping with prompt.
    Always ships KV-caches from the prompt to the token instances.
    Simulates 90% overlap by using 10x the interconnect bandwidth.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         transfer_bandwidth * 10,
                         debug)


class KVTokenJSQScheduler(KVScheduler):
    """
    KVTokenJSQScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        prompt_instance = min(self.prompt_instances,
                              key=lambda instance: instance.sched_pending_tokens)
        token_instance = min(self.token_instances,
                             key=lambda instance: instance.sched_pending_tokens)

        # ship KV-cache between instances
        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)


class OverlapKVTokenJSQScheduler(KVTokenJSQScheduler):
    """
    Same as KVTokenJSQScheduler, but overlaps the KV-shipping with prompt.
    Simulates 90% overlap by using 10x the interconnect bandwidth.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         transfer_bandwidth * 10,
                         debug)


class MixedPoolScheduler(KVScheduler):
    """
    MixedPoolScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 prompt_max_pending_batch_tokens,
                 token_max_pending_batch_tokens,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.prompt_max_pending_batch_tokens = prompt_max_pending_batch_tokens
        self.token_max_pending_batch_tokens = token_max_pending_batch_tokens
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.mixed_instances = []
        self.token_instances = []

    def is_memory_loaded(self, instance, tasks):
        """
        Check if instance is loaded by task
        """
        request_memory = sum(task.max_memory(instance) for task in tasks)
        if instance.sched_memory + request_memory >= instance.max_memory:
            return True
        return False

    def is_queue_long(self, instance, task):
        """
        Check if prompt queue is long
        """
        if len(instance.pending_queue) > 0 and \
            instance.sched_pending_tokens + task.tokens_per_iteration > \
                self.prompt_max_pending_batch_tokens:
            return True
        return False

    def find_best_prompt_instance(self, instances, prompt_task):
        """
        Check if prompt queue is long
        """
        if len(instances) == 0:
            return None
        prompt_instance = min(instances,
                              key=lambda instance: instance.sched_pending_tokens)
        if self.is_queue_long(prompt_instance, prompt_task):
            return None
        return prompt_instance

    def find_best_token_instance(self, instances, prompt_task, token_task):
        """
        Checks if instance memory is full
        """
        if len(instances) == 0:
            return None
        token_instance = min(instances,
                             key=lambda instance: (instance.sched_memory))
        if self.is_memory_loaded(token_instance, [prompt_task, token_task]):
            return None
        return token_instance

    def notify_free_instance(self, instance):
        """
        Notifies that a mixed instance is free; moves it to appropriate pool
        """
        if instance.sched_tag == "mixed":
            instance.sched_tag = None
            self.mixed_instances.remove(instance)
            if instance.tag == "prompt":
                self.prompt_instances.append(instance)
            elif instance.tag == "token":
                self.token_instances.append(instance)
            else:
                raise ValueError(f"Unsupported instance tag: {instance.tag} on \
                    {instance.name}_{instance.instance_id}")

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        """
        if (len(self.prompt_instances) == 0 or len(self.token_instances) == 0) \
            and len(self.mixed_instances) == 0:
            raise ValueError("No instances available")
        #ipdb.set_trace()
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # 先在prefill池再在mixed池中找pending queue最短的(如果最短的也很长则返回None)
        prompt_instance = None
        for instances in [self.prompt_instances, self.mixed_instances]:  
            prompt_instance = self.find_best_prompt_instance(instances, prompt_task)  # JSQ原则
            if prompt_instance is not None: # 在prefill池找到了就不用在mixed池里找了
                break
        # 找被分配内存的最短的? self.model.size.total_size  要检查是否会OOM           
        token_instance = None
        for instances in [self.token_instances, self.mixed_instances]:
            token_instance = self.find_best_token_instance(instances, prompt_task, token_task)
            if token_instance is not None: # 在token池找到了就不用在mixed池里找了
                break
        # 如果找不到合适的prefill instance并且存在token instance            
        if prompt_instance is None and len(self.token_instances) > 0:
            # 选择“pending_queue”最短的token instance-->mixed instance
            prompt_instance = min(self.token_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            self.token_instances.remove(prompt_instance)
            self.mixed_instances.append(prompt_instance)
            prompt_instance.sched_tag = "mixed"

        if token_instance is None and len(self.prompt_instances) > 0:
            # 选择“sched_memory”最少的prefill instance-->mixed instance
            token_instance = min(self.prompt_instances,
                                 key=lambda instance: (instance.sched_memory))
            self.prompt_instances.remove(token_instance)
            self.mixed_instances.append(token_instance)
            token_instance.sched_tag = "mixed"

        # 还找不到的话就在所有池子里找pending queue最短的
        if prompt_instance is None or token_instance is None:
            all_instances = self.prompt_instances + self.mixed_instances + self.token_instances
            prompt_instance = min(all_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            token_instance = prompt_instance
        # prompt和token池不一样时，kv cache transfer
        if prompt_instance != token_instance:
            # ship KV-cache between instances prompt task-->kv transfer-->token task
            self.add_kv_cache_transfer(request,
                                       prompt_instance,
                                       token_instance,
                                       self.transfer_bandwidth)
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
            token_instance.sched_memory += prompt_task.max_memory(token_instance) + \
                                           token_task.max_memory(token_instance)
        # 有可能刚好找不到token_instance时把当前prompt_instance变成mixed
        else: 
            # run on same instance
            prompt_task.instance = prompt_instance
            token_task.instance = token_instance
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance) + \
                                            token_task.max_memory(prompt_instance)
            prompt_task.chain = [token_task]

        # bookkeeping
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1


                