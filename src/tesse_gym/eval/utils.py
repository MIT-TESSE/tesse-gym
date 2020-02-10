""" Evaluation utilities for finding subclasses.
Taken from habitat-challenge baselines.agents.simple_agents.
"""


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_agent_cls(agent_class_name, base_cls):
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(base_cls)
        if sub_class.__name__ == agent_class_name
    ]
    return sub_classes[0]
