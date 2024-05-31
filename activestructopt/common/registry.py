"""
Borrowed from MatDeepLearn, 
    which borrowed this from https://github.com/Open-Catalyst-Project.

Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.

Import the global registry object using

``from activestructopt.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a model: ``@registry.register_model``
"""
import importlib


def _get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `matdeeplearn.tasks.base_task.BaseTask`
    # we can use importlib to get the module 
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"Could not import module {module_name=} for import {name=}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class {class_name=} from module {module_name=}"
        ) from e


class Registry:
    r"""Class for registry object which acts as central source of truth."""
    mapping = {
        # Mappings to respective classes.
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "optimizer_name_mapping": {},
        "objective_name_mapping": {},
        "simulation_name_mapping": {},
    }

    @classmethod
    def register_dataset(cls, name):
        def wrap(func):
            cls.mapping["dataset_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):
            cls.mapping["optimizer_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_objective(cls, name):
        def wrap(func):
            cls.mapping["objective_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_simulation(cls, name):
        def wrap(func):
            cls.mapping["simulation_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}`
        assert name.count(".") >= 1

        return _get_absolute_mapping(name)

    @classmethod
    def get_dataset_class(cls, name):
        return cls.get_class(name, "dataset_name_mapping")

    @classmethod
    def get_model_class(cls, name):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.get_class(name, "optimizer_name_mapping")

    @classmethod
    def get_objective_class(cls, name):
        return cls.get_class(name, "objective_name_mapping")

    @classmethod
    def get_simulation_class(cls, name):
        return cls.get_class(name, "simulation_name_mapping")

registry = Registry()
