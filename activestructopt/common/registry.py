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
from pathlib import Path

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

def _import_local_file(path: Path, *, project_root: Path):
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "activestructopt" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.resolve()

    module_name = ".".join(
        path.absolute().relative_to(project_root.absolute()).with_suffix("").parts
    )
    # logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


def _get_project_root():
    """
    Gets the root folder of the project (the parent of the "activestructopt" folder)
    :return: The absolute path to the project root.
    """
    root_folder = Path(__file__).parent
    while root_folder.stem != 'activestructopt':
        root_folder = root_folder.parent
    return root_folder.parent

# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports():
    project_root = _get_project_root()

    import_keys = ["dataset", "model", "objective", "optimizer", "simulation"]
    for key in import_keys:
        dir_list = (project_root / "activestructopt" / key).rglob("*.py")
        for f in dir_list:
            _import_local_file(f, project_root = project_root)


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
        #assert name.count(".") >= 1

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
