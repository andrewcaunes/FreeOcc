import logging
import os

import yaml


CONFIG_CACHE = {}


def get_configs_root():
    return os.path.dirname(os.path.abspath(__file__))


def load_yaml_file(config_path):
    if config_path not in CONFIG_CACHE:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            CONFIG_CACHE[config_path] = yaml.safe_load(file_handle)
    return CONFIG_CACHE[config_path]


def load_yaml_config(*path_parts):
    config_path = os.path.join(get_configs_root(), *path_parts)
    return load_yaml_file(config_path)


def load_core_defaults():
    return load_yaml_config("core_defaults.yaml")


def load_prompt_config(prompt_classes_filename):
    prompt_path = os.path.join(get_configs_root(), "prompts", prompt_classes_filename)
    if not os.path.exists(prompt_path):
        legacy_path = os.path.join(get_configs_root(), "classes", prompt_classes_filename)
        if os.path.exists(legacy_path):
            logging.warning(
                "Prompt config %s found in deprecated folder configs/classes. Move it to configs/prompts.",
                prompt_classes_filename,
            )
            prompt_path = legacy_path
        else:
            raise FileNotFoundError(
                f"Could not find prompt config '{prompt_classes_filename}' in configs/prompts or configs/classes"
            )
    return load_yaml_file(prompt_path)


def load_prompt_classes(prompt_classes_filename):
    prompt_config = load_prompt_config(prompt_classes_filename)
    return list(prompt_config["class_names"])


def parse_float_config_value(config_name, key_name, raw_value):
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    value_string = str(raw_value).strip()
    if value_string.endswith(","):
        value_string = value_string[:-1].strip()
    try:
        return float(value_string)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value for key '{key_name}' in {config_name}: {raw_value}"
        ) from error


def load_conf_thresh_per_class(conf_thresh_per_class_filename):
    direct_path = conf_thresh_per_class_filename
    config_path = direct_path
    if not os.path.exists(config_path):
        config_path = os.path.join(
            get_configs_root(),
            "depth_thresh",
            conf_thresh_per_class_filename,
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Could not find conf-threshold config "
                f"'{conf_thresh_per_class_filename}' as a direct path or in configs/depth_thresh"
            )

    conf_thresh_per_class = load_yaml_file(config_path)
    if not isinstance(conf_thresh_per_class, dict):
        raise ValueError(
            f"Expected a key/value mapping in {config_path}, got {type(conf_thresh_per_class).__name__}"
        )

    parsed_conf_thresh_per_class = {}
    for class_name, conf_thresh_value in conf_thresh_per_class.items():
        parsed_conf_thresh_per_class[str(class_name)] = parse_float_config_value(
            os.path.basename(config_path),
            class_name,
            conf_thresh_value,
        )
    return parsed_conf_thresh_per_class


def load_prompt_dynamic_class_names(prompt_classes_filename):
    prompt_config = load_prompt_config(prompt_classes_filename)
    dynamic_class_names = prompt_config.get("dynamic_class_names")
    if dynamic_class_names is None:
        dynamic_class_names = prompt_config.get("dynamic_prompt_class_names")
    if dynamic_class_names is None:
        return None
    return list(dynamic_class_names)


def get_default_dynamic_prompt_classes():
    return list(load_core_defaults()["dynamic_prompt_class_names"])


def get_thing_prompt_class_names():
    return list(load_core_defaults()["thing_stuff"]["thing_prompt_class_names"])


def get_priority_object_classes():
    return list(load_core_defaults()["occupancy"]["priority_object_classes"])


def get_thin_object_classes():
    return list(load_core_defaults()["occupancy"]["thin_object_classes"])


def get_driveable_class_preference():
    return list(load_core_defaults()["occupancy"]["driveable_class_preference"])


def get_occ3d_nuscenes_class_names():
    return list(load_core_defaults()["occ3d_nuscenes"]["class_names"])


def get_occ3d_nuscenes_class_mapping():
    class_names = get_occ3d_nuscenes_class_names()
    class_mapping = {class_name: class_name for class_name in class_names}
    class_mapping.update(load_core_defaults()["occ3d_nuscenes"]["class_mapping_updates"])
    return class_mapping


def get_default_dynamic_classes():
    return get_default_dynamic_prompt_classes()


def get_panoptic_thing_class_names():
    return get_thing_prompt_class_names()
