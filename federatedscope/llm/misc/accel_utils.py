import os

def _env_world_size_gt1():
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return True
    except:
        pass
    try:
        if int(os.environ.get("LOCAL_RANK", "-1")) >= 0:
            return True
    except:
        pass
    return False

def in_distributed_mode() -> bool:
    try:
        from accelerate.state import AcceleratorState
        state = AcceleratorState()
        dt = getattr(state, "distributed_type", "NO")
        name = dt.name if hasattr(dt, "name") else str(dt)
        return name != "NO"
    except Exception:
        return _env_world_size_gt1()

def allow_device_map_auto_by_env() -> bool:
    return os.environ.get("USE_DEVICE_MAP_AUTO", "1") == "1"

def should_use_device_map_auto() -> bool:
    return (not in_distributed_mode()) and allow_device_map_auto_by_env()
