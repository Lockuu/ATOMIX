from pathlib import Path


def check_config():
    config_path = Path(__file__).parent / "model_config.yaml"
    print(f"Config path: {config_path}")
    print(f"Exists: {config_path.exists()}")


if __name__ == "__main__":
    check_config()