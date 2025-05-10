import os


def check_path(path):
    """Check if a path exists and print info about it."""
    print(f"Checking: {path}")

    if os.path.exists(path):
        if os.path.isdir(path):
            print(f"✓ Directory exists: {path}")
            print(f"  Contents: {os.listdir(path)}")
        elif os.path.isfile(path):
            print(f"✓ File exists: {path}")
            print(f"  Size: {os.path.getsize(path)} bytes")
    else:
        print(f"✗ Path does not exist: {path}")

        # Check parent directory
        parent = os.path.dirname(path)
        if os.path.exists(parent):
            print(f"  Parent directory exists: {parent}")
            print(f"  Contents of parent: {os.listdir(parent)}")
        else:
            print(f"  Parent directory does not exist: {parent}")


# Current directory
print("Current working directory:", os.getcwd())

# Check various paths
check_path("data")
check_path("data/raw")
check_path(r"D:\Pyty\data\raw\dataset.csv")
check_path("data/processed")

# Check if any CSV files exist in the current directory or subdirectories
print("\nSearching for CSV files in current directory and immediate subdirectories:")
for root, dirs, files in os.walk(".", topdown=True, followlinks=False):
    # Limit depth to immediate subdirectories
    if root.count(os.sep) <= 1:
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                print(f"Found CSV: {path}")

    # Don't go deeper than immediate subdirectories
    dirs[:] = [d for d in dirs if os.path.join(root, d).count(os.sep) <= 1]