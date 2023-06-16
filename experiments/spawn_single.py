import os
import sys

def run(runinfo):
    path = "python"
    args = [
        "python",
        "train.py",
    ]

    for field, value in runinfo.items():
        if field == "config":
            args.append(value)
        else:
            args.append(f"--{field}")
            if value is not None:
                args.append(str(value))

    print("Running:")
    print(" ".join(args))
    print("")
    sys.stdout.flush()
    os.execvp(path, args)
