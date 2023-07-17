import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest="width", action="store")
parser.add_argument(dest="height", action="store")
parser.add_argument("--frames", dest="frames", action="store")
parser.add_argument("--qp", dest="qp", action="store")
parser.add_argument("--configure", dest="configure", action="store")

args = parser.parse_args(["64", "56", "--frames", "60", "--qp", "1", "--configure", "AI"])
print(args.width, args.height, args.frames, args.qp, args.configure)