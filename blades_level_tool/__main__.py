from blades_level_tool import cli
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("gui")
    sys.exit(cli.main())
