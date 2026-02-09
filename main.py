import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()
    processors['recognition'] = import_class('processor.recognition_rgb.REC_Processor')
    processors['recognition_rgb_only'] = import_class('processor.recognition_rgb_only.REC_Processor')
    processors['recognition_fusion'] = import_class('processor.recognition_fusion.REC_Processor')
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])
    arg = parser.parse_args()
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    p.start()
