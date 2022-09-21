#!/usr/bin/env python

from spt3g import core, maps
import sys

g3file = sys.argv[1]

g3 = core.G3File(g3file)
for frame in g3:

    if frame.type != core.G3FrameType.Map:
        continue
    print(frame['T'].__dict__)

    print(dir(frame['T']))
    for att in dir(frame['T']):
        try:
            print(f"{att}: {getattr(frame['T'],att)}")
        except:
            print("We can't")

    print("---------")
    print(frame['T'].x_len)
    print(frame.y_len)

    print(frame['T'].shape)
    print(frame['T'].res)
    print(frame['T'].weighted)
    print(frame['T'].alpha_center)
    print(frame['T'].delta_center)
    print(frame['T'].proj)

exit()
