import utils
import sys

a = int(sys.argv[1])
b = int(sys.argv[2])
frames = []
for i in range(a, b+1):
    frames.append("eval_imgs/{}.png".format(i))
utils.make_gif(frames, "val-set-{}-{}.gif".format(a, b))
