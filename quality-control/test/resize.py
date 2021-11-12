import os
import glob
from PIL import Image

for i, f in enumerate(glob.glob('out/raw/*.jpg')):
    # fname = f.split(os.sep)[-1]
    img = Image.open(f)
    img = img.rotate(-90)
    img = img.resize((240, 240))
    # img.save('out/resized/'+fname)
    img.save('out/resized/{0}.jpg'.format(i))
