import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

output='D:/NowWorking/newProjet/results/images_alte/'

background= Image.open(output+'dom_canRCM4ARC_large2.png')
overlay  = Image.open(output+'dom_CRCM5NA_large2.png')

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.25)
new_img.save(output+'dom_NA_ARC_F.png',"PNG")


fig = plt.figure(figsize=new_img.size)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

ax.imshow(new_img, interpolation='none')

fig.savefig(output+'dom_NA_ARC.svg', format='svg', dpi=300)


#.svg gave me high-res pictures that actually looked like my graph.
#I used 1200 dpi because a lot of scientific journals require images in 1200.
# Convert to desired format in GiMP or Inkscape after.