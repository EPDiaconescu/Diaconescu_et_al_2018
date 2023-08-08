
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fileName='MSESS_EMreanalPerformanceD_ALL_ARCcordexB_1980to2004_spatial_'
img1='D:/NowWorking/newProjet/results/domARC_CORDEX_b/performaneD_ref_Final/'+fileName+'A1.png'
img3='D:/NowWorking/newProjet/results/domARC_CORDEX_b/performaneD_ref_Final/'+fileName+'C.png'

output='D:/NowWorking/newProjet/results/domARC_CORDEX_b/performaneD_ref_Final/'

p1=Image.open(img1)
p3=Image.open(img3)


width1, height1 = p1.size
width3, height3 = p3.size

out = Image.new('RGBA', (width1, height1+height3), color='white')

out.paste(p3, (290, 0))
out.paste(p1, (0, height3))



out.save(output+fileName+'_final2.png', dpi=(300.0, 300.0))
#out.savefig(output+fileName+'B_final.svg', format='svg', dpi=1200)


print ('OK')
#.svg gave me high-res pictures that actually looked like my graph.
#I used 1200 dpi because a lot of scientific journals require images in 1200.
# Convert to desired format in GiMP or Inkscape after.