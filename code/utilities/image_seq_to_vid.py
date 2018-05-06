import numpy as np
import cv2
from skvideo.io import FFmpegWriter 

def images_seq_to_video(input_folder, outfile, count):
    writer = FFmpegWriter(outfile, outputdict={'-r': fps})
    writer = FFmpegWriter(outfile)
    
    for i in range(count):
        image = outfile + 'frame'+ str(i)+'.jpg'
        f = cv2.imread(image)
        writer.writeFrame(f)
    writer.close()