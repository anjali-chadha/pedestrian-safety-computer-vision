import cv2
import os
from skvideo.io import FFmpegWriter
import glob

# Modify the below variables in order to create frames from video
dir_path = 'LLNet_Frames'
images = []
output = 'video_2.mp4'

for f in glob.glob(dir_path + "/*.png"):
	images.append(f)

images.sort()
# Define the codec and create VideoWriter object
writer = FFmpegWriter(output)

print (images)
for i in range(len(images)):
	#image_path = dir_path + "/" + image
	# Modify below based on file names
	frame = cv2.imread(dir_path + '/LLnet_inference_frame' + str(i) + '_out.png')
	writer.writeFrame(frame) # Write out frame to video
# Release everything if job is finished
writer.close()
print("The output video is {}".format(output))
