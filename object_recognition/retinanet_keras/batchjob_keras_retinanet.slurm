#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobRetinanet1       #Set the job name to "JobExample1"
#SBATCH --time=00:15:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                   #Request 1 task
#SBATCH --mem=16G                     #Request 2560MB (2.5GB) per node
#SBATCH --ntasks-per-node=1          #Request 8 tasks/cores per node
#SBATCH --output=batch.out      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=puneetkohli@tamu.edu    #Send all emails to email_address

#First Executable Line
module load CUDA/9.0.176
module load OpenCV/3.3.0-foss-2017b-Python-3.6.3
module unload Python
module load Anaconda
module load cuDNN/7.0.5-CUDA-9.0.176
source activate puneet_keras_env
python retinanet_resnet50_obj_detect_video.py