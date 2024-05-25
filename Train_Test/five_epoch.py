import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for i in range(56,61):
#i=56
    print(i)
    process= subprocess.Popen(["python","./Semantic_Segmentation/Common_Codes/Train_Test/predict.py",str(i)],stdout=subprocess.PIPE)
    stdout, stderr=process.communicate()
    #print(stdout)