# Program to export Openpose files from videos
# Author.: Murilo
# Date...: 09/28/2018

import glob, os

videosDir = "/home/murilo/dataset/KTH/VideosTrainValidationTest"
os.chdir("/home/murilo/openpose")


def main():
    for file in glob.glob(os.path.join(videosDir, "**/*.avi"), recursive=True):
        baseVideoName = os.path.splitext(os.path.basename(file))[0]
        print(file)
        '''
      newFileName = file.replace("VideosTrainValidationTest", "VideosTrainValidationTestOP")	
      print(newFileName)

      newFilePath =  os.path.dirname(os.path.abspath(newFileName))	
      if not os.path.exists(newFilePath):
         os.makedirs(newFilePath)

      os.system("build/examples/openpose/openpose.bin --display 0 --video "+file +" --write_video " + newFileName)
      '''
        # here generate skeletons with black background
        newFileName = file.replace("VideosTrainValidationTest", "VideosTrainValidationTestOPBlack")
        print(newFileName)

        newFilePath = os.path.dirname(os.path.abspath(newFileName))
        if not os.path.exists(newFilePath):
            os.makedirs(newFilePath)

        keysPath = file[:-4]
        keysPath = keysPath.replace("VideosTrainValidationTest", "VideosTrainValidationTestOPKeys")
        print(keysPath)
        if not os.path.exists(keysPath):
            os.makedirs(keysPath)

        framesPath = file[:-4]
        framesPath = framesPath.replace("VideosTrainValidationTest", "VideosTrainValidationTestOPFrames")
        print(framesPath)
        if not os.path.exists(framesPath):
            os.makedirs(framesPath)

        os.system(
            "build/examples/openpose/openpose.bin --display 0 --video " + file + " --write_video " + newFileName + " --write_images " + framesPath + " --write_json " + keysPath + " --disable_blending")


if __name__ == "__main__":
    main()
