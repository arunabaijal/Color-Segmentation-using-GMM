# Color-Segmentation-using-GMM
Implementing color segmentation using Gaussian Mixture Models and Expectation Maximization techniques.

## System Requiremnets
`python3`<br>
`OpenCV (4.2.0)` <br>

## Run instructions
Navigate to folder containing GMM.py and run<br>
`python3 GMM.py --video=videopath`<br>
The default value for the videopath is `detectbuoy.avi`
The program will read the video into frames and store the frames in the folder `Frames\`.<br>
After program execution, the output frames will be generated in the folder `Data\Output\Frames`
and resultant video will be saved as `Result.mp4`.
