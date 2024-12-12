from squat_detection import detect_squat
from pose_analysis import analyze_video
import os



def main():
    # Ask the user for the video file path
    video_path = input("please input path to video or drag video into terminal: ").strip()

    # if ' ' in video_path:
    #     print("Error: Ensure your video file name does not contain spaces.")
    #     return

    # # Ensure the file path exists
    # if not os.path.exists(video_path):
    #     print("Error: The video path does not exist.")
    #     return

    # Perform squat detection
    squat_percent = detect_squat(video_path)

    # Print result of squat detection
    if squat_percent > 0.9:
        print(f"Squat detected with a probability of {squat_percent}%")
        
        # Proceed to pose analysis if squat is detected
        analyze_video(video_path)
    else:
        print("This video does not contain a squat.")

if __name__ == "__main__":
    main()
