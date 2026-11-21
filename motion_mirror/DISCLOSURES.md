# Open-source disclosures

All pipeline, environment, reward, retargeting, training, and agent code in this repo was written during the hackathon.

Third-party assets and libraries used:
- **Unitree H2 MJCF model** (`assets/unitree_h2/`): from unitreerobotics/unitree_mujoco (BSD-3-Clause, Unitree Robotics). The default robot for the stage-1 replay.
- **MediaPipe Pose Landmarker model** (`assets/mediapipe/pose_landmarker_full.task`): Google's pretrained body-landmark model, downloaded from Google's model storage, used to track the person's body in the video.
- **Unitree G1 MJCF model** (`assets/unitree_g1/`): from google-deepmind/mujoco_menagerie; copyright Unitree Robotics, see its LICENSE.
- **Libraries:** MuJoCo, PyTorch, NumPy, OpenCV, MediaPipe (pose keypoints), google-genai (Gemini API), imageio.
- **Research only (no code copied):** HumanPlus, OmniH2O, ASAP, PHC, GMR.
