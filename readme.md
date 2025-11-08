***Title: ***
StrumSense: Real-Time Guitar Chord Recognition & Accuracy Feedback System

***Summary:***
StrumSense is an intelligent system that detects guitar chords in real time by combining computer vision (hand posture recognition using MediaPipe/OpenCV) with audio analysis (spectrogram-based chord classification). It not only identifies the chord being played but also validates accuracy, providing instant feedback to help musicians correct their technique.
The system supports multiple angles and lighting conditions, adapts to different users’ playing styles, and includes a visual feedback dashboard showing performance stats, accuracy history, and improvement suggestions. 

***Problem statement:***
Learning guitar chords requires consistent posture, correct finger placement, and accurate strumming patterns. Traditional learning methods rely heavily on in-person feedback from teachers or slow self-assessment via recordings. Beginners often repeat mistakes without realizing it, leading to poor technique. There is no widely accessible tool that offers real-time, AI-driven feedback on both hand positioning and sound quality of guitar chords.

***Objectives:***
1.Real-Time Gesture Recognition-Use MediaPipe & OpenCV to detect hand landmarks and classify chord shapes.
2.Audio-Based Chord Verification-Process live audio input to create spectrograms and classify chord sound.
3.Cross-Verification for Accuracy-Match visual chord detection with audio recognition for final decision.
4.Instant Feedback Loop-Display feedback on whether the chord was played correctly, with tips.
5.Performance Dashboard-Show accuracy percentage, improvement over time, and common errors.
6.Robustness & Adaptability-Handle multiple lighting conditions, different users, and guitar types.

***Outcomes:***
A working real-time system that detects and verifies guitar chords with >90% accuracy for supported chords.
A user interface that shows chord name, accuracy score, and improvement suggestions instantly.
A learning tool that helps beginners practice without requiring constant teacher supervision.
Potential extension into a mobile or web app for broader accessibility.


    ├── feature_utils.py         # landmark normalization, spectrogram extraction, etc.
    ├── dataset_utils.py
    └── config.py                # global paths, chord list


