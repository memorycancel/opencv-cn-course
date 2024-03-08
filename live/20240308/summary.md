OpenCV Live!

Phil Nelson and Satya Malik discussed the latest project from the OpenCV AI competition winner, Barlow, which involves using robotics to monitor and care for plants. Dr. Brent Nelson added his thoughts on the topic, highlighting the potential applications of this technology in various industries. Abdullah, Steve, Gianluca, and Frazer discussed OpenCV's fundraising and community engagement efforts, including the tax deductibility of donations in the US. Speakers 6 and 7 discussed their experiences in designing and building robots for environmental monitoring, while Speakers 8 and 2 discussed the difficulties in developing a vision system that can adapt to changing conditions. Speaker 9 integrated computer vision and machine learning for autonomous navigation.

Transcript

https://otter.ai/u/p760SQN2j5q0bcOZqcVsHN1S5F0?view=transcript

Action Items
[ ] Capture new image dataset with winter/dead plants
[ ] Train new computer vision models on updated dataset
[ ] Integrate ROS into the robot
[ ] Explore LIDAR mapping capabilities for the next version
[ ] Add watering capabilities to robot
[ ] Present project at next OpenCV Live show
Outline
AI project for plant care and robotics.
Competition winners from Open CV AI 2023 showcase their projects, including a water-based robot.
Satya Malik, CEO of Open CV, won the grand prize in the Open CV AI competition 2023.
Phil Nelson, director of content and creative at Open CV, produces the show and hosts the giveaway.
Speaker 1 invites listeners to donate to OpenCV, highlighting tax deductions and membership perks.
Speaker 2 corrects Speaker 1's mistake about tax deductions, reminding listeners to donate by December 31 for the previous year's taxes.
A team's project and its evolution.
Abdullah introduces the Barrel team, explaining their roles in the Fraser project.
Company designs and manufactures machines for packaging industry, hosts open-source projects, and competes in hackathons to improve skills.
Using AI and robotics for agriculture.
Neuron network trained on plant images recognized plants in video, with acid camera displaying results.
The team struggled with designing a robot that could navigate through a barnyard and recognize funds while avoiding obstacles.
Speaker 6 explains how the team chose the mini Hopper version of the robotic kit for their product, citing the expertise of their colleague Eric in the technology.
Speaker 6 highlights the open-source components used in the system's architecture, including Capella and ROS, and shows diagrams of the system's components and logic and physical architectures.
Robotics, sensors, and machine learning.
Speaker 7, a CO and robotics expert, shared their process for creating robots, emphasizing the importance of understanding the cost of components and the trade-off between size and strength.
They discussed the concept of "squaring the cube," or the idea that as robots get bigger, they become weaker, and the cost of components increases, making it difficult to scale up without sacrificing performance.
Speaker 7 explains why they chose a specific robotic platform for their project, citing factors such as capabilities, complexity, and open-source integration.
Speaker 7 discusses using Ross shooter for mapping with LIDAR, showing a live demo of a lighter mark on a robot and a map being built in real time.
Speaker 3 highlights the complexity of integrating Ross shooter, but notes that it's already working out of the box for mapping with LIDAR.
Machine learning project for plant recognition.
Team aims to create a navigation system for a robot to walk through a barnyard, recognizing plants and leaves for watering.
Speaker 8 discusses the importance of keeping the main concept of a project while adapting to changes in weather conditions.
Speaker 3 explains how the team pivoted to a tomato field nearby, which was a close-up robot and a good opportunity for the demo.
Vision AI for robot navigation using Docker and YOLO.
Speaker 9 discussed the vision controller in the robot application, which relies on the LT light camera and YOLO model for fast plant classification and transmission to the robot.
The vision controller is separated from the motion controller using Docker, allowing for independent improvements and communication between the two components.
Speaker 9 explains that the Ultralight can perform inference at more than 30 frames per second, despite being a small computer, and that they decided to cap it at 30 frames per second for stability and consistent results.
Speaker 3 mentions that there is interference with the USB 3.0 port on the Raspberry Pi, making it harder to use the collector with the 3D printer they have designed, but they are testing a new version with improvements.
OpenCV AI competition with trivia giveaway.
Drew won the Open CV AI competition trivia question by answering correctly in 2020.
