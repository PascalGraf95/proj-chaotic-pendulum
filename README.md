## Chaotic pendulum prediction

## Install

Pip install the requirements in a
[**3.11>=Python>=3.7**](https://www.python.org/) environment.

```bash
pip install -r requirements.txt
```

Download
the [IDS uEye Cockpit](https://en.ids-imaging.com/files/downloads/ids-software-suite/readme/readme-ids-software-suite-win-4.96.1_EN.html)
version 4.96.1 and run the executable file

## Run Visualization Demo

The visualization demo is designed to showcase the pendulum's movement and record both the angle and angular velocity.

To run the Visualization Demo, you can choose from multiple options when executing the program:

| Argument          | Type   | Default | Required | Description                                                                                                      |
|-------------------|--------|---------|----------|------------------------------------------------------------------------------------------------------------------|
| -rec, --record    | Bool   | False   | False    | Saves all frames as pictures (If set with detect, it saves a CSV and PKL file of the angle and angular velocity) |
| -det, --detect    | Bool   | False   | False    | 	Detects the position of the angle and angular velocity of the pendulum                                          |
| -liv, --live_feed | Bool   | False   | False    | Displays the current camera view (If set with detect, it displays the angle and angular velocity)                |
| -vp, --video_path | String | None    | False    | Saves the video of the camera feed at this location                                                              |

Example run:

```bash
python -m demo_visualization --record --detect
```

This records the frames of the pendulum positions and saves all image frames, along with a CSV and PKL file.

## Run Prediction Demo

The prediction demo can predict pendulum movement and display it in comparison to the real recorded movement.

1. Start program with:

```bash
python -m demo_prediction
```

2. Bring the pendulum to the highest possible position.
3. Press the **Vorhersage starten** button
4. Wait until the program records the data, runs the prediction, and generates the images; this may take a while.
5. Eventually, you will see the results of the original pendulum movement compared to the predicted one.

## Recalibrate camera setup
