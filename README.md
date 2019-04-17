
# Lego hand
Lego sorting machine. The main task of the machine is sorting Lego parts such as: Container, Wheel, Engine.
## REQUIREMENTS
- Python 3.5.2
- Tensorflow 1.12.0
- Keras 2.2.4
## HOW TO TRAIN
1. Extract "dataset" into the root of "Lego_hand".
2. Open terminal.
3. Input ```python train.py --model=cnn```have nice training. =)
## HOW TO USE
Input ```python im_process.py --image=<path> --Weights=<weights path>``` You will get location of objects.
Add  ```--map ``` if you want to watch heatmap.
## EXAMPLES
![CNN output](https://sun1-6.userapi.com/c844722/v844722616/18d1f0/RWZLHSuz6x0.jpg)
![U-Net output](https://pp.userapi.com/c847220/v847220898/1b0b58/et8-0wIU5uI.jpg)
## DATASET
Extract to root of Lego_hand archive
- [CNN](https://drive.google.com/file/d/1D7mEB8XH9sLy6GHo89HE-NMJRcYorvLq/view?usp=sharing)
- [U-Net](https://drive.google.com/file/d/1eIevr0rBsCDAZKlFUizGdo685VjXWg1P/view?usp=sharing)
### Images for test
Just extract and set one of them in arguments.([Images](https://drive.google.com/open?id=1U0v3WrnQEql4P-VBB7b0l_9CkrfQ3_bU))
