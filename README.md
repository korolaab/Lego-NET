
# Lego-NET
Lego recognizing network. The main task of the machine is sorting Lego parts such as: Container, Wheel, Engine.
## REQUIREMENTS
- Python 3.5.2
- Tensorflow 1.12.0
- Keras 2.2.4
## HOW TO TRAIN
1. Extract "dataset" into the root of "Lego-NET".
2. Open terminal.
3. Input ```python train.py --model=cnn```have nice training. =)
## HOW TO USE
Input ```python im_process.py --image=<path> --Weights=<weights path>```. You will get a probability map for all objects.
## EXAMPLES
<div>
  <img src="https://pp.userapi.com/c844216/v844216037/1eb8e3/hl2pBkrQPhs.jpg" alt="CNN output example">
</div>
CNN output example
<div>
  <img src="https://pp.userapi.com/c845420/v845420037/1e53bd/P6j-2s2mfYc.jpg" alt="U-Net output example">
<div/p>
U-Net output example

## DATASET
Extract to root of Lego-NET archive
- [CNN](https://drive.google.com/file/d/1D7mEB8XH9sLy6GHo89HE-NMJRcYorvLq/view?usp=sharing)
- [U-Net](https://drive.google.com/file/d/1eIevr0rBsCDAZKlFUizGdo685VjXWg1P/view?usp=sharing)
### Images for test
Just extract and set one of them in arguments.([Images](https://drive.google.com/open?id=1U0v3WrnQEql4P-VBB7b0l_9CkrfQ3_bU))
