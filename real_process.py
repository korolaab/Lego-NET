import cv2
import im_process
import PIL
from PIL import Image
import numpy as np
cap = cv2.VideoCapture(0)
import time
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    objcts = im_process.main(Image.fromarray(rgb, mode='RGB'))
    # Display the resulting frame
    for obj in objcts:
        cord,label = obj
        x,y,w,h=cord
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        sample = np.asarray(cv2.resize(frame[x:x+w, y:y+h], (100, 100)) )
            # # PLT.imshow(sample)
            # # PLT.show()
            
        sample =np.expand_dims(sample, axis=0)/255  ###adding one dimension to sample
                
             
                
        res = im_process.model.predict(sample)###neral network prediction
        prob = res[0][label]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'{} {}'.format(im_process.name[label],prob),(x,y+h), font, 0.3,(0,255,0),1,cv2.LINE_AA)


    cv2.imshow('frame',frame)
    print("_________-frame___________")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(2)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

