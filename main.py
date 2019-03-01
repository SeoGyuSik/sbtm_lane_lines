import cv2
import check_line
#import control
from simple_pid import PID

pid = PID(10, 8, 10, setpoint=130)
pid.output_limits = (-1000, 1000)

standard_angle = 130
ALIVE = 0

while True:
    (HSVLOW, HSVHIGH) = check_line.get_hsv()
    if cv2.waitKey(1) == 27:
        check_line.close()
        break

while True:
    slope = check_line.get_slope(HSVLOW, HSVHIGH)
    output = pid(slope)
    #STEER = int(((slope - 100) / (160 - 100)) * (2000 - (-1999)) + (-1999))
    print(-output)
    ALIVE = ALIVE + 1
    #control.send_data(50, STEER, 0, ALIVE, 0)

    if ALIVE is 256:
        ALIVE = 0

    if cv2.waitKey(1) == 27:
        check_line.close()
        break