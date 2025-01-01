import RPi.GPIO as GPIO
import time
from datetime import datetime

# 設定GPIO編號模式
GPIO.setmode(GPIO.BCM)
# 設定GPIO pin，假設使用 GPIO 18
GPIO_PIN = 18
GPIO.setup(GPIO_PIN, GPIO.OUT)

print("START")
try:
    while True:
        # 獲取當前時間
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        # 判斷是否在晚上2點到早上8點30分之間
        if (current_hour == 2 and current_minute >= 0) or (2 < current_hour < 8) or (current_hour == 8 and current_minute <= 30):
            # 設定GPIO LOW
            GPIO.output(GPIO_PIN, GPIO.LOW)
        else:
            # 設定GPIO HIGH
            GPIO.output(GPIO_PIN, GPIO.HIGH)
        
        # 每分鐘檢查一次
        time.sleep(60)

except KeyboardInterrupt:
    # 清除GPIO設置
    GPIO.cleanup()

