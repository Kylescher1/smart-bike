#include <ESP32Servo.h>

Servo myservo;
int targetAngle = 90;  // default to center

void setup() {
  Serial.begin(115200);
  myservo.attach(14, 500, 2400);   // your servo pin + pulse range
  Serial.println("Send an angle between 0 and 180:");
}

void loop() {
  // Check if user typed something
  if (Serial.available()) {
    targetAngle = Serial.parseInt();   // read number sent
    if (targetAngle >= 0 && targetAngle <= 180) {
      Serial.print("Moving to: ");
      Serial.println(targetAngle);
      myservo.write(targetAngle);
    } else {
      Serial.println("Invalid angle. Send 0–180.");
    }
  }
}
