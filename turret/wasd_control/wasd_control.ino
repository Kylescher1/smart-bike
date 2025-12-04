#include <Servo.h>

Servo myServo;
int angle = 0;

void setup() {
  Serial.begin(9600);
  myServo.attach(6);

  Serial.println("Enter an angle from 0 to 180:");
}

void loop() {
  if (Serial.available() > 0) {
    angle = Serial.parseInt();  // Read the number typed into Serial Monitor

    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
      Serial.print("Servo moved to: ");
      Serial.println(angle);
    } else {
      Serial.println("Invalid angle. Enter 0–180.");
    }

    // Clear any leftover characters
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}
