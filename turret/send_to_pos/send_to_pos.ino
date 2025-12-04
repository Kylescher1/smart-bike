#include <ESP32Servo.h>

Servo servo1;
Servo servo2;

// ----- Servo Limits (Preserved to protect hardware) -----
const int S1_MIN = 15;
const int S1_MAX = 75;
const int S2_MIN = 0;
const int S2_MAX = 180;

void setup() {
  Serial.begin(115200);

  // Attach to same pins as before
  servo1.attach(25, 500, 2400);
  servo2.attach(26, 500, 2400);

  Serial.println("Serial Control Ready.");
  Serial.println("Enter format: <ServoNum> <Angle>");
  Serial.println("Example: '1 35' or '2 90'");
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    
    // Read the first integer (Servo Number: 1 or 2)
    int servoNum = Serial.parseInt();
    
    // Read the second integer (Angle)
    int angle = Serial.parseInt();

    // Clear buffer of newlines
    while (Serial.available() > 0 && Serial.peek() < 33) {
      Serial.read();
    }

    // Process valid input
    if (servoNum == 1) {
      // Constrain to S1 limits to prevent mechanical damage
      angle = constrain(angle, S1_MIN, S1_MAX);
      servo1.write(angle);
      
      Serial.print("Moving Servo 1 to: ");
      Serial.println(angle);
      
    } else if (servoNum == 2) {
      // Constrain to S2 limits
      angle = constrain(angle, S2_MIN, S2_MAX);
      servo2.write(angle);
      
      Serial.print("Moving Servo 2 to: ");
      Serial.println(angle);
    }
  }
}