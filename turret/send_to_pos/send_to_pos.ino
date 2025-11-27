#include <ESP32Servo.h>

Servo servo1;
Servo servo2;

// ----- Servo Limits -----
const int S1_MIN = 15;
const int S1_MAX = 50;
const int S1_HOME = 35;

const int S2_MIN = 0;
const int S2_MAX = 180;
const int S2_HOME = 90;

// ----- Joystick Pins -----
int xPin = 34;  // VRx
int yPin = 35;  // VRy
int swPin = 27; // button (not used yet)

// Deadzone: prevents servo jitter when centered
const int DEADZONE = 150;

void setup() {
  Serial.begin(115200);

  servo1.attach(14, 500, 2400);
  servo2.attach(33, 500, 2400);

  pinMode(swPin, INPUT_PULLUP);

  Serial.println("Joystick servo control ready");
}

void loop() {
  // ----- Read Joystick -----
  int xVal = analogRead(xPin);   // 0–4095
  int yVal = analogRead(yPin);

  // Center of your joystick ~ 3250
  int xCenter = 3250;
  int yCenter = 3250;

  // ----- X -> Servo 1 -----
  int xDiff = xVal - xCenter;

  int s1Angle;

  if (abs(xDiff) < DEADZONE) {
    s1Angle = S1_HOME;  // center joystick → home
  } else {
    // Map joystick range to servo range
    s1Angle = map(xVal, 0, 4095, S1_MIN, S1_MAX);
  }

  s1Angle = constrain(s1Angle, S1_MIN, S1_MAX);
  servo1.write(s1Angle);

  // ----- Y -> Servo 2 -----
  int yDiff = yVal - yCenter;

  int s2Angle;

  if (abs(yDiff) < DEADZONE) {
    s2Angle = S2_HOME;
  } else {
    s2Angle = map(yVal, 0, 4095, S2_MIN, S2_MAX);
  }

  s2Angle = constrain(s2Angle, S2_MIN, S2_MAX);
  servo2.write(s2Angle);

  // ----- Debugging -----
  Serial.print("X:");
  Serial.print(xVal);
  Serial.print(" -> S1:");
  Serial.print(s1Angle);
  Serial.print("   |   Y:");
  Serial.print(yVal);
  Serial.print(" -> S2:");
  Serial.println(s2Angle);

  delay(20);
}
