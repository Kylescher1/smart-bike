/*
  Arduino Nano Debug Sequence (PWM Enabled)
  - Servos: Pin 3 & Pin 7
  - DC Motors: Pin 5 & Pin 6 (Moved from 12 & 9 to avoid Servo conflict)
  
  Sequence:
  1. Servo Sweeps
  2. Motor Soft-Start (Ramping speed up and down)
*/

#include <Servo.h>

// --- Configuration ---
const int servoPin1 = 3;
const int servoPin2 = 7;

// CHANGED: Using Pin 5 and 6 to ensure PWM works while Servo library is active
const int motorPin1 = 5; 
const int motorPin2 = 6;

// --- Objects ---
Servo myservo1;
Servo myservo2;

// --- Timing ---
const int servoDelay = 15; 
const int pwmDelay = 10; // Speed of motor ramping

void setup() {
  Serial.begin(9600);
  Serial.println("--- Starting PWM Debug Sequence ---");

  myservo1.attach(servoPin1);
  myservo2.attach(servoPin2);
  
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);

  // Initialize
  analogWrite(motorPin1, 0);
  analogWrite(motorPin2, 0);
  myservo1.write(0);
  myservo2.write(0);
  delay(1000);
}

void loop() {
  // --- 1. Single Servo Sweeps ---
  Serial.println(">>> Test: Servo 1 Only");
  sweepSingleServo(myservo1);
  delay(200);

  Serial.println(">>> Test: Servo 2 Only");
  sweepSingleServo(myservo2);
  delay(200);

  // --- 2. Combined Servo Sweep (STAGGERED to save power) ---
  Serial.println(">>> Test: Both Servos (Staggered)");
  
  // Sweep Forward
  for (int pos = 0; pos <= 180; pos += 2) { 
    myservo1.write(pos);
    delay(10); // Give Servo 1 a moment to sip power
    myservo2.write(pos);
    delay(10); // Now let Servo 2 sip power
  }
  
  // Sweep Backward
  for (int pos = 180; pos >= 0; pos -= 2) { 
    myservo1.write(pos);
    delay(10);
    myservo2.write(pos);
    delay(10);
  }
  delay(1000);

  // --- 3. Motor Tests (Pins 5 & 6) ---
  // (Keep your motor code here)
  Serial.println(">>> Test: Motors");
  rampMotor(motorPin1);
  rampMotor(motorPin2);
  
  // Both Motors
  for (int speed = 0; speed <= 255; speed += 5) {
    analogWrite(motorPin1, speed);
    analogWrite(motorPin2, speed);
    delay(10);
  }
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);

  Serial.println("--- Done. Restarting... ---");
  delay(2000);
}

// --- Helpers ---

void sweepSingleServo(Servo &s) {
  for (int pos = 0; pos <= 180; pos += 2) { 
    s.write(pos);
    delay(servoDelay);
  }
  for (int pos = 180; pos >= 0; pos -= 2) { 
    s.write(pos);
    delay(servoDelay);
  }
}

void rampMotor(int pin) {
  // Ramp Up
  for (int speed = 0; speed <= 255; speed += 5) {
    analogWrite(pin, speed);
    delay(pwmDelay);
  }
  // Ramp Down
  for (int speed = 255; speed >= 0; speed -= 5) {
    analogWrite(pin, speed);
    delay(pwmDelay);
  }
}