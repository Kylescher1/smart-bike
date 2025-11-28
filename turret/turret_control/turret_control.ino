#include <ESP32Servo.h>

Servo servo1;  // Vertical axis (limited range)
Servo servo2;  // Horizontal axis (full range)

// ----- Servo Limits -----
const int S1_MIN = 15;
const int S1_MAX = 50;
const int S1_HOME = 35;

const int S2_MIN = 0;
const int S2_MAX = 180;
const int S2_HOME = 90;

// ----- Joystick Pins (optional) -----
int xPin = 34;  // VRx
int yPin = 35;  // VRy
int swPin = 27; // button (not used yet)

// Deadzone: prevents servo jitter when centered
const int DEADZONE = 150;

// Control mode: "JOYSTICK" or "SERIAL"
String control_mode = "SERIAL";  // Default to serial control

// Current servo positions
int current_s1 = S1_HOME;
int current_s2 = S2_HOME;

void setup() {
  Serial.begin(115200);
  
  servo1.attach(12, 500, 2400);
  servo2.attach(13, 500, 2400);
  
  pinMode(swPin, INPUT_PULLUP);
  
  // Initialize servos to home position
  servo1.write(S1_HOME);
  servo2.write(S2_HOME);
  
  Serial.println("Turret Control Ready");
  Serial.println("Mode: SERIAL (send commands like 'S1:35,S2:90')");
  Serial.println("Or set mode to JOYSTICK for joystick control");
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Check for mode change
    if (command.startsWith("MODE:")) {
      String mode = command.substring(5);
      mode.trim();
      if (mode == "JOYSTICK" || mode == "SERIAL") {
        control_mode = mode;
        Serial.print("Mode set to: ");
        Serial.println(control_mode);
      }
      return;
    }
    
    // Parse servo commands: "S1:angle,S2:angle" or "S1:angle" or "S2:angle"
    if (command.length() > 0) {
      processSerialCommand(command);
    }
  }
  
  // Execute control based on mode
  if (control_mode == "JOYSTICK") {
    joystickControl();
  }
  // SERIAL mode: servos are controlled by serial commands only
}

void processSerialCommand(String command) {
  // Parse command like "S1:35,S2:90" or "S1:35" or "S2:90"
  int s1_angle = -1;  // -1 means don't change
  int s2_angle = -1;
  
  // Split by comma if multiple commands
  int commaIndex = command.indexOf(',');
  
  if (commaIndex > 0) {
    // Two commands: "S1:35,S2:90"
    String cmd1 = command.substring(0, commaIndex);
    String cmd2 = command.substring(commaIndex + 1);
    
    s1_angle = parseServoCommand(cmd1, 1);
    s2_angle = parseServoCommand(cmd2, 2);
  } else {
    // Single command: "S1:35" or "S2:90"
    if (command.startsWith("S1:")) {
      s1_angle = parseServoCommand(command, 1);
    } else if (command.startsWith("S2:")) {
      s2_angle = parseServoCommand(command, 2);
    }
  }
  
  // Update servos
  if (s1_angle >= 0) {
    s1_angle = constrain(s1_angle, S1_MIN, S1_MAX);
    servo1.write(s1_angle);
    current_s1 = s1_angle;
    Serial.print("S1: ");
    Serial.println(s1_angle);
  }
  
  if (s2_angle >= 0) {
    s2_angle = constrain(s2_angle, S2_MIN, S2_MAX);
    servo2.write(s2_angle);
    current_s2 = s2_angle;
    Serial.print("S2: ");
    Serial.println(s2_angle);
  }
}

int parseServoCommand(String cmd, int servo_num) {
  // Parse "S1:35" or "S2:90"
  String prefix = "S" + String(servo_num) + ":";
  
  if (cmd.startsWith(prefix)) {
    String angleStr = cmd.substring(prefix.length());
    int angle = angleStr.toInt();
    return angle;
  }
  
  return -1;  // Invalid command
}

void joystickControl() {
  // ----- Read Joystick -----
  int xVal = analogRead(xPin);   // 0–4095
  int yVal = analogRead(yPin);
  
  // Center of joystick ~ 3250
  int xCenter = 3250;
  int yCenter = 3250;
  
  // ----- X -> Servo 1 (vertical) -----
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
  current_s1 = s1Angle;
  
  // ----- Y -> Servo 2 (horizontal) -----
  int yDiff = yVal - yCenter;
  
  int s2Angle;
  
  if (abs(yDiff) < DEADZONE) {
    s2Angle = S2_HOME;
  } else {
    s2Angle = map(yVal, 0, 4095, S2_MIN, S2_MAX);
  }
  
  s2Angle = constrain(s2Angle, S2_MIN, S2_MAX);
  servo2.write(s2Angle);
  current_s2 = s2Angle;
  
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

