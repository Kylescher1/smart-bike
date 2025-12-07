/*
 * ESP32 MPU6050 Servo and Vibration Motor Controller
 * 
 * Hardware:
 * - MPU6050 accelerometer/gyroscope (I2C address 0x68)
 * - Two servos: Top (GPIO 25), Bottom (GPIO 26)
 * - Two vibration motors: Left (PWM 14, DIR 27), Right (PWM 32, DIR 33)
 * 
 * Serial Protocol (115200 baud):
 * - READ: Returns accelerometer and gyro data as "ax,ay,az,gx,gy,gz"
 * - MOVE,B,ang,T,angle: Moves bottom servo to ang degrees, top servo to angle degrees
 * - VIBRATE,L,R: Sets vibration intensity (0-255) for left and right motors
 */

 #include <Wire.h>
 #include <ESP32Servo.h>
 
 // --- PIN DEFINITIONS ---
 // Servos
 #define SERVO_TOP_PIN 25
 #define SERVO_BOTTOM_PIN 26
 
 // Vibration Motors
 #define L_PWM 14
 #define L_DIR 27
 #define R_PWM 32
 #define R_DIR 33
 
 // MPU6050 I2C Address
 #define MPU6050_ADDR 0x68
 #define MPU6050_PWR_MGMT_1 0x6B
 #define MPU6050_ACCEL_XOUT_H 0x3B
 
 // ----- Servo Limits (Preserved to protect hardware) -----
 const int S1_MIN = 15;  // Bottom servo minimum
 const int S1_MAX = 75;  // Bottom servo maximum
 const int S2_MIN = 0;   // Top servo minimum
 const int S2_MAX = 180; // Top servo maximum
 
 // Servo objects
 Servo servoTop;
 Servo servoBottom;
 
 // Serial communication buffer
 String inputString = "";
 bool stringComplete = false;
 
void setup() {
  // CRITICAL: Initialize vibration motor pins FIRST to prevent unwanted vibration on boot
  // Set as outputs immediately (before any delays that might allow motors to run)
  pinMode(L_PWM, OUTPUT);
  pinMode(L_DIR, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(R_DIR, OUTPUT);
  
  // Immediately set to safe state (0 PWM, LOW direction) - do this before any delays
  analogWrite(L_PWM, 0);
  digitalWrite(L_DIR, LOW);
  analogWrite(R_PWM, 0);
  digitalWrite(R_DIR, LOW);
  
  // Initialize serial communication
  Serial.begin(115200);
  Serial.setTimeout(100);
  inputString.reserve(200);
  
 // Initialize I2C for MPU6050
 Wire.begin();
 delay(100);
 
 // Wake up MPU6050
 Wire.beginTransmission(MPU6050_ADDR);
 Wire.write(MPU6050_PWR_MGMT_1);
 Wire.write(0); // Wake up MPU6050
 byte error = Wire.endTransmission();
 delay(100);
 
 // Verify MPU6050 is responding (optional diagnostic)
 // If error != 0, MPU6050 may not be connected
   
  // Initialize servos
  servoTop.attach(SERVO_TOP_PIN);
  servoBottom.attach(SERVO_BOTTOM_PIN);
  
  // Set servos to initial positions (middle of range)
  servoBottom.write((S1_MIN + S1_MAX) / 2);
  servoTop.write((S2_MIN + S2_MAX) / 2);
  
  // Ensure motors are still stopped (redundant but safe)
  analogWrite(L_PWM, 0);
  digitalWrite(L_DIR, LOW);
  analogWrite(R_PWM, 0);
  digitalWrite(R_DIR, LOW);
  
  delay(500);
}
 
 void loop() {
   // Check for serial input
   if (Serial.available() > 0) {
     inputString = Serial.readStringUntil('\n');
     inputString.trim();
     
     if (inputString.length() > 0) {
       processCommand(inputString);
     }
   }
 }
 
 void processCommand(String cmd) {
   cmd.toUpperCase();
   
   // READ command: Return accelerometer and gyro data
   if (cmd == "READ") {
     readMPU6050();
   }
   // MOVE command: MOVE,B,ang,T,angle
   else if (cmd.startsWith("MOVE,")) {
     handleMoveCommand(cmd);
   }
   // VIBRATE command: VIBRATE,L,R
   else if (cmd.startsWith("VIBRATE,")) {
     handleVibrateCommand(cmd);
   }
   else {
     Serial.println("ERROR");
   }
 }
 
void readMPU6050() {
  // Check if MPU6050 is responding
  Wire.beginTransmission(MPU6050_ADDR);
  byte error = Wire.endTransmission();
  if (error != 0) {
    Serial.println("ERROR");
    return;
  }
  
  // Request data from MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  
  // Request 14 bytes and wait a bit for the data to be ready
  Wire.requestFrom(MPU6050_ADDR, 14, true);
  delayMicroseconds(100);  // Small delay to ensure data is ready
  
  // Wait for data with timeout
  unsigned long startTime = millis();
  while (Wire.available() < 14 && (millis() - startTime) < 10) {
    delayMicroseconds(100);
  }
  
  if (Wire.available() >= 14) {
    // Read accelerometer data (16-bit signed)
    int16_t accelX = (Wire.read() << 8) | Wire.read();
    int16_t accelY = (Wire.read() << 8) | Wire.read();
    int16_t accelZ = (Wire.read() << 8) | Wire.read();
    
    // Skip temperature
    Wire.read();
    Wire.read();
    
    // Read gyroscope data (16-bit signed)
    int16_t gyroX = (Wire.read() << 8) | Wire.read();
    int16_t gyroY = (Wire.read() << 8) | Wire.read();
    int16_t gyroZ = (Wire.read() << 8) | Wire.read();
    
    // Convert to g and degrees/s (assuming ±2g and ±250°/s ranges)
    float ax = accelX / 16384.0;
    float ay = accelY / 16384.0;
    float az = accelZ / 16384.0;
    float gx = gyroX / 131.0;
    float gy = gyroY / 131.0;
    float gz = gyroZ / 131.0;
    
    // Send comma-separated values
    Serial.print(ax, 6);
    Serial.print(",");
    Serial.print(ay, 6);
    Serial.print(",");
    Serial.print(az, 6);
    Serial.print(",");
    Serial.print(gx, 6);
    Serial.print(",");
    Serial.print(gy, 6);
    Serial.print(",");
    Serial.println(gz, 6);
  } else {
    Serial.println("ERROR");
  }
}
 
 void handleMoveCommand(String cmd) {
   // Parse: MOVE,B,ang,T,angle
   int firstComma = cmd.indexOf(',');
   int secondComma = cmd.indexOf(',', firstComma + 1);
   int thirdComma = cmd.indexOf(',', secondComma + 1);
   int fourthComma = cmd.indexOf(',', thirdComma + 1);
   
   if (firstComma == -1 || secondComma == -1 || thirdComma == -1 || fourthComma == -1) {
     Serial.println("ERROR");
     return;
   }
   
   String bPart = cmd.substring(firstComma + 1, secondComma);
   String angStr = cmd.substring(secondComma + 1, thirdComma);
   String tPart = cmd.substring(thirdComma + 1, fourthComma);
   String angleStr = cmd.substring(fourthComma + 1);
   
   if (bPart != "B" || tPart != "T") {
     Serial.println("ERROR");
     return;
   }
   
   int bottomAngle = angStr.toInt();
   int topAngle = angleStr.toInt();
   
   // Clamp angles to servo limits
   bottomAngle = constrain(bottomAngle, S1_MIN, S1_MAX);
   topAngle = constrain(topAngle, S2_MIN, S2_MAX);
   
   servoBottom.write(bottomAngle);
   servoTop.write(topAngle);
   
   Serial.println("OK");
 }
 
 void handleVibrateCommand(String cmd) {
   // Parse: VIBRATE,L,R
   int firstComma = cmd.indexOf(',');
   int secondComma = cmd.indexOf(',', firstComma + 1);
   
   if (firstComma == -1 || secondComma == -1) {
     Serial.println("ERROR");
     return;
   }
   
   int leftIntensity = cmd.substring(firstComma + 1, secondComma).toInt();
   int rightIntensity = cmd.substring(secondComma + 1).toInt();
   
   // Clamp to 0-255
   leftIntensity = constrain(leftIntensity, 0, 255);
   rightIntensity = constrain(rightIntensity, 0, 255);
   
   // Set vibration motors
   analogWrite(L_PWM, leftIntensity);
   analogWrite(R_PWM, rightIntensity);
   
   // Set direction (LOW = one direction, HIGH = reverse)
   // Adjust based on your motor driver requirements
   digitalWrite(L_DIR, leftIntensity > 0 ? LOW : LOW);
   digitalWrite(R_DIR, rightIntensity > 0 ? LOW : LOW);
   
   Serial.println("OK");
 }
 
 