/*
 * ESP32 MPU6050 Servo Controller
 * 
 * Hardware:
 * - MPU6050 accelerometer/gyroscope (I2C address 0x68, SDA=GPIO4, SCL=GPIO15)
 * - Top servo: GPIO 32, range 5-60°, home position 35°
 * - Bottom servo: GPIO 33, range 0-180°, home position 90°
 * - Vibration motors: Disabled
 * 
 * Serial Protocol (115200 baud):
 * - READ: Returns accelerometer and gyro data as "ax,ay,az,gx,gy,gz"
 * - MOVE,B,ang,T,angle: Moves bottom servo (GPIO33) to ang degrees, top servo (GPIO32) to angle degrees
 */

 #include <Wire.h>
 #include <ESP32Servo.h>
 
 // --- PIN DEFINITIONS ---
 // Servos
 #define SERVO_TOP_PIN 32
 #define SERVO_BOTTOM_PIN 33
 
// Vibration Motors (COMMENTED OUT - servos are on pins 32/33)
// #define L_PWM 14
// #define L_DIR 27
// #define R_PWM 32
// #define R_DIR 33

// LEDC PWM configuration for vibration motors (COMMENTED OUT)
// Using ledcAttach API to avoid conflicts with ESP32Servo library
// #define PWM_FREQ 20000    // 20kHz PWM frequency for motors (above human hearing, smooth vibration)
// #define PWM_RESOLUTION 8  // 8-bit resolution (0-255)
 
 // MPU6050/MPU6500/MPU9250 I2C Address
 // Standard address is 0x68, but can be 0x69 if AD0 pin is high
 #define MPU6050_ADDR 0x68
 #define MPU6050_ADDR_ALT 0x69  // Alternate address if AD0 is high
 #define MPU6050_PWR_MGMT_1 0x6B
 #define MPU6050_ACCEL_XOUT_H 0x3B
 #define MPU6050_WHO_AM_I 0x75  // Device ID register
 
 // I2C Pin definitions for GY-521 module
 // GY-521: SDA=GPIO4, SCL=GPIO15
 #define I2C_SDA_PIN 4
 #define I2C_SCL_PIN 15
 
 // ----- Servo Limits -----
 // Pin 32 = Top servo: range 5-60°, home at 35°
 const int S2_MIN = 5;   // Top servo (GPIO 32) minimum
 const int S2_MAX = 60;  // Top servo (GPIO 32) maximum
 const int S2_HOME = 35; // Top servo (GPIO 32) home position
 // Pin 33 = Bottom servo: range 0-180°, home at 90°
 const int S1_MIN = 0;   // Bottom servo (GPIO 33) minimum
 const int S1_MAX = 180; // Bottom servo (GPIO 33) maximum
 const int S1_HOME = 90; // Bottom servo (GPIO 33) home position
 
 // Servo objects
 Servo servoTop;
 Servo servoBottom;
 
// Serial communication buffer
String inputString = "";
bool stringComplete = false;

// Store current vibration motor intensities (COMMENTED OUT - vibration motors disabled)
// int currentLeftIntensity = 0;
// int currentRightIntensity = 0;

// Store detected MPU address
byte detectedMPUAddress = MPU6050_ADDR;
 
void setup() {
  // VIBRATION MOTOR INITIALIZATION (COMMENTED OUT - vibration motors disabled)
  // CRITICAL: Initialize vibration motor pins FIRST to prevent unwanted vibration on boot
  // Set as outputs immediately (before any delays that might allow motors to run)
  // pinMode(L_DIR, OUTPUT);
  // pinMode(R_DIR, OUTPUT);
  
  // Set direction pins to safe state
  // digitalWrite(L_DIR, LOW);
  // digitalWrite(R_DIR, LOW);
  
  // Initialize PWM channels explicitly for vibration motors using LEDC
  // This prevents conflicts with ESP32Servo library which also uses LEDC channels
  // ledcAttach(L_PWM, PWM_FREQ, PWM_RESOLUTION);
  // ledcAttach(R_PWM, PWM_FREQ, PWM_RESOLUTION);
  
  // Immediately set to safe state (0 PWM) - do this before any delays
  // ledcWrite(L_PWM, 0);
  // ledcWrite(R_PWM, 0);
  
  // Initialize serial communication
  Serial.begin(115200);
  Serial.setTimeout(100);
  inputString.reserve(200);
  
 // Initialize I2C for MPU6050/MPU6500/MPU9250
 Serial.println("\n=== I2C Initialization ===");
 #if defined(I2C_SDA_PIN) && defined(I2C_SCL_PIN)
   Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
   Serial.print("I2C initialized with custom pins: SDA=");
   Serial.print(I2C_SDA_PIN);
   Serial.print(", SCL=");
   Serial.println(I2C_SCL_PIN);
 #else
   Wire.begin();  // Use default pins (SDA=21, SCL=22)
   Serial.println("I2C initialized with default pins: SDA=21, SCL=22");
 #endif
 Wire.setClock(100000);  // Start with slower 100kHz clock for reliability
 delay(200);  // Longer delay for I2C bus to stabilize
 
 // Scan I2C bus to see what devices are present
 Serial.println("\n=== I2C Bus Scan ===");
 byte devicesFound = 0;
 for (byte address = 1; address < 127; address++) {
   Wire.beginTransmission(address);
   byte error = Wire.endTransmission();
   if (error == 0) {
     Serial.print("Device found at address 0x");
     if (address < 16) Serial.print("0");
     Serial.print(address, HEX);
     if (address == MPU6050_ADDR || address == MPU6050_ADDR_ALT) {
       Serial.println(" <- MPU6050/6500/9250 (expected)");
     } else {
       Serial.println();
     }
     devicesFound++;
   }
 }
 if (devicesFound == 0) {
   Serial.println("No I2C devices found!");
   Serial.println("Troubleshooting:");
   Serial.println("  1. Check wiring: SDA->GPIO21, SCL->GPIO22");
   Serial.println("  2. Verify power: VCC->3.3V, GND->GND");
   Serial.println("  3. Check pull-up resistors (4.7kΩ on SDA/SCL)");
   Serial.println("  4. Try slower I2C speed or different pins");
 }
 Serial.print("Total devices found: ");
 Serial.println(devicesFound);
 
 // Try to detect MPU at standard address (0x68) or alternate (0x69)
 Serial.println("\n=== MPU Detection ===");
 byte mpuAddress = 0;
 byte error1 = 0, error2 = 0;
 
 // Check standard address (0x68)
 Wire.beginTransmission(MPU6050_ADDR);
 error1 = Wire.endTransmission();
 Serial.print("Address 0x68: ");
 if (error1 == 0) {
   Serial.println("OK");
 } else {
   Serial.print("Error code ");
   Serial.println(error1);
   Serial.println("  (0=OK, 1=data too long, 2=NACK on address, 3=NACK on data, 4=other)");
 }
 
 // Check alternate address (0x69)
 Wire.beginTransmission(MPU6050_ADDR_ALT);
 error2 = Wire.endTransmission();
 Serial.print("Address 0x69: ");
 if (error2 == 0) {
   Serial.println("OK");
 } else {
   Serial.print("Error code ");
   Serial.println(error2);
 }
 
 if (error1 == 0) {
   mpuAddress = MPU6050_ADDR;
   Serial.print("✓ MPU detected at address 0x");
   Serial.println(mpuAddress, HEX);
   detectedMPUAddress = mpuAddress;
 } else if (error2 == 0) {
   mpuAddress = MPU6050_ADDR_ALT;
   Serial.print("✓ MPU detected at alternate address 0x");
   Serial.println(mpuAddress, HEX);
   detectedMPUAddress = mpuAddress;
 } else {
   Serial.println("✗ WARNING: MPU not detected at 0x68 or 0x69");
   Serial.println("  Using address 0x68 as default (may not work)");
   mpuAddress = MPU6050_ADDR;  // Default to standard address
   detectedMPUAddress = mpuAddress;
 }
 
 // Try to initialize MPU if detected
 if (error1 == 0 || error2 == 0) {
   Serial.println("\n=== MPU Initialization ===");
   
   // Wake up MPU6050/MPU6500/MPU9250
   Wire.beginTransmission(mpuAddress);
   Wire.write(MPU6050_PWR_MGMT_1);
   Wire.write(0); // Wake up device (clear sleep bit)
   byte error = Wire.endTransmission();
   if (error == 0) {
     Serial.println("Wake command sent successfully");
   } else {
     Serial.print("Wake command failed (error: ");
     Serial.print(error);
     Serial.println(")");
   }
   delay(100);
   
   // Additional initialization for MPU6500/MPU9250 compatibility
   // Reset device (bit 7 of PWR_MGMT_1)
   Wire.beginTransmission(mpuAddress);
   Wire.write(MPU6050_PWR_MGMT_1);
   Wire.write(0x80); // Reset
   error = Wire.endTransmission();
   if (error == 0) {
     Serial.println("Reset command sent successfully");
   }
   delay(200);  // Longer delay after reset
   
   // Wake up again after reset
   Wire.beginTransmission(mpuAddress);
   Wire.write(MPU6050_PWR_MGMT_1);
   Wire.write(0); // Wake up
   error = Wire.endTransmission();
   if (error == 0) {
     Serial.println("Wake command after reset sent successfully");
   }
   delay(100);
   
   // Increase I2C speed now that device is initialized
   Wire.setClock(400000);  // 400kHz fast mode
   Serial.println("I2C clock increased to 400kHz");
   
   // Verify device is responding
   Wire.beginTransmission(mpuAddress);
   error = Wire.endTransmission();
   if (error == 0) {
     Serial.println("✓ MPU initialization complete and responding");
   } else {
     Serial.print("✗ ERROR: MPU not responding after init (error code: ");
     Serial.print(error);
     Serial.println(")");
   }
 } else {
   Serial.println("\n⚠ Skipping MPU initialization - device not detected");
 }
 
 Serial.println("=== Setup Complete ===\n");
   
  // Initialize servos
  Serial.println("\n=== Servo Initialization ===");
  Serial.print("Attaching servos: Top=GPIO");
  Serial.print(SERVO_TOP_PIN);
  Serial.print(", Bottom=GPIO");
  Serial.println(SERVO_BOTTOM_PIN);
  
  servoTop.attach(SERVO_TOP_PIN);
  servoBottom.attach(SERVO_BOTTOM_PIN);
  
  Serial.println("Servos attached successfully");
  
  // Set servos to home positions
  Serial.print("Setting home positions: Bottom (GPIO33)=");
  Serial.print(S1_HOME);
  Serial.print("°, Top (GPIO32)=");
  Serial.print(S2_HOME);
  Serial.println("°");
  
  servoBottom.write(S1_HOME);  // GPIO 33: Bottom servo, home at 90°
  servoTop.write(S2_HOME);      // GPIO 32: Top servo, home at 35°
  
  Serial.println("✓ Servos initialized and set to center position");
  
  // VIBRATION MOTOR STATE RE-ASSERTION (COMMENTED OUT - vibration motors disabled)
  // CRITICAL: Re-assert vibration motor state after servo initialization
  // ESP32Servo may reconfigure LEDC channels, so we need to ensure motors stay off
  // ledcWrite(L_PWM, 0);
  // ledcWrite(R_PWM, 0);
  // digitalWrite(L_DIR, LOW);
  // digitalWrite(R_DIR, LOW);
  
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
  // VIBRATE command (COMMENTED OUT - vibration motors disabled)
  // else if (cmd.startsWith("VIBRATE,")) {
  //   handleVibrateCommand(cmd);
  // }
   else {
     Serial.println("ERROR");
   }
 }
 
void readMPU6050() {
  // Check if MPU6050/MPU6500/MPU9250 is responding
  Wire.beginTransmission(detectedMPUAddress);
  byte error = Wire.endTransmission();
  if (error != 0) {
    Serial.println("ERROR");
    return;
  }
  
  // Request data from MPU
  Wire.beginTransmission(detectedMPUAddress);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  
  // Request 14 bytes and wait a bit for the data to be ready
  Wire.requestFrom(detectedMPUAddress, 14, true);
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
   
   // VIBRATION MOTOR STATE RE-ASSERTION (COMMENTED OUT - vibration motors disabled)
   // CRITICAL: Re-assert vibration motor state after servo write
   // ESP32Servo.write() may reconfigure LEDC channels, causing motors to activate
   // Reapply the current vibration state to ensure motors stay at intended level
   // ledcWrite(L_PWM, currentLeftIntensity);
   // ledcWrite(R_PWM, currentRightIntensity);
   
   Serial.println("OK");
 }
 
 // VIBRATION COMMAND HANDLER (COMMENTED OUT - vibration motors disabled)
 // void handleVibrateCommand(String cmd) {
 //   // Parse: VIBRATE,L,R
 //   int firstComma = cmd.indexOf(',');
 //   int secondComma = cmd.indexOf(',', firstComma + 1);
 //   
 //   if (firstComma == -1 || secondComma == -1) {
 //     Serial.println("ERROR");
 //     return;
 //   }
 //   
 //   int leftIntensity = cmd.substring(firstComma + 1, secondComma).toInt();
 //   int rightIntensity = cmd.substring(secondComma + 1).toInt();
 //   
 //   // Clamp to 0-255
 //   leftIntensity = constrain(leftIntensity, 0, 255);
 //   rightIntensity = constrain(rightIntensity, 0, 255);
 //   
 //   // Store intensity values for reapplication after servo operations
 //   currentLeftIntensity = leftIntensity;
 //   currentRightIntensity = rightIntensity;
 //   
 //   // Set vibration motors using LEDC (not analogWrite to avoid conflicts)
 //   ledcWrite(L_PWM, leftIntensity);
 //   ledcWrite(R_PWM, rightIntensity);
 //   
 //   // Set direction (LOW = one direction, HIGH = reverse)
 //   // Adjust based on your motor driver requirements
 //   digitalWrite(L_DIR, leftIntensity > 0 ? LOW : LOW);
 //   digitalWrite(R_DIR, rightIntensity > 0 ? LOW : LOW);
 //   
 //   Serial.println("OK");
 // }
 
 