/*
 * ESP32 MPU6050 Servo Controller
 * 
 * Hardware:
 * - MPU6050 accelerometer/gyroscope (I2C address 0x68, SDA=GPIO4, SCL=GPIO15)
 * - Top servo: GPIO 32, range 5-60°, home position 35°
 * - Bottom servo: GPIO 33, range 0-180°, home position 90°
 * - Haptic motors: IN1=GPIO25, IN2=GPIO26 (Left), IN3=GPIO27, IN4=GPIO14 (Right)
 * 
 * Serial Protocol (115200 baud):
 * - READ: Returns accelerometer and gyro data as "ax,ay,az,gx,gy,gz"
 * - MOVE,B,ang,T,angle: Moves bottom servo (GPIO33) to ang degrees, top servo (GPIO32) to angle degrees
 * - VIBRATE,L,R: Sets haptic intensity (0-255) for left and right motors
 */

 #include <Wire.h>
 #include <ESP32Servo.h>
 
 // --- PIN DEFINITIONS ---
 // Servos
 #define SERVO_TOP_PIN 32
 #define SERVO_BOTTOM_PIN 33
 
// Haptic/Vibration Motors - Dual H-bridge control (IN1-4)
// Motor 1 (Left): IN1=GPIO25, IN2=GPIO26
// Motor 2 (Right): IN3=GPIO27, IN4=GPIO14
#define HAPTIC_IN1 25  // Left motor forward
#define HAPTIC_IN2 26  // Left motor reverse
#define HAPTIC_IN3 27  // Right motor forward
#define HAPTIC_IN4 14  // Right motor reverse

// LEDC PWM configuration for haptic motors
#define PWM_FREQ 20000    // 20kHz PWM frequency (above human hearing, smooth vibration)
#define PWM_RESOLUTION 8  // 8-bit resolution (0-255)
 
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

// Store current haptic motor intensities (0-255)
int currentLeftIntensity = 0;
int currentRightIntensity = 0;

// Store detected MPU address
byte detectedMPUAddress = MPU6050_ADDR;
 
void setup() {
  // CRITICAL: Initialize haptic motor pins FIRST to prevent unwanted vibration on boot
  // Set as outputs immediately (before any delays that might allow motors to run)
  pinMode(HAPTIC_IN1, OUTPUT);
  pinMode(HAPTIC_IN2, OUTPUT);
  pinMode(HAPTIC_IN3, OUTPUT);
  pinMode(HAPTIC_IN4, OUTPUT);
  
  // Set all pins to LOW (motors stopped) - do this before any delays
  digitalWrite(HAPTIC_IN1, LOW);
  digitalWrite(HAPTIC_IN2, LOW);
  digitalWrite(HAPTIC_IN3, LOW);
  digitalWrite(HAPTIC_IN4, LOW);
  
  // Initialize PWM channels for haptic motors using LEDC
  // IN1 and IN3 use PWM for speed control
  // IN2 and IN4 are used as digital outputs (direction control)
  ledcAttach(HAPTIC_IN1, PWM_FREQ, PWM_RESOLUTION);  // Left motor PWM
  ledcAttach(HAPTIC_IN3, PWM_FREQ, PWM_RESOLUTION);  // Right motor PWM
  // IN2 and IN4 are digital outputs, no PWM needed
  
  // Immediately set to safe state (0 PWM, all LOW) - do this before any delays
  ledcWrite(HAPTIC_IN1, 0);
  ledcWrite(HAPTIC_IN3, 0);
  digitalWrite(HAPTIC_IN2, LOW);
  digitalWrite(HAPTIC_IN4, LOW);
  
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
  
  // CRITICAL: Re-assert haptic motor state after servo initialization
  // ESP32Servo may reconfigure LEDC channels, so we need to ensure motors stay off
  setHapticMotors(0, 0);
  
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
   
   // CRITICAL: Re-assert haptic motor state after servo write
   // ESP32Servo.write() may reconfigure LEDC channels, causing motors to activate
   // Reapply the current haptic state to ensure motors stay at intended level
   setHapticMotors(currentLeftIntensity, currentRightIntensity);
   
   Serial.println("OK");
 }
 
// Helper function to set haptic motors using H-bridge control
void setHapticMotors(int leftIntensity, int rightIntensity) {
  // Clamp intensities to 0-255
  leftIntensity = constrain(leftIntensity, 0, 255);
  rightIntensity = constrain(rightIntensity, 0, 255);
  
  // Store current intensities
  currentLeftIntensity = leftIntensity;
  currentRightIntensity = rightIntensity;
  
  // Left motor (IN1/IN2): Use PWM for speed control
  // IN1 = PWM control, IN2 = digital direction (LOW for forward)
  if (leftIntensity > 0) {
    ledcWrite(HAPTIC_IN1, leftIntensity);  // Forward PWM (0-255)
    digitalWrite(HAPTIC_IN2, LOW);          // Reverse off (LOW = forward direction)
  } else {
    ledcWrite(HAPTIC_IN1, 0);                // Forward off (PWM = 0)
    digitalWrite(HAPTIC_IN2, LOW);          // Keep LOW when stopped
  }
  
  // Right motor (IN3/IN4): Use PWM for speed control
  // IN3 = PWM control, IN4 = digital direction (LOW for forward)
  if (rightIntensity > 0) {
    ledcWrite(HAPTIC_IN3, rightIntensity);  // Forward PWM (0-255)
    digitalWrite(HAPTIC_IN4, LOW);          // Reverse off (LOW = forward direction)
  } else {
    ledcWrite(HAPTIC_IN3, 0);                // Forward off (PWM = 0)
    digitalWrite(HAPTIC_IN4, LOW);          // Keep LOW when stopped
  }
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
  
  // Set haptic motors
  setHapticMotors(leftIntensity, rightIntensity);
  
  Serial.println("OK");
}
 
 