/*
  Arduino Nano Complete Turret Control
  =====================================
  Combines servo control for yolo_gimbal.py with full sensor suite
  
  Hardware:
  - Servos: Top (Pin 3), Bottom (Pin 7)
  - Motors: Pin 5 & Pin 6 (PWM vibration)
  - LEDs: Pin 8 & Pin 10 (Status indicators)
  - GY-521 (MPU-6050): A4 (SDA) & A5 (SCL) - Accelerometer/Gyro
  - TF03 LiDAR: Pin 11 (RX, Brown wire) & Pin 12 (TX, Blue wire)
  
  Serial Commands (115200 baud):
  SERVO CONTROL:
    HOME                    - Move both servos to home (90°)
    TOP:<angle>            - Set top servo (0-180°)
    BOTTOM:<angle>         - Set bottom servo (0-180°)
    BOTH:<angle>           - Set both servos (0-180°)
    
  SERVO TESTING:
    TEST_TOP_MIN           - Test top servo minimum limit
    TEST_TOP_MAX           - Test top servo maximum limit
    TEST_BOTTOM_MIN        - Test bottom servo minimum limit
    TEST_BOTTOM_MAX        - Test bottom servo maximum limit
    
  SERVO LIMITS:
    SET_TOP_MIN:<value>    - Set top minimum limit
    SET_TOP_MAX:<value>    - Set top maximum limit
    SET_BOTTOM_MIN:<value> - Set bottom minimum limit
    SET_BOTTOM_MAX:<value> - Set bottom maximum limit
    SET_MIN:<value>        - Set both minimum limits
    SET_MAX:<value>        - Set both maximum limits
    GET_LIMITS             - Print current limits
    
  MOTORS (VIBRATION):
    MOTOR1:<speed>         - Set motor 1 speed (0-255)
    MOTOR2:<speed>         - Set motor 2 speed (0-255)
    VIBRATE:<duration>     - Quick vibration test (ms)
    
  SENSORS:
    DISTANCE               - Get TF03 LiDAR distance (cm)
    GET_RANGE              - Get TF03 LiDAR distance (inches)
    GYRO                   - Get MPU-6050 accelerometer/gyro data
    READ_SENSORS           - Get all sensor data
    
  LEDS:
    LED1:<0|1>            - Control LED 1 (on/off)
    LED2:<0|1>            - Control LED 2 (on/off)
    BLINK:<count>         - Blink LEDs N times
    
  INFO:
    STATUS                 - Print current status
    HELP                   - Show this help
    
  Compatible with yolo_gimbal.py for automatic object tracking!
*/

#include <Servo.h>
#include <Wire.h> 
#include <SoftwareSerial.h>

// ===== PIN CONFIGURATION =====
const int PIN_SERVO_TOP = 3;
const int PIN_SERVO_BOTTOM = 7;
const int PIN_MOTOR_1 = 5; 
const int PIN_MOTOR_2 = 6;
const int PIN_LED_1 = 8;  
const int PIN_LED_2 = 10;
const int PIN_LIDAR_RX = 11;  // TF03 Brown wire
const int PIN_LIDAR_TX = 12;  // TF03 Blue wire

// ===== SETTINGS =====
const int MPU_ADDR = 0x68;  // MPU-6050 I2C address
const int SERVO_HOME = 90;

// ===== SERVO LIMITS (adjustable via commands) =====
int top_min = 60;
int top_max = 120;
int bottom_min = 0;
int bottom_max = 180;

// ===== OBJECTS =====
Servo topServo;
Servo bottomServo;
SoftwareSerial lidarSerial(PIN_LIDAR_RX, PIN_LIDAR_TX);

// ===== STATE VARIABLES =====
// Servo positions
int topPos = SERVO_HOME;
int bottomPos = SERVO_HOME;

// Sensor data - MPU-6050
int16_t AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ;

// Sensor data - TF03 LiDAR
int dist_cm;           // Distance in cm
int uart[9];           // LiDAR data buffer
const int HEADER = 0x59;
int checksum;

// Sensor availability flags
bool gyro_available = true;
bool lidar_available = true;

// Servo idle tracking (detach after idle to reduce buzzing)
unsigned long topLastMove = 0;
unsigned long bottomLastMove = 0;
const unsigned long SERVO_IDLE_TIME = 2000;  // 2 seconds
bool topAttached = true;
bool bottomAttached = true;

// Serial input buffer
char inputBuffer[32];
int bufferIndex = 0;

// ===== SETUP =====
void setup() {
  // Initialize serial
  Serial.begin(115200);
  Serial.setTimeout(100);
  lidarSerial.begin(115200);  // TF03 LiDAR baud rate
  
  Serial.println(F("=== TURRET COMPLETE CONTROL ==="));
  Serial.println(F("Initializing systems..."));
  
  // Setup pins
  pinMode(PIN_MOTOR_1, OUTPUT);
  pinMode(PIN_MOTOR_2, OUTPUT);
  pinMode(PIN_LED_1, OUTPUT);
  pinMode(PIN_LED_2, OUTPUT);
  
  // Initialize motors OFF
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  
  // Attach servos
  topServo.attach(PIN_SERVO_TOP, 500, 2500);
  bottomServo.attach(PIN_SERVO_BOTTOM, 500, 2500);
  topAttached = true;
  bottomAttached = true;
  
  // Move to home position
  topServo.write(SERVO_HOME);
  bottomServo.write(SERVO_HOME);
  topPos = SERVO_HOME;
  bottomPos = SERVO_HOME;
  topLastMove = millis();
  bottomLastMove = millis();
  
  // Initialize MPU-6050 (GY-521)
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // Wake up MPU-6050
  byte error = Wire.endTransmission(true);
  
  if (error == 0) {
    Serial.println(F("  ✓ MPU-6050 (Gyro/Accel) initialized"));
    gyro_available = true;
  } else {
    Serial.println(F("  ✗ MPU-6050 not found (continuing without gyro)"));
    gyro_available = false;
  }
  
  // Test TF03 LiDAR
  delay(100);  // Give LiDAR time to start
  float test_range = readLidarCm();
  if (test_range > 0) {
    Serial.println(F("  ✓ TF03 LiDAR initialized"));
    lidar_available = true;
  } else {
    Serial.println(F("  ✗ TF03 LiDAR not responding (continuing without distance)"));
    lidar_available = false;
  }
  
  // Startup LED sequence
  digitalWrite(PIN_LED_1, HIGH);
  digitalWrite(PIN_LED_2, LOW);
  delay(200);
  digitalWrite(PIN_LED_1, LOW);
  digitalWrite(PIN_LED_2, HIGH);
  delay(200);
  digitalWrite(PIN_LED_1, LOW);
  digitalWrite(PIN_LED_2, LOW);
  
  // Print status
  Serial.println(F("\n=== SYSTEM READY ==="));
  Serial.print(F("Servo limits - Top: "));
  Serial.print(top_min);
  Serial.print(F("-"));
  Serial.print(top_max);
  Serial.print(F("°, Bottom: "));
  Serial.print(bottom_min);
  Serial.print(F("-"));
  Serial.print(bottom_max);
  Serial.println(F("°"));
  Serial.println(F("Type HELP for available commands"));
  Serial.println();
}

// ===== MAIN LOOP =====
void loop() {
  // Process serial commands
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';  // Null terminate
        processCommand(inputBuffer);
        bufferIndex = 0;
      }
    } else if (bufferIndex < sizeof(inputBuffer) - 1) {
      // Convert to uppercase
      if (c >= 'a' && c <= 'z') {
        c = c - 'a' + 'A';
      }
      inputBuffer[bufferIndex++] = c;
    }
  }
  
  // Auto-detach servos after idle to reduce buzzing
  unsigned long currentTime = millis();
  
  if (topAttached && (currentTime - topLastMove) > SERVO_IDLE_TIME) {
    topServo.detach();
    topAttached = false;
  }
  
  if (bottomAttached && (currentTime - bottomLastMove) > SERVO_IDLE_TIME) {
    bottomServo.detach();
    bottomAttached = false;
  }
  
  delay(10);
}

// ===== COMMAND PROCESSOR =====
void processCommand(char* cmd) {
  // ===== SERVO CONTROL =====
  if (strcmp(cmd, "HOME") == 0) {
    moveToHome();
    Serial.println(F("OK: Moved to home position"));
  }
  
  else if (strncmp(cmd, "TOP:", 4) == 0) {
    int angle = atoi(cmd + 4);
    if (angle >= 0 && angle <= 180) {
      setTopServo(angle);
      Serial.print(F("OK: Top servo set to "));
      Serial.println(angle);
    } else {
      Serial.println(F("ERROR: Angle must be 0-180"));
    }
  }
  
  else if (strncmp(cmd, "BOTTOM:", 7) == 0) {
    int angle = atoi(cmd + 7);
    if (angle >= 0 && angle <= 180) {
      setBottomServo(angle);
      Serial.print(F("OK: Bottom servo set to "));
      Serial.println(angle);
    } else {
      Serial.println(F("ERROR: Angle must be 0-180"));
    }
  }
  
  else if (strncmp(cmd, "BOTH:", 5) == 0) {
    int angle = atoi(cmd + 5);
    if (angle >= 0 && angle <= 180) {
      setTopServo(angle);
      setBottomServo(angle);
      Serial.print(F("OK: Both servos set to "));
      Serial.println(angle);
    } else {
      Serial.println(F("ERROR: Angle must be 0-180"));
    }
  }
  
  // ===== SERVO TESTING =====
  else if (strcmp(cmd, "TEST_TOP_MIN") == 0) {
    Serial.println(F("Testing TOP servo MINIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(true, true);
  }
  
  else if (strcmp(cmd, "TEST_TOP_MAX") == 0) {
    Serial.println(F("Testing TOP servo MAXIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(true, false);
  }
  
  else if (strcmp(cmd, "TEST_BOTTOM_MIN") == 0) {
    Serial.println(F("Testing BOTTOM servo MINIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(false, true);
  }
  
  else if (strcmp(cmd, "TEST_BOTTOM_MAX") == 0) {
    Serial.println(F("Testing BOTTOM servo MAXIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(false, false);
  }
  
  // ===== SERVO LIMITS =====
  else if (strncmp(cmd, "SET_TOP_MIN:", 12) == 0) {
    int minVal = atoi(cmd + 12);
    if (minVal >= 0 && minVal < top_max) {
      top_min = minVal;
      Serial.print(F("OK: Top minimum limit set to "));
      Serial.println(top_min);
    } else {
      Serial.println(F("ERROR: MIN must be 0-179 and less than MAX"));
    }
  }
  
  else if (strncmp(cmd, "SET_TOP_MAX:", 12) == 0) {
    int maxVal = atoi(cmd + 12);
    if (maxVal > top_min && maxVal <= 180) {
      top_max = maxVal;
      Serial.print(F("OK: Top maximum limit set to "));
      Serial.println(top_max);
    } else {
      Serial.println(F("ERROR: MAX must be 1-180 and greater than MIN"));
    }
  }
  
  else if (strncmp(cmd, "SET_BOTTOM_MIN:", 15) == 0) {
    int minVal = atoi(cmd + 15);
    if (minVal >= 0 && minVal < bottom_max) {
      bottom_min = minVal;
      Serial.print(F("OK: Bottom minimum limit set to "));
      Serial.println(bottom_min);
    } else {
      Serial.println(F("ERROR: MIN must be 0-179 and less than MAX"));
    }
  }
  
  else if (strncmp(cmd, "SET_BOTTOM_MAX:", 15) == 0) {
    int maxVal = atoi(cmd + 15);
    if (maxVal > bottom_min && maxVal <= 180) {
      bottom_max = maxVal;
      Serial.print(F("OK: Bottom maximum limit set to "));
      Serial.println(bottom_max);
    } else {
      Serial.println(F("ERROR: MAX must be 1-180 and greater than MIN"));
    }
  }
  
  else if (strncmp(cmd, "SET_MIN:", 8) == 0) {
    int minVal = atoi(cmd + 8);
    if (minVal >= 0 && minVal < 180) {
      top_min = minVal;
      bottom_min = minVal;
      Serial.print(F("OK: Both minimum limits set to "));
      Serial.println(minVal);
    } else {
      Serial.println(F("ERROR: MIN must be 0-179"));
    }
  }
  
  else if (strncmp(cmd, "SET_MAX:", 8) == 0) {
    int maxVal = atoi(cmd + 8);
    if (maxVal > 0 && maxVal <= 180) {
      top_max = maxVal;
      bottom_max = maxVal;
      Serial.print(F("OK: Both maximum limits set to "));
      Serial.println(maxVal);
    } else {
      Serial.println(F("ERROR: MAX must be 1-180"));
    }
  }
  
  else if (strcmp(cmd, "GET_LIMITS") == 0) {
    Serial.println(F("Current limits:"));
    Serial.print(F("  Top - MIN: "));
    Serial.print(top_min);
    Serial.print(F(", MAX: "));
    Serial.println(top_max);
    Serial.print(F("  Bottom - MIN: "));
    Serial.print(bottom_min);
    Serial.print(F(", MAX: "));
    Serial.println(bottom_max);
  }
  
  // ===== MOTORS (VIBRATION) =====
  else if (strncmp(cmd, "MOTOR1:", 7) == 0) {
    int speed = atoi(cmd + 7);
    if (speed >= 0 && speed <= 255) {
      analogWrite(PIN_MOTOR_1, speed);
      Serial.print(F("OK: Motor 1 set to "));
      Serial.println(speed);
    } else {
      Serial.println(F("ERROR: Speed must be 0-255"));
    }
  }
  
  else if (strncmp(cmd, "MOTOR2:", 7) == 0) {
    int speed = atoi(cmd + 7);
    if (speed >= 0 && speed <= 255) {
      analogWrite(PIN_MOTOR_2, speed);
      Serial.print(F("OK: Motor 2 set to "));
      Serial.println(speed);
    } else {
      Serial.println(F("ERROR: Speed must be 0-255"));
    }
  }
  
  else if (strncmp(cmd, "VIBRATE:", 8) == 0) {
    int duration = atoi(cmd + 8);
    if (duration > 0 && duration <= 5000) {
      Serial.print(F("OK: Vibrating for "));
      Serial.print(duration);
      Serial.println(F("ms"));
      analogWrite(PIN_MOTOR_1, 200);
      analogWrite(PIN_MOTOR_2, 200);
      delay(duration);
      analogWrite(PIN_MOTOR_1, 0);
      analogWrite(PIN_MOTOR_2, 0);
    } else {
      Serial.println(F("ERROR: Duration must be 1-5000ms"));
    }
  }
  
  // ===== SENSORS =====
  else if (strcmp(cmd, "DISTANCE") == 0) {
    // Read TF03 LiDAR and return distance in cm (for yolo_gimbal.py)
    if (!lidar_available) {
      Serial.println(F("ERROR: LiDAR not available"));
      return;
    }
    
    float distance = readLidarCm();
    if (distance > 0) {
      Serial.print(F("OK: DISTANCE:"));
      Serial.print(distance, 1);
      Serial.println(F(" cm"));
    } else {
      Serial.println(F("ERROR: Failed to read LiDAR"));
    }
  }
  
  else if (strcmp(cmd, "GET_RANGE") == 0) {
    // Read TF03 LiDAR and return distance in inches (legacy)
    if (!lidar_available) {
      Serial.println(F("ERROR: LiDAR not available"));
      return;
    }
    
    float distance_cm = readLidarCm();
    if (distance_cm > 0) {
      float inches = distance_cm / 2.54;
      Serial.print(F("OK: Range: "));
      Serial.print(inches, 2);
      Serial.println(F(" in"));
    } else {
      Serial.println(F("ERROR: Failed to read LiDAR"));
    }
  }
  
  else if (strcmp(cmd, "GYRO") == 0) {
    // Read MPU-6050 accelerometer/gyro
    if (!gyro_available) {
      Serial.println(F("ERROR: MPU-6050 not available"));
      return;
    }
    
    readGyro();
    Serial.println(F("OK: Gyro/Accel data:"));
    Serial.print(F("  AcX: ")); Serial.print(AcX);
    Serial.print(F(" | AcY: ")); Serial.print(AcY);
    Serial.print(F(" | AcZ: ")); Serial.println(AcZ);
    Serial.print(F("  GyX: ")); Serial.print(GyX);
    Serial.print(F(" | GyY: ")); Serial.print(GyY);
    Serial.print(F(" | GyZ: ")); Serial.println(GyZ);
    Serial.print(F("  Temp: ")); Serial.print(Tmp/340.0 + 36.53); Serial.println(F("°C"));
  }
  
  else if (strcmp(cmd, "READ_SENSORS") == 0) {
    // Read all sensors at once
    Serial.println(F("=== ALL SENSORS ==="));
    
    // LiDAR
    if (lidar_available) {
      float distance = readLidarCm();
      if (distance > 0) {
        Serial.print(F("Distance: "));
        Serial.print(distance, 1);
        Serial.print(F(" cm ("));
        Serial.print(distance/2.54, 2);
        Serial.println(F(" in)"));
      } else {
        Serial.println(F("Distance: Read failed"));
      }
    } else {
      Serial.println(F("Distance: Sensor not available"));
    }
    
    // Gyro/Accel
    if (gyro_available) {
      readGyro();
      Serial.print(F("Accel: X="));
      Serial.print(AcX);
      Serial.print(F(" Y="));
      Serial.print(AcY);
      Serial.print(F(" Z="));
      Serial.println(AcZ);
      Serial.print(F("Gyro:  X="));
      Serial.print(GyX);
      Serial.print(F(" Y="));
      Serial.print(GyY);
      Serial.print(F(" Z="));
      Serial.println(GyZ);
      Serial.print(F("Temp:  "));
      Serial.print(Tmp/340.0 + 36.53);
      Serial.println(F("°C"));
    } else {
      Serial.println(F("Gyro/Accel: Sensor not available"));
    }
  }
  
  // ===== LEDS =====
  else if (strncmp(cmd, "LED1:", 5) == 0) {
    int state = atoi(cmd + 5);
    digitalWrite(PIN_LED_1, state ? HIGH : LOW);
    Serial.print(F("OK: LED1 "));
    Serial.println(state ? F("ON") : F("OFF"));
  }
  
  else if (strncmp(cmd, "LED2:", 5) == 0) {
    int state = atoi(cmd + 5);
    digitalWrite(PIN_LED_2, state ? HIGH : LOW);
    Serial.print(F("OK: LED2 "));
    Serial.println(state ? F("ON") : F("OFF"));
  }
  
  else if (strncmp(cmd, "BLINK:", 6) == 0) {
    int count = atoi(cmd + 6);
    if (count > 0 && count <= 20) {
      Serial.print(F("OK: Blinking "));
      Serial.print(count);
      Serial.println(F(" times"));
      for (int i = 0; i < count; i++) {
        digitalWrite(PIN_LED_1, HIGH);
        digitalWrite(PIN_LED_2, LOW);
        delay(100);
        digitalWrite(PIN_LED_1, LOW);
        digitalWrite(PIN_LED_2, HIGH);
        delay(100);
      }
      digitalWrite(PIN_LED_1, LOW);
      digitalWrite(PIN_LED_2, LOW);
    } else {
      Serial.println(F("ERROR: Count must be 1-20"));
    }
  }
  
  // ===== INFO =====
  else if (strcmp(cmd, "STATUS") == 0) {
    Serial.println(F("=== STATUS ==="));
    Serial.print(F("Top servo position: "));
    Serial.println(topPos);
    Serial.print(F("Bottom servo position: "));
    Serial.println(bottomPos);
    Serial.print(F("Top limits - MIN: "));
    Serial.print(top_min);
    Serial.print(F(", MAX: "));
    Serial.println(top_max);
    Serial.print(F("Bottom limits - MIN: "));
    Serial.print(bottom_min);
    Serial.print(F(", MAX: "));
    Serial.println(bottom_max);
    Serial.print(F("MPU-6050: "));
    Serial.println(gyro_available ? F("Available") : F("Not available"));
    Serial.print(F("TF03 LiDAR: "));
    Serial.println(lidar_available ? F("Available") : F("Not available"));
  }
  
  else if (strcmp(cmd, "HELP") == 0) {
    printHelp();
  }
  
  // Unknown command
  else {
    Serial.print(F("ERROR: Unknown command '"));
    Serial.print(cmd);
    Serial.println(F("'. Type HELP for available commands."));
  }
}

// ===== SERVO FUNCTIONS =====
void setTopServo(int angle) {
  angle = constrain(angle, top_min, top_max);
  
  if (angle != topPos) {
    if (!topAttached) {
      topServo.attach(PIN_SERVO_TOP, 500, 2500);
      topAttached = true;
      delay(50);
    }
    
    topServo.write(angle);
    topPos = angle;
    topLastMove = millis();
    delay(15);
  }
}

void setBottomServo(int angle) {
  angle = constrain(angle, bottom_min, bottom_max);
  
  if (angle != bottomPos) {
    if (!bottomAttached) {
      bottomServo.attach(PIN_SERVO_BOTTOM, 500, 2500);
      bottomAttached = true;
      delay(50);
    }
    
    bottomServo.write(angle);
    bottomPos = angle;
    bottomLastMove = millis();
    delay(15);
  }
}

void moveToHome() {
  setTopServo(SERVO_HOME);
  setBottomServo(SERVO_HOME);
}

void testLimit(bool isTop, bool isMin) {
  int startPos = isTop ? topPos : bottomPos;
  int targetPos = isMin ? 0 : 180;
  int step = isMin ? -1 : 1;
  
  for (int pos = startPos; pos != targetPos; pos += step) {
    if (Serial.available() > 0) {
      while (Serial.available() > 0) Serial.read();
      Serial.println(F("Test stopped by user"));
      return;
    }
    
    if (isTop) {
      if (!topAttached) {
        topServo.attach(PIN_SERVO_TOP, 500, 2500);
        topAttached = true;
        delay(50);
      }
      topServo.write(pos);
      topPos = pos;
      topLastMove = millis();
      Serial.print("TOP: ");
    } else {
      if (!bottomAttached) {
        bottomServo.attach(PIN_SERVO_BOTTOM, 500, 2500);
        bottomAttached = true;
        delay(50);
      }
      bottomServo.write(pos);
      bottomPos = pos;
      bottomLastMove = millis();
      Serial.print("BOTTOM: ");
    }
    Serial.println(pos);
    
    digitalWrite(PIN_LED_1, HIGH);
    delay(100);
    digitalWrite(PIN_LED_1, LOW);
    delay(100);
  }
  
  Serial.println(F("Test complete. Check for physical interference."));
}

// ===== SENSOR FUNCTIONS =====
void readGyro() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // Starting register
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  
  if (Wire.available() >= 14) {
    AcX = Wire.read() << 8 | Wire.read();
    AcY = Wire.read() << 8 | Wire.read();
    AcZ = Wire.read() << 8 | Wire.read();
    Tmp = Wire.read() << 8 | Wire.read();
    GyX = Wire.read() << 8 | Wire.read();
    GyY = Wire.read() << 8 | Wire.read();
    GyZ = Wire.read() << 8 | Wire.read();
  }
}

float readLidarCm() {
  /*
   * Read TF03 LiDAR with checksum validation
   * Returns distance in cm, or -1.0 if read failed
   */
  
  unsigned long startT = millis();
  while(millis() - startT < 50) {  // 50ms timeout
    
    if (lidarSerial.available()) {
      // Hunt for Header 1
      if (lidarSerial.read() == HEADER) {
        uart[0] = HEADER;
        
        // Hunt for Header 2
        if (lidarSerial.read() == HEADER) {
          uart[1] = HEADER;
          
          // Read data payload
          for (int i = 2; i < 9; i++) {
            uart[i] = lidarSerial.read();
          }
          
          // Verify checksum
          checksum = uart[0] + uart[1] + uart[2] + uart[3] + uart[4] + uart[5] + uart[6] + uart[7];
          
          if (uart[8] == (checksum & 0xff)) {
            // Checksum passed!
            dist_cm = uart[2] + uart[3] * 256;
            return (float)dist_cm;
          }
        }
      }
    }
  }
  
  return -1.0;  // Timeout or read failed
}

// ===== HELP =====
void printHelp() {
  Serial.println(F("=== AVAILABLE COMMANDS ==="));
  Serial.println(F("SERVO CONTROL:"));
  Serial.println(F("  HOME                    - Move both servos to home (90°)"));
  Serial.println(F("  TOP:<angle>            - Set top servo (0-180°)"));
  Serial.println(F("  BOTTOM:<angle>         - Set bottom servo (0-180°)"));
  Serial.println(F("  BOTH:<angle>           - Set both servos (0-180°)"));
  Serial.println();
  Serial.println(F("SERVO TESTING:"));
  Serial.println(F("  TEST_TOP_MIN           - Test top servo minimum limit"));
  Serial.println(F("  TEST_TOP_MAX           - Test top servo maximum limit"));
  Serial.println(F("  TEST_BOTTOM_MIN        - Test bottom servo minimum limit"));
  Serial.println(F("  TEST_BOTTOM_MAX        - Test bottom servo maximum limit"));
  Serial.println();
  Serial.println(F("SERVO LIMITS:"));
  Serial.println(F("  SET_TOP_MIN:<value>    - Set top minimum limit"));
  Serial.println(F("  SET_TOP_MAX:<value>    - Set top maximum limit"));
  Serial.println(F("  SET_BOTTOM_MIN:<value> - Set bottom minimum limit"));
  Serial.println(F("  SET_BOTTOM_MAX:<value> - Set bottom maximum limit"));
  Serial.println(F("  SET_MIN:<value>        - Set both minimum limits"));
  Serial.println(F("  SET_MAX:<value>        - Set both maximum limits"));
  Serial.println(F("  GET_LIMITS             - Print current limits"));
  Serial.println();
  Serial.println(F("MOTORS (VIBRATION):"));
  Serial.println(F("  MOTOR1:<speed>         - Set motor 1 speed (0-255)"));
  Serial.println(F("  MOTOR2:<speed>         - Set motor 2 speed (0-255)"));
  Serial.println(F("  VIBRATE:<duration>     - Quick vibration test (ms)"));
  Serial.println();
  Serial.println(F("SENSORS:"));
  Serial.println(F("  DISTANCE               - Get TF03 LiDAR distance (cm)"));
  Serial.println(F("  GET_RANGE              - Get TF03 LiDAR distance (inches)"));
  Serial.println(F("  GYRO                   - Get MPU-6050 gyro/accel data"));
  Serial.println(F("  READ_SENSORS           - Get all sensor data"));
  Serial.println();
  Serial.println(F("LEDS:"));
  Serial.println(F("  LED1:<0|1>            - Control LED 1"));
  Serial.println(F("  LED2:<0|1>            - Control LED 2"));
  Serial.println(F("  BLINK:<count>         - Blink LEDs N times"));
  Serial.println();
  Serial.println(F("INFO:"));
  Serial.println(F("  STATUS                 - Print current status"));
  Serial.println(F("  HELP                   - Show this help"));
  Serial.println(F("========================="));
}

