/*
  Turret Debug Script for ESP32
  - Allows safe servo limit testing
  - Manual control mode via serial commands
  - Servos: Top (Pin 3) & Bottom (Pin 7)
  - Motors: Pin 5 & Pin 6 (PWM)
  - LEDs: Pin 8 & Pin 10 (Status indicators)
  
  Serial Commands:
  - HOME: Move both servos to home position (90)
  - TOP:<angle>: Set top servo to angle (0-180)
  - BOTTOM:<angle>: Set bottom servo to angle (0-180)
  - BOTH:<angle>: Set both servos to angle
  - TEST_TOP_MIN: Gradually test top servo minimum limit
  - TEST_TOP_MAX: Gradually test top servo maximum limit
  - TEST_BOTTOM_MIN: Gradually test bottom servo minimum limit
  - TEST_BOTTOM_MAX: Gradually test bottom servo maximum limit
  - SET_MIN:<value>: Set minimum limit for both servos
  - SET_MAX:<value>: Set maximum limit for both servos
  - GET_LIMITS: Print current limits
  - MOTOR1:<speed>: Set motor 1 speed (0-255)
  - MOTOR2:<speed>: Set motor 2 speed (0-255)
  - GET_RANGE: Get ToF sensor range reading (if connected)
  - STATUS: Print current positions and limits
  - HELP: Print available commands
*/

#include <Servo.h>
#include <SoftwareSerial.h>

// --- PIN CONFIGURATION ---
const int PIN_SERVO_TOP = 7;
const int PIN_SERVO_BOTTOM = 3;
const int PIN_MOTOR_1 = 5; 
const int PIN_MOTOR_2 = 6;
const int PIN_LED_1 = 8;  
const int PIN_LED_2 = 10;

// TF03 LiDAR Pins (SoftwareSerial)
const int PIN_LIDAR_RX = 11; // Connect TF03 Brown
const int PIN_LIDAR_TX = 12; // Connect TF03 Blue
bool tof_available = true;  // TF03 LiDAR is connected

// --- SERVO LIMITS (will be updated via commands) ---
//TOP
int top_min = 60;
int top_max = 120;



//BOTTOM
int bottom_min = 0;
int bottom_max = 180;

int servo_home = 90;


// --- Objects ---
Servo topServo;
Servo bottomServo;
SoftwareSerial lidarSerial(PIN_LIDAR_RX, PIN_LIDAR_TX);

// --- Current Positions ---
int topPos = servo_home;
int bottomPos = servo_home;

// --- LiDAR Variables ---
int dist_cm;     // LiDAR Distance in cm
int check;       // Checksum calc
int uart[9];     // LiDAR Data Buffer
const int HEADER = 0x59;

// --- Servo idle tracking (for detach to reduce buzzing) ---
unsigned long topLastMove = 0;
unsigned long bottomLastMove = 0;
const unsigned long SERVO_IDLE_TIME = 1;  // Detach after 200ms idle (reduces jitter from SoftwareSerial)
bool topAttached = true;
bool bottomAttached = true;

// --- Serial Input Buffer (using char array instead of String) ---
char inputBuffer[32];  // Max command length
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  lidarSerial.begin(115200);  // TF03 LiDAR Speed (Factory Default)
  
  // Attach Servos with default min/max pulse widths (500-2500 microseconds)
  // This provides better stability and reduces buzzing
  topServo.attach(PIN_SERVO_TOP, 500, 2500);
  bottomServo.attach(PIN_SERVO_BOTTOM, 500, 2500);
  
  // Setup Pins
  pinMode(PIN_MOTOR_1, OUTPUT);
  pinMode(PIN_MOTOR_2, OUTPUT);
  pinMode(PIN_LED_1, OUTPUT);
  pinMode(PIN_LED_2, OUTPUT);
  
  // Initialize Motors OFF
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  
  // Move to HOME position
  topServo.write(servo_home);
  bottomServo.write(servo_home);
  topPos = servo_home;
  bottomPos = servo_home;
  topLastMove = millis();
  bottomLastMove = millis();
  
  // Status LED
  digitalWrite(PIN_LED_1, HIGH);
  delay(500);
  digitalWrite(PIN_LED_1, LOW);
  
  Serial.println(F("=== TURRET DEBUG MODE ==="));
  Serial.println(F("Type HELP for available commands"));
  Serial.print(F("Top limits - MIN: "));
  Serial.print(top_min);
  Serial.print(F(", MAX: "));
  Serial.println(top_max);
  Serial.print(F("Bottom limits - MIN: "));
  Serial.print(bottom_min);
  Serial.print(F(", MAX: "));
  Serial.println(bottom_max);
  Serial.println(F("Ready for commands..."));
}

void loop() {
  // Check for serial input character by character
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';  // Null terminate
        processCommand(inputBuffer);
        bufferIndex = 0;  // Reset buffer
      }
    } else if (bufferIndex < sizeof(inputBuffer) - 1) {
      // Convert to uppercase and store
      if (c >= 'a' && c <= 'z') {
        c = c - 'a' + 'A';
      }
      inputBuffer[bufferIndex++] = c;
    }
  }
  
  // Detach servos after idle time to reduce buzzing
  unsigned long currentTime = millis();
  
  if (topAttached && (currentTime - topLastMove) > SERVO_IDLE_TIME) {
    topServo.detach();
    topAttached = false;
  }
  
  if (bottomAttached && (currentTime - bottomLastMove) > SERVO_IDLE_TIME) {
    bottomServo.detach();
    bottomAttached = false;
  }
  
  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}

void processCommand(char* cmd) {
  // HOME command
  if (strcmp(cmd, "HOME") == 0) {
    moveToHome();
    Serial.println(F("OK: Moved to home position"));
  }
  
  // TOP servo command
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
  
  // BOTTOM servo command
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
  
  // BOTH servos command
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
  
  // TEST commands - gradual movement to find limits
  else if (strcmp(cmd, "TEST_TOP_MIN") == 0) {
    Serial.println(F("Testing TOP servo MINIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(true, true); // top servo, minimum
  }
  
  else if (strcmp(cmd, "TEST_TOP_MAX") == 0) {
    Serial.println(F("Testing TOP servo MAXIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(true, false); // top servo, maximum
  }
  
  else if (strcmp(cmd, "TEST_BOTTOM_MIN") == 0) {
    Serial.println(F("Testing BOTTOM servo MINIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(false, true); // bottom servo, minimum
  }
  
  else if (strcmp(cmd, "TEST_BOTTOM_MAX") == 0) {
    Serial.println(F("Testing BOTTOM servo MAXIMUM limit..."));
    Serial.println(F("Watch for physical interference. Press any key to stop."));
    testLimit(false, false); // bottom servo, maximum
  }
  
  // SET limits - separate commands for top and bottom
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
  
  // Legacy SET_MIN/SET_MAX for backward compatibility (sets both)
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
  
  // GET limits
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
  
  // MOTOR commands
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
  
  // STATUS command
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
  }
  
  // GET_RANGE command - read ToF sensor
  else if (strcmp(cmd, "GET_RANGE") == 0) {
    float range = readToFRange();
    if (range >= 0) {
      Serial.print(F("OK: Range: "));
      Serial.print(range, 2);
      Serial.println(F(" in"));
    } else {
      Serial.println(F("ERROR: ToF sensor not available"));
    }
  }
  
  // HELP command
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

void setTopServo(int angle) {
  // Clamp to top servo limits
  angle = constrain(angle, top_min, top_max);
  
  // Only update if position actually changed (prevents buzzing)
  if (angle != topPos) {
    // Re-attach if detached
    if (!topAttached) {
      topServo.attach(PIN_SERVO_TOP, 500, 2500);
      topAttached = true;
      delay(50);  // Give servo time to initialize
    }
    
    topServo.write(angle);
    topPos = angle;
    topLastMove = millis();  // Update last move time
    delay(50); // Give servo time to move
  }
}

void setBottomServo(int angle) {
  // Clamp to bottom servo limits
  angle = constrain(angle, bottom_min, bottom_max);
  
  // Only update if position actually changed (prevents buzzing)
  if (angle != bottomPos) {
    // Re-attach if detached
    if (!bottomAttached) {
      bottomServo.attach(PIN_SERVO_BOTTOM, 500, 2500);
      bottomAttached = true;
      delay(50);  // Give servo time to initialize
    }
    
    bottomServo.write(angle);
    bottomPos = angle;
    bottomLastMove = millis();  // Update last move time
    delay(50); // Give servo time to move
  }
}

void moveToHome() {
  setTopServo(servo_home);
  setBottomServo(servo_home);
}

void testLimit(bool isTop, bool isMin) {
  // Start from current position
  int startPos = isTop ? topPos : bottomPos;
  int targetPos;
  int step;
  int currentMin = isTop ? top_min : bottom_min;
  int currentMax = isTop ? top_max : bottom_max;
  
  if (isMin) {
    targetPos = 0;  // Test to absolute minimum
    step = -1;
  } else {
    targetPos = 180;  // Test to absolute maximum
    step = 1;
  }
  
  // Move gradually, checking for serial input to stop
  for (int pos = startPos; pos != targetPos; pos += step) {
    // Check for stop command
    if (Serial.available() > 0) {
      // Clear buffer
      while (Serial.available() > 0) Serial.read();
      Serial.println(F("Test stopped by user"));
      return;
    }
    
    if (isTop) {
      // Re-attach if needed
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
      // Re-attach if needed
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
    
    // Blink LED to show activity
    digitalWrite(PIN_LED_1, HIGH);
    delay(100);
    digitalWrite(PIN_LED_1, LOW);
    delay(100);
  }
  
  // Final position
  if (isTop) {
    topServo.write(targetPos);
    topPos = targetPos;
    Serial.print("TOP reached: ");
  } else {
    bottomServo.write(targetPos);
    bottomPos = targetPos;
    Serial.print("BOTTOM reached: ");
  }
  Serial.println(targetPos);
  Serial.println(F("Test complete. Check for physical interference."));
}

float readToFRange() {
  /*
   * Read TF03 LiDAR and return distance in inches.
   * Uses robust checksum validation to ensure data integrity.
   * 
   * Returns distance in inches, or -1.0 if sensor not available or read failed.
   */
  
  if (!tof_available) {
    return -1.0;
  }
  
  // Try to find a valid LiDAR packet (up to 50ms timeout)
  unsigned long startT = millis();
  while(millis() - startT < 50) { 
    
    if (lidarSerial.available()) {
      // 1. Hunt for Header 1
      if (lidarSerial.read() == HEADER) {
        uart[0] = HEADER;
        
        // 2. Hunt for Header 2
        if (lidarSerial.read() == HEADER) {
          uart[1] = HEADER;
          
          // 3. Read Data Payload
          for (int i = 2; i < 9; i++) {
            uart[i] = lidarSerial.read();
          }
          
          // 4. Verify Checksum
          check = uart[0] + uart[1] + uart[2] + uart[3] + uart[4] + uart[5] + uart[6] + uart[7];
          
          if (uart[8] == (check & 0xff)) {
            // Checksum Passed! Extract distance
            dist_cm = uart[2] + uart[3] * 256;
            
            // Convert cm to inches (1 inch = 2.54 cm)
            float inches = dist_cm / 2.54;
            
            return inches;
          }
        }
      }
    }
  }
  
  // Timeout - no valid packet received
  return -1.0;
}

void printHelp() {
  Serial.println(F("=== AVAILABLE COMMANDS ==="));
  Serial.println(F("HOME                    - Move both servos to home (90)"));
  Serial.println(F("TOP:<angle>            - Set top servo (0-180)"));
  Serial.println(F("BOTTOM:<angle>         - Set bottom servo (0-180)"));
  Serial.println(F("BOTH:<angle>           - Set both servos (0-180)"));
  Serial.println(F("TEST_TOP_MIN           - Test top servo minimum limit"));
  Serial.println(F("TEST_TOP_MAX           - Test top servo maximum limit"));
  Serial.println(F("TEST_BOTTOM_MIN        - Test bottom servo minimum limit"));
  Serial.println(F("TEST_BOTTOM_MAX        - Test bottom servo maximum limit"));
  Serial.println(F("SET_TOP_MIN:<value>    - Set top minimum limit"));
  Serial.println(F("SET_TOP_MAX:<value>    - Set top maximum limit"));
  Serial.println(F("SET_BOTTOM_MIN:<value> - Set bottom minimum limit"));
  Serial.println(F("SET_BOTTOM_MAX:<value> - Set bottom maximum limit"));
  Serial.println(F("SET_MIN:<value>        - Set both minimum limits (legacy)"));
  Serial.println(F("SET_MAX:<value>        - Set both maximum limits (legacy)"));
  Serial.println(F("GET_LIMITS             - Print current limits"));
  Serial.println(F("MOTOR1:<speed>         - Set motor 1 speed (0-255)"));
  Serial.println(F("MOTOR2:<speed>         - Set motor 2 speed (0-255)"));
  Serial.println(F("GET_RANGE              - Get ToF sensor range (inches)"));
  Serial.println(F("STATUS                 - Print current status"));
  Serial.println(F("HELP                   - Show this help"));
  Serial.println(F("========================="));
}

