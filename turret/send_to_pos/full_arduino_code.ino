/*
  Arduino Nano PCB MASTER CODE
  ---------------------------------
  1. Motion: Servos (Pin 3/7) + Motors (Pin 5/6)
  2. Visuals: LEDs (Pin 8/10)
  3. Sensors: GY-521 (A4/A5) + TF03 LiDAR (Pin 11/12)
  ---------------------------------
  * LiDAR Mode: 115200 Baud with Checksum Validation
*/

#include <Servo.h>
#include <Wire.h> 
#include <SoftwareSerial.h>

// --- 1. PIN CONFIGURATION ---
const int PIN_SERVO_TOP = 3;
const int PIN_SERVO_BOTTOM = 7;
const int PIN_MOTOR_1 = 5; 
const int PIN_MOTOR_2 = 6;
const int PIN_LED_1 = 8;  
const int PIN_LED_2 = 10;

// LiDAR Pins (SoftwareSerial)
const int PIN_LIDAR_RX = 11; // Connect TF03 Brown
const int PIN_LIDAR_TX = 12; // Connect TF03 Blue

// --- 2. SETTINGS ---
const int MPU_ADDR = 0x68; 
const int SERVO_MIN = 0;   
const int SERVO_MAX = 180; 
const int SERVO_HOME = 90; 

// --- Objects ---
Servo topServo;
Servo bottomServo;
SoftwareSerial lidarSerial(PIN_LIDAR_RX, PIN_LIDAR_TX); 

// --- Variables ---
int16_t AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ; // Gyro Data
int dist;        // LiDAR Distance
int check;       // Checksum calc
int uart[9];     // LiDAR Data Buffer
const int HEADER = 0x59;

void setup() {
  Serial.begin(9600);       // USB to PC
  lidarSerial.begin(115200); // LiDAR Speed (Factory Default)
  
  Serial.println("--- PCB MASTER CONFIG LOADED ---");

  // Attach Hardware
  topServo.attach(PIN_SERVO_TOP);
  bottomServo.attach(PIN_SERVO_BOTTOM);
  
  pinMode(PIN_MOTOR_1, OUTPUT);
  pinMode(PIN_MOTOR_2, OUTPUT);
  pinMode(PIN_LED_1, OUTPUT);
  pinMode(PIN_LED_2, OUTPUT);

  // Homing
  Serial.println(">>> System Homing...");
  topServo.write(SERVO_HOME); 
  bottomServo.write(SERVO_HOME);
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  
  // Setup GY-521
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0);     
  Wire.endTransmission(true);
  
  delay(1000);
  Serial.println(">>> System Ready.");
}

void loop() {
  // ============================================
  // PHASE 1: SCISSORS (Fast Blinking)
  // ============================================
  Serial.println(">>> 1. Scissors Mode");
  
  for (int pos = SERVO_MIN; pos <= SERVO_MAX; pos += 5) { 
    topServo.write(pos); 
    bottomServo.write(SERVO_MAX - pos);
    blinkFast(pos);
    delay(15);
  }
  for (int pos = SERVO_MAX; pos >= SERVO_MIN; pos -= 5) {
    topServo.write(pos); 
    bottomServo.write(SERVO_MAX - pos);
    blinkFast(pos);
    delay(15);
  }

  // ============================================
  // PHASE 2: CHAOS MODE (Motors + Servos)
  // ============================================
  Serial.println(">>> 2. Chaos Mode");
  
  analogWrite(PIN_MOTOR_1, 160);
  analogWrite(PIN_MOTOR_2, 160);
  
  int safeMin = 45; int safeMax = 135;
  for (int pos = safeMin; pos <= safeMax; pos += 3) { 
    topServo.write(pos); bottomServo.write(pos); 
    blinkFast(pos); delay(10);
  }
  for (int pos = safeMax; pos >= safeMin; pos -= 3) { 
    topServo.write(pos); bottomServo.write(pos); 
    blinkFast(pos); delay(10);
  }
  
  // Stop Motors & Home
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  topServo.write(SERVO_HOME);
  bottomServo.write(SERVO_HOME);
  delay(500);

  // ============================================
  // PHASE 3: SENSOR READ (Slow Blinking)
  // ============================================
  Serial.println(">>> 3. Reading Sensors (Hold Steady)");

  // Loop 30 times (approx 3 seconds)
  for(int i=0; i<30; i++) {
    readFullGyro();    // Update GY-521 Variables
    readLidarRobust(); // Update 'dist' Variable using Checksum
    
    // Slow Blink Logic
    if(i % 5 == 0) {
      digitalWrite(PIN_LED_1, !digitalRead(PIN_LED_1)); // Toggle LED 1
      digitalWrite(PIN_LED_2, !digitalRead(PIN_LED_1)); // Toggle LED 2 opposite
    }
    
    // Print Combined Data
    Serial.print("Dist: "); Serial.print(dist); Serial.print(" cm");
    Serial.print("\t | AcX: "); Serial.print(AcX/100); 
    Serial.print("\t | GyZ: "); Serial.println(GyZ);
    
    delay(100); 
  }

  Serial.println("--- Restarting Sequence ---");
}

// --- Helper Functions ---

void blinkFast(int counter) {
  if ((counter / 10) % 2 == 0) {
    digitalWrite(PIN_LED_1, HIGH); digitalWrite(PIN_LED_2, LOW);
  } else {
    digitalWrite(PIN_LED_1, LOW); digitalWrite(PIN_LED_2, HIGH);
  }
}

void readFullGyro() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); 
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

// --- ROBUST LIDAR READ (CHECKSUM + 115200) ---
void readLidarRobust() {
  // Try to find a valid packet for up to 20ms
  unsigned long startT = millis();
  while(millis() - startT < 20) { 
    
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
            // Checksum Passed! Update Global 'dist'
            dist = uart[2] + uart[3] * 256;
            return; // Exit as soon as we get a good reading
          }
        }
      }
    }
  }
}