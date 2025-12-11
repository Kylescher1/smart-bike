/*
  Arduino Nano PCB Final Debug
  - Servos: Top (Pin 3) & Bottom (Pin 7)
  - Motors: Pin 5 & Pin 6 (PWM)
  - Sensor: GY-521 (A4/A5) - Full Data
  - LEDs: Pin 8 & Pin 10 (Blinking Status)
*/

#include <Servo.h>
#include <Wire.h> 

// --- 1. PIN CONFIGURATION ---
const int PIN_SERVO_TOP = 3;
const int PIN_SERVO_BOTTOM = 7;
const int PIN_MOTOR_1 = 5; 
const int PIN_MOTOR_2 = 6;
const int PIN_LED_1 = 8;  
const int PIN_LED_2 = 10;
const int MPU_ADDR = 0x68; 

// --- 2. LIMITS & SETTINGS ---
const int SERVO_MIN = 0;   // Change this if it hits mechanical limits
const int SERVO_MAX = 180; // Change this if it hits mechanical limits
const int SERVO_HOME = 90; // The safe "middle" position

// --- Objects ---
Servo topServo;
Servo bottomServo;

// --- Data Variables ---
int16_t AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ;

void setup() {
  Serial.begin(9600);
  Serial.println("--- PCB Final Config: Limits & Blinking ---");

  // Attach Servos
  topServo.attach(PIN_SERVO_TOP);
  bottomServo.attach(PIN_SERVO_BOTTOM);
  
  // Setup Pins
  pinMode(PIN_MOTOR_1, OUTPUT);
  pinMode(PIN_MOTOR_2, OUTPUT);
  pinMode(PIN_LED_1, OUTPUT);
  pinMode(PIN_LED_2, OUTPUT);

  // 1. Move to HOME immediately
  Serial.println(">>> Homing Servos...");
  topServo.write(SERVO_HOME); 
  bottomServo.write(SERVO_HOME);
  
  // Initialize Motors OFF
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  
  // Setup GY-521 Sensor
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0);     
  Wire.endTransmission(true);
  Serial.println(">>> Sensor Started");
  
  delay(1000);
}

void loop() {
  // ============================================
  // PHASE 1: SCISSORS (Fast Blinking)
  // ============================================
  Serial.println(">>> 1. Scissors Mode");
  
  // Sweep Forward
  for (int pos = SERVO_MIN; pos <= SERVO_MAX; pos += 5) { 
    topServo.write(pos); 
    bottomServo.write(SERVO_MAX - pos); // Invert direction for bottom
    
    blinkFast(pos); // Flash LEDs based on position
    delay(15);
  }
  
  // Sweep Backward
  for (int pos = SERVO_MAX; pos >= SERVO_MIN; pos -= 5) {
    topServo.write(pos); 
    bottomServo.write(SERVO_MAX - pos);
    
    blinkFast(pos);
    delay(15);
  }

  // ============================================
  // PHASE 2: CHAOS MODE (Motors + Servos + Blinking)
  // ============================================
  Serial.println(">>> 2. Chaos Mode (Full Load)");
  
  // Turn Motors ON
  analogWrite(PIN_MOTOR_1, 160);
  analogWrite(PIN_MOTOR_2, 160);
  
  // Sweep Range (Limited to middle 90 degrees for safety)
  int safeMin = 45; 
  int safeMax = 135;

  for (int pos = safeMin; pos <= safeMax; pos += 3) { 
    topServo.write(pos); 
    bottomServo.write(pos); 
    blinkFast(pos);
    delay(10);
  }
  for (int pos = safeMax; pos >= safeMin; pos -= 3) { 
    topServo.write(pos); 
    bottomServo.write(pos); 
    blinkFast(pos);
    delay(10);
  }
  
  // Stop Motors
  analogWrite(PIN_MOTOR_1, 0);
  analogWrite(PIN_MOTOR_2, 0);
  
  // Return to Home before reading sensors
  topServo.write(SERVO_HOME);
  bottomServo.write(SERVO_HOME);
  delay(500);

  // ============================================
  // PHASE 3: SENSOR READ (Slow Blinking)
  // ============================================
  Serial.println(">>> 3. Reading Sensors");

  // Read loop (approx 4 seconds)
  for(int i=0; i<20; i++) {
    readFullSensor();
    
    // Toggle LEDs every other loop (Slow Blink)
    if(i % 2 == 0) {
      digitalWrite(PIN_LED_1, HIGH);
      digitalWrite(PIN_LED_2, LOW);
    } else {
      digitalWrite(PIN_LED_1, LOW);
      digitalWrite(PIN_LED_2, HIGH);
    }
    
    // Print Data
    Serial.print("AcX:"); Serial.print(AcX);
    Serial.print(" | AcY:"); Serial.print(AcY);
    Serial.print(" | GyZ:"); Serial.println(GyZ); // Printing rotation Z
    
    delay(200); 
  }

  Serial.println("--- Loop Complete ---");
}

// --- Helper Functions ---

// Flashes LEDs rapidly based on the servo position number
void blinkFast(int counter) {
  // If counter is even, LED 1 ON. If odd, LED 2 ON.
  // We divide by 10 to make the blink visible to human eye
  if ((counter / 10) % 2 == 0) {
    digitalWrite(PIN_LED_1, HIGH);
    digitalWrite(PIN_LED_2, LOW);
  } else {
    digitalWrite(PIN_LED_1, LOW);
    digitalWrite(PIN_LED_2, HIGH);
  }
}

void readFullSensor() {
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