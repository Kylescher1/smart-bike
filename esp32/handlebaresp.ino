/*
  Arduino Nano "Intense" Debug + Sensor
  - Servos: Pin 3 & Pin 7
  - Motors: Pin 5 & Pin 6 (PWM)
  - Sensor: GY-521 on A4 & A5
*/

#include <Servo.h>
#include <Wire.h> // Required for GY-521 sensor

// --- Configuration ---
const int servoPin1 = 3;
const int servoPin2 = 7;
const int motorPin1 = 5; 
const int motorPin2 = 6;
const int MPU_ADDR = 0x68; // I2C address of the GY-521

// --- Objects ---
Servo myservo1;
Servo myservo2;

// --- Sensor Variables ---
int16_t AcX, AcY, AcZ;

void setup() {
  Serial.begin(9600);
  Serial.println("--- Starting Intense Debug + Sensor ---");

  // 1. Setup Servos
  myservo1.attach(servoPin1);
  myservo2.attach(servoPin2);
  
  // 2. Setup Motors
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);

  // 3. Initialize Outputs
  analogWrite(motorPin1, 0);
  analogWrite(motorPin2, 0);
  myservo1.write(90); // Start centered
  myservo2.write(90);
  
  // 4. Setup GY-521
  // Note: If wiring is bad, code might hang here. 
  // Watch Serial Monitor for "Sensor Started".
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // Wake up MPU-6050
  Wire.endTransmission(true);
  Serial.println(">>> Sensor Started");
  
  delay(1000);
}

void loop() {
  // ============================================
  // PHASE 1: INTENSE SERVO MOVES
  // ============================================
  
  Serial.println(">>> 1. Scissors (Opposite Directions)");
  // Servo 1 goes 0->180, Servo 2 goes 180->0
  for (int pos = 0; pos <= 180; pos += 5) { // Step 5 is fast
    myservo1.write(pos);
    myservo2.write(180 - pos);
    delay(15);
  }
  // Reverse
  for (int pos = 180; pos >= 0; pos -= 5) {
    myservo1.write(pos);
    myservo2.write(180 - pos);
    delay(15);
  }

  Serial.println(">>> 2. The Twitch (Holding Torque)");
  // Shake quickly around center
  for(int i = 0; i < 6; i++) {
    myservo1.write(80); myservo2.write(100);
    delay(80);
    myservo1.write(100); myservo2.write(80);
    delay(80);
  }
  // Recenter
  myservo1.write(90); myservo2.write(90);
  delay(200);

  // ============================================
  // PHASE 2: CHAOS MODE (Motors + Servos)
  // ============================================
  
  Serial.println(">>> 3. Chaos Mode (Full Load)");
  
  // Turn motors ON (60% speed)
  analogWrite(motorPin1, 160);
  analogWrite(motorPin2, 160);
  
  // Sweep servos while motors are running
  for (int pos = 45; pos <= 135; pos += 3) { 
    myservo1.write(pos);
    myservo2.write(pos);
    delay(10);
  }
  for (int pos = 135; pos >= 45; pos -= 3) { 
    myservo1.write(pos);
    myservo2.write(pos);
    delay(10);
  }
  
  // Turn motors OFF
  analogWrite(motorPin1, 0);
  analogWrite(motorPin2, 0);
  delay(500);

  // ============================================
  // PHASE 3: SENSOR READ
  // ============================================
  
  Serial.println(">>> 4. Reading Sensor (Check Serial Monitor)");
  // We read 20 times to give you time to tilt the robot
  for(int i=0; i<20; i++) {
    readSensor();
    
    // Visualize TILT roughly
    Serial.print("Tilt X: "); Serial.print(AcX / 100); 
    Serial.print("\t Tilt Y: "); Serial.println(AcY / 100);
    
    delay(100); 
  }

  Serial.println("--- Restarting Loop ---");
  delay(1000);
}

// --- Helper Functions ---

void readSensor() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // Starting register for Accel Readings
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true); // Request 6 bytes (AcX, AcY, AcZ)
  
  // If sensor is disconnected, these lines might return garbage or hang
  if (Wire.available() >= 6) {
    AcX = Wire.read() << 8 | Wire.read();
    AcY = Wire.read() << 8 | Wire.read();
    AcZ = Wire.read() << 8 | Wire.read();
  }
}