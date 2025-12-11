/*
 * ESP32 SIMPLIFIED DIRECT DRIVE
 * - Servos: Manual PWM (No Library) to prevent conflicts.
 * - Haptics: Direct PWM (No H-Bridge).
 * * WARNING: Driving motors directly from GPIO pins may damage the ESP32
 * if the motors draw >20mA. Use at your own risk.
 */

 #include <Wire.h>


 // ===================== PINS =====================
 #define SERVO_TOP_PIN    32
 #define SERVO_BOTTOM_PIN 33
 #define LED_GREEN 12   // Change to 27 ONLY if not used by haptics
 
 
 // HAPTICS (Positive wire to PIN, Negative to GND)
 #define HAPTIC_PIN_LEFT  25 
 #define HAPTIC_PIN_RIGHT 26 // Changed from 14/27 to 26 for simplicity
 
 // ===================== SETTINGS =====================
 // HAPTICS: 20kHz, 8-bit
 #define FREQ_HAPTIC     20000
 #define RES_HAPTIC      8
 
 // SERVOS: 50Hz, 14-bit (Safe Mode)
 #define FREQ_SERVO      50
 #define RES_SERVO       14 
 
 // MPU6050
 #define MPU_ADDR 0x68
 
 // ===================== SERVO MATH =====================
 // 50Hz = 20,000us period. 14-bit max = 16383.
 const int MIN_COUNTS = 410;   // ~500us
 const int MAX_COUNTS = 1966;  // ~2400us
 
 const int S1_MIN_DEG = 0;
 const int S1_MAX_DEG = 180;
 const int S2_MIN_DEG = 5;
 const int S2_MAX_DEG = 60;
 
 uint32_t degreesToDuty(int angle, int minAng, int maxAng) {
   angle = constrain(angle, minAng, maxAng);
   return map(angle, minAng, maxAng, MIN_COUNTS, MAX_COUNTS);
 }
 
 // ===================== SETUP =====================
 void setup() {
   Serial.begin(115200);
 
     // ----- GREEN LED STARTUP FLASH -----
   pinMode(LED_GREEN, OUTPUT);
 
   // Flash LED 3 times at startup
   for (int i = 0; i < 3; i++) {
     digitalWrite(LED_GREEN, HIGH);
     delay(200);
     digitalWrite(LED_GREEN, LOW);
     delay(200);
   }
 
 
   while (!Serial) delay(10);
   Serial.println("\n=== DIRECT DRIVE MODE ===");
 
   // 1. SETUP HAPTICS
   // We simply attach the pins. No direction pins needed.
   if (!ledcAttach(HAPTIC_PIN_LEFT, FREQ_HAPTIC, RES_HAPTIC)) Serial.println("! Left Haptic Fail");
   if (!ledcAttach(HAPTIC_PIN_RIGHT, FREQ_HAPTIC, RES_HAPTIC)) Serial.println("! Right Haptic Fail");
 
   // Ensure off at start
   ledcWrite(HAPTIC_PIN_LEFT, 0);
   ledcWrite(HAPTIC_PIN_RIGHT, 0);
 
   // 2. SETUP SERVOS
   if (ledcAttach(SERVO_TOP_PIN, FREQ_SERVO, RES_SERVO)) {
     ledcWrite(SERVO_TOP_PIN, degreesToDuty(35, S2_MIN_DEG, S2_MAX_DEG));
     Serial.println("✓ Top Servo");
   }
 
   if (ledcAttach(SERVO_BOTTOM_PIN, FREQ_SERVO, RES_SERVO)) {
     ledcWrite(SERVO_BOTTOM_PIN, degreesToDuty(90, S1_MIN_DEG, S1_MAX_DEG));
     Serial.println("✓ Bot Servo");
   }
 
   // 3. I2C SETUP
   Wire.begin(4, 15);
   Wire.setClock(400000);
   Wire.beginTransmission(MPU_ADDR);
   Wire.write(0x6B); Wire.write(0); Wire.endTransmission();
   
   Serial.println("=== READY ===");
   digitalWrite(LED_GREEN, HIGH);  // Solid ON during normal run
 
 }
 
 // ===================== LOOP =====================
 void loop() {
   if (Serial.available()) {
     String cmd = Serial.readStringUntil('\n');
     cmd.trim();
     if (cmd.length() > 0) processCmd(cmd);
   }
 }
 
 void processCmd(String cmd) {
   cmd.toUpperCase();
   
   if (cmd == "READ") {
     readMPU();
   }
   else if (cmd.startsWith("MOVE,")) {
     handleMove(cmd);
   }
   else if (cmd.startsWith("VIBRATE,")) {
     // VIBRATE,100,200
     int split = cmd.indexOf(',', 8);
     if (split > 0) {
       int left = cmd.substring(8, split).toInt();
       int right = cmd.substring(split + 1).toInt();
       
       ledcWrite(HAPTIC_PIN_LEFT, constrain(left, 0, 255));
       ledcWrite(HAPTIC_PIN_RIGHT, constrain(right, 0, 255));
       Serial.println("OK");
     }
   }
 }
 
 void handleMove(String cmd) {
   int bIdx = cmd.indexOf(",B,");
   int tIdx = cmd.indexOf(",T,");
   if (bIdx > 0 && tIdx > 0) {
     int targetB = cmd.substring(bIdx + 3, tIdx).toInt();
     int targetT = cmd.substring(tIdx + 3).toInt();
 
     // Soft-ish start to help power
     ledcWrite(SERVO_BOTTOM_PIN, degreesToDuty(targetB, S1_MIN_DEG, S1_MAX_DEG));
     delay(15); 
     ledcWrite(SERVO_TOP_PIN, degreesToDuty(targetT, S2_MIN_DEG, S2_MAX_DEG));
     Serial.println("OK");
   }
 }
 
 void readMPU() {
   Wire.beginTransmission(MPU_ADDR);
   Wire.write(0x3B);
   if (Wire.endTransmission(false) != 0) return;
   
   if (Wire.requestFrom(MPU_ADDR, 14) == 14) {
     int16_t ax = (Wire.read()<<8)|Wire.read();
     int16_t ay = (Wire.read()<<8)|Wire.read();
     int16_t az = (Wire.read()<<8)|Wire.read();
     Wire.read(); Wire.read(); 
     int16_t gx = (Wire.read()<<8)|Wire.read();
     int16_t gy = (Wire.read()<<8)|Wire.read();
     int16_t gz = (Wire.read()<<8)|Wire.read();
     
     Serial.print(ax/16384.0); Serial.print(",");
     Serial.print(ay/16384.0); Serial.print(",");
     Serial.print(az/16384.0); Serial.print(",");
     Serial.print(gx/131.0);   Serial.print(",");
     Serial.print(gy/131.0);   Serial.print(",");
     Serial.println(gz/131.0);
   } else {
     Serial.println("ERROR");
   }
 }