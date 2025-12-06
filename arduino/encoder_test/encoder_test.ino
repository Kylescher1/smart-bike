#include <Arduino.h>

// --- HARDWARE PIN DEFINITIONS ---
const int PIN_IN1 = 25; 
const int PIN_IN2 = 26; 
const int ENC_A = 33;   
const int ENC_B = 32;   

// --- ENCODER CONFIG ---
volatile long encoderPosition = 0;
long lastEncoderPosition = 0;
const int PPR = 2786;  // Pulses Per Revolution (from your original code: 2786.2)

// --- PWM CONFIG ---
const int freq = 50;         
const int resolution = 16;   

int usToDuty(int microseconds) {
  return (int)((microseconds / 20000.0) * 65535.0);
}

// --- ENCODER INTERRUPT HANDLERS ---
// X4 resolution: triggers on both rising and falling edges of both channels
void IRAM_ATTR readEncoderA() {
  // If A and B are different, we are moving forward.
  // If A and B are the same, we are moving backward.
  if (digitalRead(ENC_A) != digitalRead(ENC_B)) {
    encoderPosition++;
  } else {
    encoderPosition--;
  }
}

void IRAM_ATTR readEncoderB() {
  // Logic is inverted for Pin B
  if (digitalRead(ENC_A) == digitalRead(ENC_B)) {
    encoderPosition++;
  } else {
    encoderPosition--;
  }
}

void armESC() {
  Serial.println("--- ARMING ESC ---");
  int neutral = usToDuty(1500);
  ledcWrite(PIN_IN1, neutral);
  ledcWrite(PIN_IN2, neutral);
  delay(2000);
  Serial.println("--- ESC ARMED ---");
}

void printEncoderInfo() {
  long currentPos = encoderPosition;
  float revolutions = (float)currentPos / PPR;
  float degrees = revolutions * 360.0;
  
  Serial.print("Position: ");
  Serial.print(currentPos);
  Serial.print(" pulses | ");
  Serial.print(revolutions, 3);
  Serial.print(" revs | ");
  Serial.print(degrees, 2);
  Serial.println(" degrees");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("========================================");
  Serial.println("Encoder Test Script");
  Serial.println("========================================");
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  'r' - Reset encoder position to 0");
  Serial.println("  'p' - Print current encoder position");
  Serial.println("  'g' - Go: Rotate motor forward until PPR reached");
  Serial.println("  's' - Stop motor");
  Serial.println();
  Serial.println("PPR (Pulses Per Revolution): ");
  Serial.println(PPR);
  Serial.println();
  
  // Setup encoder pins with pullups
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  
  // Attach interrupts for X4 resolution (CHANGE on both channels)
  attachInterrupt(digitalPinToInterrupt(ENC_A), readEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), readEncoderB, CHANGE);
  
  // Setup PWM for motor control
  ledcAttach(PIN_IN1, freq, resolution);
  ledcAttach(PIN_IN2, freq, resolution);
  
  armESC();
  
  encoderPosition = 0;
  lastEncoderPosition = 0;
  
  Serial.println("Ready! Rotate encoder by hand or send commands.");
  Serial.println();
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case 'r':
      case 'R':
        // Reset encoder position
        noInterrupts();
        encoderPosition = 0;
        interrupts();
        Serial.println("Encoder position reset to 0");
        printEncoderInfo();
        break;
        
      case 'p':
      case 'P':
        // Print current position
        printEncoderInfo();
        break;
        
      case 'g':
      case 'G': {
        // Go: Rotate motor forward until PPR reached
        Serial.println("Starting rotation to reach PPR...");
        noInterrupts();
        long startPosition = encoderPosition;
        interrupts();
        
        // Start motor forward (above deadband)
        int pulseWidth = 1600; // Slow forward
        ledcWrite(PIN_IN1, usToDuty(pulseWidth));
        ledcWrite(PIN_IN2, usToDuty(1500)); // Keep CH2 neutral
        
        unsigned long startTime = millis();
        bool targetReached = false;
        
        while (!targetReached) {
          noInterrupts();
          long currentPos = encoderPosition;
          interrupts();
          
          long pulsesFromStart = abs(currentPos - startPosition);
          
          // Print progress every 100 pulses
          if (pulsesFromStart % 100 == 0 && pulsesFromStart > 0) {
            float progress = (float)pulsesFromStart / PPR * 100.0;
            Serial.print("Progress: ");
            Serial.print(pulsesFromStart);
            Serial.print("/");
            Serial.print(PPR);
            Serial.print(" pulses (");
            Serial.print(progress, 1);
            Serial.println("%)");
          }
          
          // Check if we've reached PPR
          if (pulsesFromStart >= PPR) {
            targetReached = true;
            // Stop motor
            ledcWrite(PIN_IN1, usToDuty(1500));
            ledcWrite(PIN_IN2, usToDuty(1500));
            
            unsigned long elapsed = millis() - startTime;
            // RPM calculation: elapsed ms for 1 revolution (PPR pulses)
            // Convert to seconds: elapsed / 1000.0
            // Revolutions per second: 1.0 / (elapsed / 1000.0) = 1000.0 / elapsed
            // RPM: (1000.0 / elapsed) * 60.0 = 60000.0 / elapsed
            float rpm = 60000.0 / elapsed;
            
            Serial.println();
            Serial.println("Target reached!");
            printEncoderInfo();
            Serial.print("Time elapsed: ");
            Serial.print(elapsed);
            Serial.println(" ms");
            Serial.print("Approximate RPM: ");
            Serial.println(rpm, 2);
          }
          
          // Safety timeout (10 seconds)
          if (millis() - startTime > 10000) {
            ledcWrite(PIN_IN1, usToDuty(1500));
            ledcWrite(PIN_IN2, usToDuty(1500));
            Serial.println("Timeout! Motor stopped.");
            break;
          }
          
          delay(10); // Small delay to prevent overwhelming serial output
        }
        break;
      }
        
      case 's':
      case 'S':
        // Stop motor
        ledcWrite(PIN_IN1, usToDuty(1500));
        ledcWrite(PIN_IN2, usToDuty(1500));
        Serial.println("Motor stopped");
        printEncoderInfo();
        break;
        

  
  // Continuously print encoder position if it changes (for manual rotation testing)
  noInterrupts();
  long currentPos = encoderPosition;
  interrupts();
  
  if (currentPos != lastEncoderPosition) {
    printEncoderInfo();
    lastEncoderPosition = currentPos;
  }
  
  delay(50); // Small delay to prevent overwhelming serial output
}
