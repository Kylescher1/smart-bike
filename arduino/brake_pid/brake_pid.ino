#include <Arduino.h>
#include "BluetoothSerial.h"

// --- BLUETOOTH CONFIG ---
BluetoothSerial SerialBT;
const char* BT_DEVICE_NAME = "SmartBike_Brake";

// --- HARDWARE PIN DEFINITIONS ---
const int PIN_IN1 = 25; 
const int ENC_A = 33;   
const int ENC_B = 32;   

// --- ENCODER CONFIG ---
volatile long encoderPosition = 0;
long lastEncoderPosition = 0;
const int PPR = 696;  // Pulses Per Revolution (from your original code: 2786.2)

// --- PID CONFIG ---
double Kp = 0.15;  // Proportional gain (increased to get above deadband)
double Ki = 0.1;  // Integral gain

// --- PWM CONFIG ---
// Experiment with frequency: 50Hz (standard ESC), 100Hz, 200Hz, 400Hz
// Higher frequency = smoother control but ESC may not respond well
const int freq = 100;         // Try: 50, 100, 200, 400
const int resolution = 16;    // Max 16-bit for ESP32 (65535 steps)

int usToDuty(int microseconds) {
  // Period in microseconds = 1/freq * 1,000,000
  float periodUs = (1.0 / freq) * 1000000.0;
  int maxDuty = (1 << resolution) - 1;  // 2^resolution - 1
  return (int)((microseconds / periodUs) * maxDuty);
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

void sendMessage(const char* msg) {
  Serial.println(msg);
  if (SerialBT.hasClient()) {
    SerialBT.println(msg);
  }
}

void sendMessage(String msg) {
  Serial.println(msg);
  if (SerialBT.hasClient()) {
    SerialBT.println(msg);
  }
}

void armESC() {
  sendMessage("--- ARMING ESC ---");
  int neutral = usToDuty(1500);
  ledcWrite(PIN_IN1, neutral);
  delay(2000);
  sendMessage("--- ESC ARMED ---");
}

void printEncoderInfo() {
  long currentPos = encoderPosition;
  float revolutions = (float)currentPos / PPR;
  float degrees = revolutions * 360.0;
  
  String msg = "Position: " + String(currentPos) + " pulses | " + 
               String(revolutions, 3) + " revs | " + 
               String(degrees, 2) + " degrees";
  sendMessage(msg);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize Bluetooth Serial
  if (!SerialBT.begin(BT_DEVICE_NAME)) {
    Serial.println("An error occurred initializing Bluetooth");
  } else {
    Serial.println("Bluetooth initialized successfully");
    Serial.print("Device name: ");
    Serial.println(BT_DEVICE_NAME);
    Serial.println("Ready to pair. Look for 'SmartBike_Brake' on your phone.");
  }
  
  Serial.println("========================================");
  Serial.println("SmartBike Brake Control");
  Serial.println("========================================");

  
  // Setup encoder pins with pullups
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  
  // Attach interrupts for X4 resolution (CHANGE on both channels)
  attachInterrupt(digitalPinToInterrupt(ENC_A), readEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), readEncoderB, CHANGE);
  
  // Setup PWM for motor control
  ledcAttach(PIN_IN1, freq, resolution);
  
  armESC();
  
  encoderPosition = 0;
  lastEncoderPosition = 0;

  // Stop motor movements
  ledcWrite(PIN_IN1, usToDuty(1500));

  
  sendMessage("SMARTBIKE BRAKE ARMED");
  sendMessage("========================================");
  sendMessage("");
  sendMessage("Commands:");
  sendMessage("  'g' or 'G' - Start brake actuation loop");
  sendMessage("  'r' or 'R' - Reset encoder position to 0");
  sendMessage("  'p' or 'P' - Print current encoder position");
  sendMessage("");
  sendMessage("Waiting for command (Serial or Bluetooth)...");
  sendMessage("");
}

void runBrakeLoop() {

   // Reset encoder position to 0 at start of brake actuation
  noInterrupts();
  encoderPosition = 0;
  interrupts();

  sendMessage("Starting brake actuation sequence...");
  sendMessage("CSV Format: encoderPosition(P),Goal(deg),Current(deg),Error(deg),PWM(us)");
  sendMessage("---");
  
  unsigned long startTime = millis();
  unsigned long lastPlotTime = 0;
  const unsigned long plotInterval = 50; // Plot every 50ms
  
  // Phase 1: Close motor at 1800 for 3 seconds
  sendMessage("Phase 1: Closing motor at 1800 for 3 seconds...");
  unsigned long phase1Start = millis();
  unsigned long phase1Duration = 3000; // 3 seconds
  
  while (millis() - phase1Start < phase1Duration) {
    unsigned long currentTime = millis();
    
    // Read encoder position safely
    noInterrupts();
    long encoderPulses = encoderPosition;
    interrupts();
    
    // Set motor to 1800
    int pulseWidth = 1200;
    ledcWrite(PIN_IN1, usToDuty(pulseWidth));
    
    // Plot output (CSV format for serial plotter)
    if (currentTime - lastPlotTime >= plotInterval) {
      String csv = String(encoderPulses) + "," + String(0) + "," + 
                   String(0) + "," + String(0) + "," + String(pulseWidth);
      Serial.println(csv);
      if (SerialBT.hasClient()) {
        SerialBT.println(csv);
      }
      lastPlotTime = currentTime;
    }
    
    delay(10);
  }
  
  // Stop motor
  sendMessage("Stopping motor...");
  ledcWrite(PIN_IN1, usToDuty(1500));
  
  // Read encoder position for final plot point
  noInterrupts();
  long encoderPulses = encoderPosition;
  interrupts();
  
  String csv = String(encoderPulses) + "," + String(0) + "," + 
               String(0) + "," + String(0) + "," + String(1500);
  Serial.println(csv);
  if (SerialBT.hasClient()) {
    SerialBT.println(csv);
  }
  
  delay(100); // Brief pause before next phase
  
  // Phase 2: Back off motor at 1300 for 0.5 seconds
  sendMessage("Phase 2: Backing off motor at 1300 for 0.5 seconds...");
  unsigned long phase2Start = millis();
  unsigned long phase2Duration = 1000; // 0.5 seconds
  
  while (millis() - phase2Start < phase2Duration) {
    unsigned long currentTime = millis();
    
    // Read encoder position safely
    noInterrupts();
    encoderPulses = encoderPosition;
    interrupts();
    
    // Set motor to 1300
    int pulseWidth = 1250;
    ledcWrite(PIN_IN1, usToDuty(pulseWidth));
    
    // Plot output (CSV format for serial plotter)
    if (currentTime - lastPlotTime >= plotInterval) {
      String csv = String(encoderPulses) + "," + String(0) + "," + 
                   String(0) + "," + String(0) + "," + String(pulseWidth);
      Serial.println(csv);
      if (SerialBT.hasClient()) {
        SerialBT.println(csv);
      }
      lastPlotTime = currentTime;
    }
    
    delay(10);
  }
  
  // Stop motor at neutral
  ledcWrite(PIN_IN1, usToDuty(1500));
  
  // Final plot point
  noInterrupts();
  encoderPulses = encoderPosition;
  interrupts();
  
  String csv = String(encoderPulses) + "," + String(0) + "," + 
               String(0) + "," + String(0) + "," + String(1500);
  Serial.println(csv);
  if (SerialBT.hasClient()) {
    SerialBT.println(csv);
  }
  
  sendMessage("---");
  String completeMsg = "Brake sequence completed! Total time: " + 
                       String(millis() - startTime) + " ms";
  sendMessage(completeMsg);
  sendMessage("");
}

void loop() {
  // Check for serial commands (USB Serial)
  if (Serial.available() > 0) {
    char command = Serial.read();
    processCommand(command);
  }
  
  // Check for Bluetooth commands
  if (SerialBT.available() > 0) {
    char command = SerialBT.read();
    processCommand(command);
  }
  
  delay(50); // Small delay before checking for commands
}

void processCommand(char command) {
  switch (command) {
    case 'g':
    case 'G':
      // Start brake actuation loop
      runBrakeLoop();
      sendMessage("Ready for next command...");
      break;
      
    case 'r':
    case 'R':
      // Reset encoder position
      noInterrupts();
      encoderPosition = 0;
      interrupts();
      sendMessage("Encoder position reset to 0");
      printEncoderInfo();
      break;
      
    case 'p':
    case 'P':
      // Print current position
      printEncoderInfo();
      break;
      
    default:
      if (command != '\n' && command != '\r') {
        String msg = "Unknown command: " + String(command);
        sendMessage(msg);
        sendMessage("Commands: g=go, r=reset, p=print");
      }
      break;
  }
}
