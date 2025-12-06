#include <Arduino.h>

// --- HARDWARE PIN DEFINITIONS ---
const int PIN_IN1 = 25; 
const int PIN_IN2 = 26; 

// --- PWM CONFIG ---
// Note: Duty cycle is NOT 100% max! At 50Hz (20ms period):
//   - 1500us = 7.5% duty cycle (neutral)
//   - 2000us = 10% duty cycle (typical RC ESC max - convention, not physical limit)
// The ESC interprets pulse width (microseconds), not duty cycle percentage
// RC standard: 1000-2000us range (some ESCs may accept slightly beyond)
const int freq = 50;         // 50Hz = 20ms period
const int resolution = 16;   // 16-bit resolution

int usToDuty(int microseconds) {
  return (int)((microseconds / 20000.0) * 65535.0);
}

void armESC() {
  Serial.println("--- ARMING ESC ---");
  int neutral = usToDuty(1500);
  ledcWrite(PIN_IN1, neutral);
  ledcWrite(PIN_IN2, neutral);
  delay(2000);
  Serial.println("--- ESC ARMED ---");
  Serial.println();
}

void setChannel(int channel, int microseconds, const char* channelName) {
  int duty = usToDuty(microseconds);
  ledcWrite(channel, duty);
  Serial.print(channelName);
  Serial.print(" set to ");
  Serial.print(microseconds);
  Serial.print(" us (duty: ");
  Serial.print(duty);
  Serial.println(")");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("========================================");
  Serial.println("ESC Channel Test Script");
  Serial.println("========================================");
  Serial.println();
  
  ledcAttach(PIN_IN1, freq, resolution);
  ledcAttach(PIN_IN2, freq, resolution);
  
  armESC();
  
  Serial.println("Starting test sequence...");
  Serial.println("Each step will hold for 3 seconds");
  Serial.println();
  delay(2000);
}

void loop() {
  // Test Channel 1 (PIN_IN1) - Full Range Test
  Serial.println(">>> TESTING CHANNEL 1 (PIN_IN1) - FULL RANGE <<<");
  Serial.println();
  
  // Keep CH2 at neutral throughout
  setChannel(PIN_IN2, 1500, "CH2");
  
  // Neutral
  Serial.println("Step 1: NEUTRAL (1500us)");
  setChannel(PIN_IN1, 1500, "CH1");
  delay(3000);
  
  // Forward progression - starting above deadband (~1550us makes noise but no movement)
  Serial.println("Step 2: MIN FORWARD (1600us) - Above deadband");
  setChannel(PIN_IN1, 1600, "CH1");
  delay(3000);
  
  Serial.println("Step 3: SLOW FORWARD (1650us)");
  setChannel(PIN_IN1, 1650, "CH1");
  delay(3000);
  
  Serial.println("Step 4: MEDIUM FORWARD (1700us)");
  setChannel(PIN_IN1, 1700, "CH1");
  delay(3000);
  
  Serial.println("Step 5: MEDIUM-FAST FORWARD (1750us)");
  setChannel(PIN_IN1, 1750, "CH1");
  delay(3000);
  
  Serial.println("Step 6: FAST FORWARD (1800us)");
  setChannel(PIN_IN1, 1800, "CH1");
  delay(3000);
  
  Serial.println("Step 7: MAX FORWARD (1850us)");
  setChannel(PIN_IN1, 1850, "CH1");
  delay(3000);
  
  // Back to neutral
  Serial.println("Step 8: NEUTRAL (1500us)");
  setChannel(PIN_IN1, 1500, "CH1");
  delay(3000);
  
  // Reverse progression - starting above deadband
  Serial.println("Step 9: MIN REVERSE (1400us) - Above deadband");
  setChannel(PIN_IN1, 1400, "CH1");
  delay(3000);
  
  Serial.println("Step 10: SLOW REVERSE (1350us)");
  setChannel(PIN_IN1, 1350, "CH1");
  delay(3000);
  
  Serial.println("Step 11: MEDIUM REVERSE (1300us)");
  setChannel(PIN_IN1, 1300, "CH1");
  delay(3000);
  
  Serial.println("Step 12: MEDIUM-FAST REVERSE (1250us)");
  setChannel(PIN_IN1, 1250, "CH1");
  delay(3000);
  
  Serial.println("Step 13: FAST REVERSE (1200us)");
  setChannel(PIN_IN1, 1200, "CH1");
  delay(3000);
  
  Serial.println("Step 14: MAX REVERSE (1150us)");
  setChannel(PIN_IN1, 1150, "CH1");
  delay(3000);
  
  // Final neutral
  Serial.println("Step 15: NEUTRAL (1500us) - STOP");
  setChannel(PIN_IN1, 1500, "CH1");
  delay(5000);
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("Test cycle complete. Restarting...");
  Serial.println("========================================");
  Serial.println();
  delay(3000);
}
