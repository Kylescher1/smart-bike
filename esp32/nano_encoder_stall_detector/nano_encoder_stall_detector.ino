/*
 * ESP32 Quadrature Encoder Stall Detector
 * 
 * Optimized for high-speed encoder reading
 * 
 * Hardware:
 * - Quadrature encoder connected to pins 25 (ENC_A) and 26 (ENC_B)
 * - Both pins use hardware interrupts
 * 
 * Serial Protocol (115200 baud):
 * - Sends: "POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>\n"
 */

// ESP32 includes
#include "soc/gpio_struct.h"
#include "soc/gpio_reg.h"
#include "soc/soc.h"

// Encoder pins
#define ENC_A 25
#define ENC_B 26

// Stall detection config
#define STALL_TIMEOUT_MS 500    // Time window to detect stall (ms)
#define STALL_THRESHOLD 5       // Minimum pulses to consider motor moving
#define STATUS_INTERVAL 20      // Status update interval (ms) - faster updates
#define VELOCITY_UPDATE_MS 50   // Velocity calculation interval (ms) - faster updates

// Pulse count from encoder (quadrature, so 4x resolution)
volatile int64_t encoderValue = 0;

// Previous encoder state for quadrature decoding
volatile uint8_t lastEncoderState = 0;

// Last movement timestamp (using micros for IRAM safety)
volatile unsigned long lastMovementMicros = 0;

// Previous encoder value for stall detection
int64_t lastEncoderValue = 0;

// Timing variables
unsigned long previousMillis = 0;

// Velocity calculation
int64_t lastPositionForVelocity = 0;
unsigned long lastVelocityTime = 0;
float currentVelocity = 0.0;  // pulses per second

// Stall detection
bool isMoving = false;
bool isStalled = false;

// Fast pin read macros (using direct GPIO register access for ESP32)
// Using GPIO struct directly for maximum speed in interrupts
inline uint8_t IRAM_ATTR fastReadA() {
  return (GPIO.in >> ENC_A) & 0x01;
}

inline uint8_t IRAM_ATTR fastReadB() {
  return (GPIO.in >> ENC_B) & 0x01;
}

// Quadrature decoder lookup table
// Based on Gray code sequence: 00 -> 01 -> 11 -> 10 -> 00 (forward)
// Index = (lastState << 2) | currentState
// Value: +1 = forward, -1 = reverse, 0 = invalid/no change
static const int8_t quadratureTable[16] = {
  0,  // 00 -> 00: no change
  +1, // 00 -> 01: forward
  -1, // 00 -> 10: reverse
  0,  // 00 -> 11: invalid
  -1, // 01 -> 00: reverse
  0,  // 01 -> 01: no change
  0,  // 01 -> 10: invalid
  +1, // 01 -> 11: forward
  +1, // 10 -> 00: forward
  0,  // 10 -> 01: invalid
  0,  // 10 -> 10: no change
  -1, // 10 -> 11: reverse
  0,  // 11 -> 00: invalid
  -1, // 11 -> 01: reverse
  +1, // 11 -> 10: forward
  0   // 11 -> 11: no change
};

// Common encoder update function using state machine
// IRAM_ATTR ensures this runs from RAM for faster execution
void IRAM_ATTR updateEncoder() {
  // Read current state of both pins using fast register access (bit0 = A, bit1 = B)
  uint8_t currentState = (fastReadA() ? 0x01 : 0x00) | (fastReadB() ? 0x02 : 0x00);
  
  // Only process if state actually changed
  if (currentState != lastEncoderState) {
    // Look up direction in quadrature table
    uint8_t transition = (lastEncoderState << 2) | currentState;
    int8_t direction = quadratureTable[transition];
    
    if (direction != 0) {
      encoderValue += direction;
      lastMovementMicros = micros();  // Use micros() which is IRAM-safe
    }
    
    lastEncoderState = currentState;
  }
}

// Encoder A interrupt handler
void IRAM_ATTR updateEncoderA() {
  updateEncoder();
}

// Encoder B interrupt handler
void IRAM_ATTR updateEncoderB() {
  updateEncoder();
}

void setup() {
  // Setup Serial Monitor at higher baud rate for faster communication
  Serial.begin(115200);
  delay(500);  // Reduced delay
  
  Serial.println("========================================");
  Serial.println("ESP32 Encoder Stall Detector");
  Serial.println("========================================");
  Serial.println();
  
  // Set encoder pins as input with internal pullup
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  
  // Read and display initial pin states
  int pinAState = digitalRead(ENC_A);
  int pinBState = digitalRead(ENC_B);
  Serial.print("Initial pin states - Pin ");
  Serial.print(ENC_A);
  Serial.print(" (ENC_A): ");
  Serial.print(pinAState);
  Serial.print(", Pin ");
  Serial.print(ENC_B);
  Serial.print(" (ENC_B): ");
  Serial.println(pinBState);
  Serial.println();
  
  // Attach interrupts for both pins (ESP32 supports interrupts on all GPIO pins)
  attachInterrupt(digitalPinToInterrupt(ENC_A), updateEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), updateEncoderB, CHANGE);
  
  // Initialize timing
  previousMillis = millis();
  lastVelocityTime = millis();
  lastMovementMicros = micros();
  lastPositionForVelocity = 0;
  lastEncoderValue = 0;
  
  // Initialize encoder state (read both pins and encode as 2-bit value)
  // State encoding: bit0 = A, bit1 = B
  lastEncoderState = (pinAState ? 0x01 : 0x00) | (pinBState ? 0x02 : 0x00);
  
  Serial.println("Status format: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>");
  Serial.println("Pin states: 0=LOW, 1=HIGH");
  Serial.print("Update rate: ");
  Serial.print(1000 / STATUS_INTERVAL);
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Ready! Monitoring encoder...");
  Serial.println();
}

void loop() {
  // Update status at regular intervals (faster updates)
  unsigned long currentMillis = millis();
  
  if (currentMillis - previousMillis >= STATUS_INTERVAL) {
    previousMillis = currentMillis;
    
    // Read current encoder value safely (disable interrupts briefly)
    int64_t currentEncoderValue;
    noInterrupts();
    currentEncoderValue = encoderValue;
    interrupts();
    
    // Calculate velocity (faster updates)
    unsigned long timeDelta = currentMillis - lastVelocityTime;
    if (timeDelta >= VELOCITY_UPDATE_MS) {
      int64_t positionDelta = currentEncoderValue - lastPositionForVelocity;
      currentVelocity = (float)positionDelta / (timeDelta / 1000.0);  // pulses per second
      lastPositionForVelocity = currentEncoderValue;
      lastVelocityTime = currentMillis;
    }
    
    // Check for stall
    unsigned long lastMoveMillis;
    noInterrupts();
    lastMoveMillis = lastMovementMicros / 1000;  // Convert to milliseconds
    interrupts();
    
    unsigned long timeSinceLastMove = currentMillis - lastMoveMillis;
    int64_t positionDelta = abs(currentEncoderValue - lastEncoderValue);
    
    if (positionDelta >= STALL_THRESHOLD) {
      // Motor has moved
      isMoving = true;
      isStalled = false;
      lastEncoderValue = currentEncoderValue;
    } else if (timeSinceLastMove > STALL_TIMEOUT_MS) {
      // No movement detected within timeout period
      isMoving = false;
      isStalled = true;
    } else {
      // Within timeout, but not enough movement yet
      isMoving = (positionDelta > 0);
      isStalled = false;
    }
    
    // Read raw pin states
    int pinAState = digitalRead(ENC_A);
    int pinBState = digitalRead(ENC_B);
    
    // Send status: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>
    Serial.print("POS,");
    Serial.print(currentEncoderValue);
    Serial.print(",");
    Serial.print(currentVelocity, 2);
    Serial.print(",");
    Serial.print(isMoving ? 1 : 0);
    Serial.print(",");
    Serial.print(isStalled ? 1 : 0);
    Serial.print(",");
    Serial.print(pinAState);
    Serial.print(",");
    Serial.println(pinBState);
  }
  
  // Small delay to prevent watchdog issues (ESP32 specific)
  delay(1);
}
