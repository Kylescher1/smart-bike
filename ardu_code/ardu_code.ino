#include <Wire.h>
#include <MPU6050.h>
#include <MadgwickAHRS.h>

MPU6050 mpu;
Madgwick filter;

// Sampling rate
unsigned long lastUpdate = 0;
float deltat = 0;

// Gyro scale: convert deg/sec to rad/sec
const float DEG2RAD = 3.14159265359 / 180.0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize MPU6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  Serial.println("MPU6050 ready.");

  // Start AHRS filter
  filter.begin(100);  // 100 Hz update frequency
  lastUpdate = micros();
}

void loop() {
  // Time delta
  unsigned long now = micros();
  deltat = (now - lastUpdate) / 1000000.0f;
  lastUpdate = now;

  // Read raw MPU data
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert to proper units
  float axg = ax / 16384.0;     // g
  float ayg = ay / 16384.0;
  float azg = az / 16384.0;

  float gxrs = (gx / 131.0) * DEG2RAD;   // rad/s
  float gyrs = (gy / 131.0) * DEG2RAD;
  float gzrs = (gz / 131.0) * DEG2RAD;

  // Update Madgwick filter
  filter.updateIMU(gxrs, gyrs, gzrs, axg, ayg, azg);

  // Get quaternion
  float qw = filter.q0;
  float qx = filter.q1;
  float qy = filter.q2;
  float qz = filter.q3;

  // Output quaternion
  Serial.print("q = ");
  Serial.print(qw, 6); Serial.print(", ");
  Serial.print(qx, 6); Serial.print(", ");
  Serial.print(qy, 6); Serial.print(", ");
  Serial.println(qz, 6);

  delay(5);  // ~200 Hz actual
}
