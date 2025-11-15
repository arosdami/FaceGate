#include <Servo.h>

Servo lockServo;
const int SERVO_PIN = 3;  // Változtasd meg a pin számát!

void setup() {
  lockServo.attach(SERVO_PIN);
  Serial.begin(9600);
  lockServo.write(90);
  Serial.println("UNO READY - D3 SERVO");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "UNLOCK") {
      lockServo.write(0);
      Serial.println("NYITVA");
    }
    else if (cmd == "LOCK") {
      lockServo.write(90);
      Serial.println("ZARVA");
    }
  }
}