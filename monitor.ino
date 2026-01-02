#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#define LED_PIN 4
#define SENSOR_POWER_PIN_1 5
#define SENSOR_PIN_1 34

#define SENSOR_POWER_PIN_2 18
#define SENSOR_PIN_2 35


#define BLE_ADVERTISING_TIME 45000
#define SLEEP_TIME 60e6

#define SERVICE_UUID        "12345678-1234-1234-1234-1234567890ab"
#define CHARACTERISTIC_UUID "abcdefab-1234-5678-1234-abcdefabcdef"

BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;

bool deviceConnected = false;
unsigned long bleStartTime;
int sensorValue1 = 0;
int sensorValue2 = 0;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) {
    deviceConnected = true;
    Serial.println("Client connected");
  }
  void onDisconnect(BLEServer*) {
    deviceConnected = false;
    Serial.println("Client disconnected");
  }
};

class MyCharacteristicCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pChar) {
    Serial.println("Client wrote request");
    pChar->setValue(sensorValue1);
    uint16_t payload[2];
    payload[0] = sensorValue1;
    payload[1] = sensorValue2;
    pChar->setValue((uint8_t*)payload, sizeof(payload));
    pChar->notify();
  }
};

int readSensor(int sensor_pin) {
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(sensor_pin);
    delay(10);
  }
  return sum / 10;
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  pinMode(SENSOR_POWER_PIN_1, OUTPUT);
  pinMode(SENSOR_POWER_PIN_2, OUTPUT);

  digitalWrite(LED_PIN, HIGH);
  digitalWrite(SENSOR_POWER_PIN_1, HIGH);
  digitalWrite(SENSOR_POWER_PIN_2, HIGH);
  delay(200);

  sensorValue1 = readSensor(SENSOR_PIN_1);
  sensorValue2 = readSensor(SENSOR_PIN_2);

  Serial.print("Sensor value 1: ");
  Serial.println(sensorValue1);
  Serial.print("Sensor value 2: ");
  Serial.println(sensorValue2);

  BLEDevice::init("ESP32_Sensor");

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_NOTIFY
  );

  BLE2902* desc = new BLE2902();
  desc->setNotifications(true);
  pCharacteristic->addDescriptor(desc);
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
  pCharacteristic->setValue(sensorValue1);

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  bleStartTime = millis();
  Serial.println("BLE advertising started");
}

void loop() {
  if (millis() - bleStartTime > BLE_ADVERTISING_TIME) {
    Serial.println("BLE window expired, sleeping");
    digitalWrite(LED_PIN, LOW);
    digitalWrite(SENSOR_POWER_PIN_1, LOW);
    digitalWrite(SENSOR_POWER_PIN_2, LOW);
    esp_deep_sleep(SLEEP_TIME);
  }
  delay(100);
}
