// I2C device class (I2Cdev) demonstration Arduino sketch for MPU6050 class using DMP (MotionApps v2.0)
// 6/21/2012 by Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
// Changelog:
//      2019-07-08 - Added Auto Calibration and offset generator
//       - and altered FIFO retrieval sequence to avoid using blocking code
//      2016-04-18 - Eliminated a potential infinite loop
//      2013-05-08 - added seamless Fastwire support
//                 - added note about gyro calibration
//      2012-06-21 - added note about Arduino 1.0.1 + Leonardo compatibility error
//      2012-06-20 - improved FIFO overflow handling and simplified read process
//      2012-06-19 - completely rearranged DMP initialization code and simplification
//      2012-06-13 - pull gyro and accel data from FIFO packet instead of reading directly
//      2012-06-09 - fix broken FIFO read sequence and change interrupt detection to RISING
//      2012-06-05 - add gravity-compensated initial reference frame acceleration output
//                 - add 3D math helper file to DMP6 example sketch
//                 - add Euler output and Yaw/Pitch/Roll output formats
//      2012-06-04 - remove accel offset clearing for better results (thanks Sungon Lee)
//      2012-06-01 - fixed gyro sensitivity to be 2000 deg/sec instead of 250
//      2012-05-30 - basic DMP initialization working

/* ============================================
I2Cdev device library code is placed under the MIT license
Copyright (c) 2012 Jeff Rowberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
*/

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
#include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high

/* =========================================================================
   NOTE: In addition to connection 3.3v, GND, SDA, and SCL, this sketch
   depends on the MPU-6050's INT pin being connected to the Arduino's
   external interrupt #0 pin. On the Arduino Uno and Mega 2560, this is
   digital I/O pin 2.
 * ========================================================================= */

/* =========================================================================
   NOTE: Arduino v1.0.1 with the Leonardo board generates a compile error
   when using Serial.write(buf, len). The Teapot output uses this method.
   The solution requires a modification to the Arduino USBAPI.h file, which
   is fortunately simple, but annoying. This will be fixed in the next IDE
   release. For more info, see these links:

   http://arduino.cc/forum/index.php/topic,109987.0.html
   http://code.google.com/p/arduino/issues/detail?id=958
 * ========================================================================= */

///////////////////////////////////////////////////////////////new tw, replaces ypr and realaccel
#define OUTPUT_READABLE_YPR_REALACCEL_GYRO

// uncomment "OUTPUT_READABLE_QUATERNION" if you want to see the actual
// quaternion components in a [w, x, y, z] format (not best for parsing
// on a remote host such as Processing or something though)
//#define OUTPUT_READABLE_QUATERNION

// uncomment "OUTPUT_READABLE_EULER" if you want to see Euler angles
// (in degrees) calculated from the quaternions coming from the FIFO.
// Note that Euler angles suffer from gimbal lock (for more info, see
// http://en.wikipedia.org/wiki/Gimbal_lock)
//#define OUTPUT_READABLE_EULER

// uncomment "OUTPUT_READABLE_YAWPITCHROLL" if you want to see the yaw/
// pitch/roll angles (in degrees) calculated from the quaternions coming
// from the FIFO. Note this also requires gravity vector calculations.
// Also note that yaw/pitch/roll angles suffer from gimbal lock (for
// more info, see: http://en.wikipedia.org/wiki/Gimbal_lock)
///#define OUTPUT_READABLE_YAWPITCHROLL /////////////////////////////////////////////////////////////new tw

// uncomment "OUTPUT_READABLE_REALACCEL" if you want to see acceleration
// components with gravity removed. This acceleration reference frame is
// not compensated for orientation, so +X is always +X according to the
// sensor, just without the effects of gravity. If you want acceleration
// compensated for orientation, us OUTPUT_READABLE_WORLDACCEL instead.
//#define OUTPUT_READABLE_REALACCEL //new tw

// uncomment "OUTPUT_READABLE_WORLDACCEL" if you want to see acceleration
// components with gravity removed and adjusted for the world frame of
// reference (yaw is relative to initial orientation, since no magnetometer
// is present in this case). Could be quite handy in some cases.
//#define OUTPUT_READABLE_WORLDACCEL

// uncomment "OUTPUT_TEAPOT" if you want output that matches the
// format used for the InvenSense teapot demo
//#define OUTPUT_TEAPOT /////////////////////////////////////////////////////////////new tw

#define INTERRUPT_PIN 2 // use pin 2 on Arduino Uno & most boards
#define LED_PIN 13      // (Arduino is 13, Teensy is 11, Teensy++ is 6)
bool blinkState = false;
long count = 0;

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;        // [w, x, y, z]         quaternion container
VectorInt16 aa;      // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;  // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld; // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity; // [x, y, z]            gravity vector
float euler[3];      // [psi, theta, phi]    Euler angle container
float ypr[3];        // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

//new tw
VectorInt16 gyro; // [x, y, z]            gyro sensor measurements

// packet structure for InvenSense teapot demo
uint8_t teapotPacket[14] = {'$', 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, '\r', '\n'};

//new tw
float min_yaw = 9999999;
float max_yaw = -9999999;
float min_pitch = 9999999;
float max_pitch = -9999999;
float min_roll = 9999999;
float max_roll = -9999999;
int16_t min_aareal_x = 32767;
int16_t max_aareal_x = -32767;
int16_t min_aareal_y = 32767;
int16_t max_aareal_y = -32767;
int16_t min_aareal_z = 32767;
int16_t max_aareal_z = -32767;

//new tw
enum BeetleStates
{
  IDLING,
  MOVING
};                            //two states
BeetleStates bState = IDLING; //Start state is Idling

bool hello = false;
int count_moving = 0;
int count_idling = 0;
float total_acceleration = 0;
float abs_yaw = 0;
float abs_pitch = 0;
float abs_roll = 0;

int mode = 1; //0 = normal output, 1 = raw data output

//new tw
const int analogInPin = A0; // Analog input pin that the potentiometer is attached to
const int analogOutPin = 9; // Analog output pin that the LED is attached to

int sensorValue = 0; // value read from the pot

// riyas params
bool handshakeBool = false;
char msg;
volatile int isReadyToSendData = false;
//int isReadyToGetData = false;
char transmit_buffer[70];
char timestamp_arr[20];
unsigned long timestamp = 0;
char yaw[2];
char pitch[2];
char roll[2];
char accx[2];
char accy[2];
char accz[2];
char gyrx[2];
char gyry[2];
char gyrz[2];

struct DataPacket
{
  int type;
  int yaw;
  int pitch;
  int roll;
  int accx;
  int accy;
  int accz;
  int checksum;
  int padding1;
  int padding2;
};

struct TimePacket
{
  int type;
  int padding1;
  long rec;
  long sent;
  int padding2;
  int padding3;
  int padding4;
  int padding5;
};

// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false; // indicates whether MPU interrupt pin has gone high
void dmpDataReady()
{
  mpuInterrupt = true;
}

//new tw
// ================================================================
// ===                      STATE MACHINE                       ===
// ================================================================
int readValues()
{
  if (total_acceleration > 0.23 && ((abs_pitch > 10) || (abs_roll > 10)))
  {           //(abs_yaw > 10) excluding yaw for now, unable to fix the drift
    return 1; //MOVING at this moment
  }
  return 0; //IDLING at this moment
}

bool isMoving()
{ //current state is idling, check if it is moving
  if (readValues() == 1)
  { //read MOVING
    count_moving++;
    if (count_moving >= 5)
    {
      return true; //Finally, idle->move
    }
  }
  else
  { //read IDLING, moving is false alarm
    count_moving = 0;
  }

  return false; //return false
}

bool isIdling()
{ //current state is moving, check if it is idling
  if (readValues() == 0)
  { //read IDLING
    count_idling++;
    if (count_idling >= 5)
    {
      return true; //Finally, move->idle
    }
  }
  else
  { //read MOVING, idling is false alarm
    count_idling = 0;
  }
  return false; //return false
}

void checkBeetleState()
{
  switch (bState)
  { //Depending on the state
  case IDLING:
  {
    if (isMoving()) //Check if it is moving
      bState = MOVING;
    break; //Get out of switch
  }
  case MOVING:
  {
    if (isIdling()) //Check if it is idling
      bState = IDLING;
    break; //Get out of switch
  }
  }
}

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup()
{
// join I2C bus (I2Cdev library doesn't do this automatically)
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif

  // initialize serial communication
  // (115200 chosen because it is required for Teapot Demo output, but it's
  // really up to you depending on your project)
  Serial.begin(115200);
  while (!Serial)
    ; // wait for Leonardo enumeration, others continue immediately

  // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3V or Arduino
  // Pro Mini running at 3.3V, cannot handle this baud rate reliably due to
  // the baud timing being too misaligned with processor ticks. You must use
  // 38400 or slower in these cases, or use some kind of external separate
  // crystal solution for the UART timer.

  // initialize device

  Serial.println(F("Initializing I2C devices..."));
  mpu.initialize();
  pinMode(INTERRUPT_PIN, INPUT);

  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));

  // wait for ready
  Serial.println(F("\nSend any character to begin DMP programming and demo: "));

  unsigned long tmp_recv_timestamp = 0;
  unsigned long tmp_send_timestamp = 0;
  while (!handshakeBool)
  {
    if (Serial.available())
    {
      msg = Serial.read();
      msg = (char)msg;
      if (msg == 'H')
      {
        TimePacket packet;
        packet.type = 2;
        packet.padding1 = 0;
        tmp_recv_timestamp = millis();
        packet.rec = tmp_recv_timestamp;
        tmp_send_timestamp = millis();
        packet.sent = tmp_send_timestamp;
        packet.padding2 = 0;
        packet.padding3 = 0;
        packet.padding4 = 0;
        packet.padding5 = 0;

        //char buffer1[20];
        //memcpy(buffer1, packet, 20);

        Serial.write((byte *)&packet, sizeof(packet));

        handshakeBool = true;
      }
    }
  }

  // while (Serial.available() && Serial.read()); // empty buffer
  // while (!Serial.available());                 // wait for data
  // while (Serial.available() && Serial.read()); // empty buffer again

  // load and configure the DMP
  // Serial.writeln(F("Initializing DMP..."));
  devStatus = mpu.dmpInitialize();

  // supply your own gyro offsets here, scaled for min sensitivity
  //Your offsets:  -1138 1330  2042  78  -7  -33

  //90 tilted
  //-2435  -3894 3480  51  -18 -9

  mpu.setXAccelOffset(-1138);
  mpu.setYAccelOffset(-1330);
  mpu.setZAccelOffset(2402); // 1688 factory default for my test chip
  //Gyro
  mpu.setXGyroOffset(78);  //220
  mpu.setYGyroOffset(-7);  //76
  mpu.setZGyroOffset(-33); //-85

  // make sure it worked (returns 0 if so)
  if (devStatus == 0)
  {
    // Calibration Time: generate offsets and calibrate our MPU6050
    mpu.CalibrateAccel(6); //new tw stay on
    mpu.CalibrateGyro(6);  //new tw
                           //    mpu.PrintActiveOffsets();
    // turn on the DMP, now that it's ready
    // Serial.writeln(F("Enabling DMP..."));
    mpu.setDMPEnabled(true);

    // enable Arduino interrupt detection
    //Serial.write(F("Enabling interrupt detection (Arduino external interrupt "));
    //Serial.write(digitalPinToInterrupt(INTERRUPT_PIN));
    //Serial.writeln(F(")..."));
    //attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();

    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    //Serial.writeln(F("DMP ready! Waiting for first interrupt..."));
    dmpReady = true;

    // get expected DMP packet size for later comparison
    packetSize = mpu.dmpGetFIFOPacketSize();
  }
  else
  {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    //Serial.write(F("DMP Initialization failed (code "));
    //Serial.write(devStatus);
    //Serial.writeln(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);
}

// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================
void loop()
{

  // if programming failed, don't try to do anything
  if (!dmpReady)
    return;

  // wait for MPU interrupt or extra packet(s) available

  //Tw code
  if (fifoCount < packetSize)
  {
    //no more interrupt
    fifoCount = mpu.getFIFOCount();
  }

  // reset interrupt flag and get INT_STATUS byte
  mpuInterrupt = false;
  mpuIntStatus = mpu.getIntStatus();

  // get current FIFO count
  fifoCount = mpu.getFIFOCount();

  if (fifoCount < packetSize)
  {
    //Lets go back and wait for another interrupt. We shouldn't be here, we got an interrupt from another event
    // This is blocking so don't do it   while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();
  }
  // check for overflow (this should never happen unless our code is too inefficient)

  else if (hello)
  { //new tw
    if (Serial.available() > 0)
    {
      char val = Serial.read();
      if (val == 'c')
      {
        //Serial.writeln("Go");
        hello = !hello;
      }
    }
  }
  else if (!hello)
  { //new tw

    //delay(1000);
    if (Serial.available() > 0)
    {
      char val = Serial.read();
      //Serial.writeln("HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
      if (val == 'c')
      {
        //Serial.writeln("Stop");
        hello = !hello;
        return;
      }
    }
    // read a packet from FIFO
    while (fifoCount >= packetSize)
    { // Lets catch up to NOW, someone is using the dreaded delay()!
      mpu.getFIFOBytes(fifoBuffer, packetSize);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount -= packetSize;
    }

#ifdef OUTPUT_READABLE_QUATERNION
    // display quaternion values in easy matrix form: w x y z
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    //Serial.write("quat\t");
    //Serial.write(q.w);
    //Serial.write("\t");
    //Serial.write(q.x);
    //Serial.write("\t");
    //Serial.write(q.y);
    //Serial.write("\t");
    //Serial.writeln(q.z);
#endif

#ifdef OUTPUT_READABLE_EULER
    // display Euler angles in degrees
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetEuler(euler, &q);
    //Serial.write("euler\t");
    //Serial.write(euler[0] * 180/M_PI);
    //Serial.write("\t");
    //Serial.write(euler[1] * 180/M_PI);
    //Serial.write("\t");
    //Serial.writeln(euler[2] * 180/M_PI);
#endif

//new tw
#ifdef OUTPUT_READABLE_YPR_REALACCEL_GYRO

    // Serial.writeln("//////////////////////////");
    if (mode == 0)
    {
      // display Euler angles in degrees
      mpu.dmpGetQuaternion(&q, fifoBuffer);
      mpu.dmpGetGravity(&gravity, &q);
      mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
      //Serial.writeln("Edited!!");
      //Serial.write("ypr\t");
      //Serial.write(ypr[0] * 180/M_PI);
      //Serial.write("\t");
      //Serial.write(ypr[1] * 180/M_PI);
      //Serial.write("\t");
      //Serial.write(ypr[2] * 180/M_PI);
      //Serial.write("\t");

      mpu.dmpGetAccel(&aa, fifoBuffer);
      mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
      //Serial.write("areal\t");
      //Serial.write(aaReal.x);
      //Serial.write("\t");
      //Serial.write(aaReal.y);
      //Serial.write("\t");
      //Serial.writeln(aaReal.z);

      mpu.dmpGetGyro(&gyro, fifoBuffer);
      //Serial.write("gyro\t");
      //Serial.write(gyro.x);
      //Serial.write("\t");
      //Serial.write(gyro.y);
      //Serial.write("\t");
      //Serial.write(gyro.z);
      //Serial.write("\t");

      total_acceleration = sqrt(pow(aaReal.x / 8192.0, 2) + pow(aaReal.y / 8192.0, 2) + pow(aaReal.z / 8192.0, 2));
      //Serial.write("total accel\t");
      //Serial.writeln(total_acceleration);

      abs_yaw = abs(ypr[0] * 180 / M_PI);
      abs_pitch = abs(ypr[1] * 180 / M_PI);
      abs_roll = abs(ypr[2] * 180 / M_PI);
      //Serial.write("Abs ypr\t");
      //Serial.write(abs_yaw);
      //Serial.write("\t");
      //Serial.write(abs_pitch);
      //Serial.write("\t");
      //Serial.write(abs_roll);
      //Serial.write("\t");

      checkBeetleState();

      /*if (total_acceleration > 0.23 && ((abs_yaw > 10) || (abs_pitch > 10) || (abs_roll > 10))) {
               Serial.write("===========================MOVING\t");
            } else {
                Serial.write("########################IDLE\t");
            }*/

      if (bState == MOVING)
      {
        //Serial.writeln("State: MOVING");
      }
      else
      {
        //Serial.writeln("State: IDLING");
      }

      // read the analog in value:
      sensorValue = analogRead(analogInPin);
      float voltage = sensorValue * (5.0 / 1023.0);
      // print out the value you read:
      //Serial.write("Voltage = ");
      //Serial.writeln(voltage);
    }
    else
    {

      // display Euler angles in degrees
      mpu.dmpGetQuaternion(&q, fifoBuffer);
      mpu.dmpGetGravity(&gravity, &q);
      mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
      //Serial.println("Edited!!");
      //Serial.print("|ypr|");
      //Serial.print(ypr[0] * 180/M_PI);
      //Serial.print("|");
      //Serial.print(ypr[1] * 180/M_PI);
      //Serial.print("|");
      //Serial.print(ypr[2] * 180/M_PI);
      //Serial.println("|");

      mpu.dmpGetAccel(&aa, fifoBuffer);
      mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
      //Serial.print("|areal|");
      //Serial.print(aaReal.x);
      //Serial.print("|");
      //Serial.print(aaReal.y);
      //Serial.print("|");
      //Serial.print(aaReal.z);
      //Serial.println("|");

      mpu.dmpGetGyro(&gyro, fifoBuffer);
      //Serial.write("|gyro|");
      //Serial.write(gyro.x);
      //Serial.write("|");
      //Serial.write(gyro.y);
      //Serial.write("|");
      //Serial.write(gyro.z);
      //Serial.writeln("|");

      total_acceleration = sqrt(pow(aaReal.x / 8192.0, 2) + pow(aaReal.y / 8192.0, 2) + pow(aaReal.z / 8192.0, 2));
      //Serial.write("|total accel|");
      //Serial.write(total_acceleration);
      //Serial.writeln("|");

      abs_yaw = abs(ypr[0] * 180 / M_PI);
      abs_pitch = abs(ypr[1] * 180 / M_PI);
      abs_roll = abs(ypr[2] * 180 / M_PI);
      //Serial.write("|Abs ypr|");
      //Serial.write(abs_yaw);
      //Serial.write("|");
      //Serial.write(abs_pitch);
      //Serial.write("|");
      //Serial.write(abs_roll);
      //Serial.writeln("|");

      checkBeetleState();

      /*if (total_acceleration > 0.23 && ((abs_yaw > 10) || (abs_pitch > 10) || (abs_roll > 10))) {
               Serial.write("===========================MOVING\t");
            } else {
                Serial.write("########################IDLE\t");
            }*/

      if (bState == MOVING)
      {
        //Serial.writeln("|MOVING|");
      }
      else
      {
        //Serial.writeln("|IDLING|");
      }
    }
    //Serial.writeln("//////////////////////////");
    delay(27);

#endif

#ifdef OUTPUT_READABLE_YAWPITCHROLL
    // display Euler angles in degrees
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    //Serial.write("ypr\t");
    //Serial.write(ypr[0] * 180/M_PI);
    //Serial.write("\t");
    //Serial.write(ypr[1] * 180/M_PI);
    //Serial.write("\t");
    //Serial.writeln(ypr[2] * 180/M_PI);
#endif

#ifdef OUTPUT_READABLE_REALACCEL
    // display real acceleration, adjusted to remove gravity
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
    //Serial.write("areal\t");
    //Serial.write(aaReal.x);
    //Serial.write("\t");
    //Serial.write(aaReal.y);
    //Serial.write("\t");
    //Serial.writeln(aaReal.z);
#endif

#ifdef OUTPUT_READABLE_WORLDACCEL
    // display initial world-frame acceleration, adjusted to remove gravity
    // and rotated based on known orientation from quaternion
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
    mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);
    //Serial.write("aworld\t");
    //Serial.write(aaWorld.x);
    //Serial.write("\t");
    //Serial.write(aaWorld.y);
    //Serial.write("\t");
    //Serial.writeln(aaWorld.z);
#endif

#ifdef OUTPUT_TEAPOT
    // display quaternion values in InvenSense Teapot demo format:
    teapotPacket[2] = fifoBuffer[0];
    teapotPacket[3] = fifoBuffer[1];
    teapotPacket[4] = fifoBuffer[4];
    teapotPacket[5] = fifoBuffer[5];
    teapotPacket[6] = fifoBuffer[8];
    teapotPacket[7] = fifoBuffer[9];
    teapotPacket[8] = fifoBuffer[12];
    teapotPacket[9] = fifoBuffer[13];
    //Serial.write(teapotPacket, 14);
    teapotPacket[11]++; // packetCount, loops at 0xFF on purpose
#endif

    // blink LED to indicate activity
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);

    isReadyToSendData = true;
  }

  sendData();
}

int getChecksum(DataPacket packet)
{
  return packet.type ^ packet.yaw ^ packet.pitch ^ packet.roll ^
         packet.accx ^ packet.accy ^ packet.accz ^ packet.padding1 ^ packet.padding2;
}

void sendData()
{

  unsigned long tmp_recv_timestamp = 0;
  unsigned long tmp_send_timestamp = 0;
  if (Serial.available())
  { // Handshake
    msg = Serial.read();
    msg = (char)msg;
    if (msg == 'H')
    {
      TimePacket packet;
      packet.type = 2;
      packet.padding1 = 0;
      tmp_recv_timestamp = millis();
      packet.rec = tmp_recv_timestamp;
      tmp_send_timestamp = millis();
      packet.sent = tmp_send_timestamp;
      packet.padding2 = 0;
      packet.padding3 = 0;
      packet.padding4 = 0;
      packet.padding5 = 0;

      Serial.write((uint8_t *)&packet, sizeof(packet));
    }
  }

  if (isReadyToSendData)
  {
    DataPacket packet;
    packet.type = 1;
    packet.yaw = int(round(ypr[0] * 100 * 180 / M_PI));
    packet.pitch = int(round(ypr[1] * 100 * 180 / M_PI));
    packet.roll = int(round(ypr[2] * 100 * 180 / M_PI));
    packet.accx = int(aaReal.x);
    packet.accy = int(aaReal.y);
    packet.accz = int(aaReal.z);
    packet.padding1 = 0;
    packet.padding2 = 0;
    packet.checksum = getChecksum(packet);

    Serial.write((uint8_t *)&packet, sizeof(packet));
  }
}
