#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "creds.h"

#define LAMP 4

// Select camera model
#define CAMERA_MODEL_AI_THINKER // Has PSRAM

#include "camera_pins.h"

// #define CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY
constexpr int tensor_pool_size = 3600 * 1024;
// EXT_RAM_ATTR uint8_t tensor_pool[tensor_pool_size];
uint8_t *tensor_pool;

// Define the model to be used
const tflite::Model *prod_model;

// Define the interpreter
tflite::MicroInterpreter *interpreter;

// Input/Output nodes for the network
TfLiteTensor *input;
TfLiteTensor *output;

static tflite::ErrorReporter *error_reporter;
static tflite::MicroErrorReporter micro_error;

// Define ops resolver and error reporting
static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);

int8_t *model_input_buffer = nullptr;

boolean wifiConnected = false;
WiFiServer Server(8840);
WiFiClient RemoteClient;

camera_fb_t *fb = NULL;
TaskHandle_t processTaskHandle;
TaskHandle_t captureTaskHandle;

void setup_tflite();
void setup_camera();
void softmax2(double *a, double *b);
void connectToWiFi(const char *ssid, const char *pwd);
void WiFiEvent(WiFiEvent_t event);
void CheckForConnections();

void captureTask(void *param)
{
  while (true)
  {
    if (wifiConnected)
    {
      CheckForConnections();
    }

    // digitalWrite(LAMP, HIGH);

    // Take Picture with Camera
    fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("Camera capture failed");
      continue;
    }

    // digitalWrite(LAMP, LOW);

    // for (int i = 0; i < fb->len; i++)
    // {
    //   model_input_buffer[i] = fb->buf[i] - 128;
    // }

    for (int i = 0; i < 96 * 96; i++)
    {
      int pxIdx = i * 3;
      // data is in BGR format not RGB
      uint8_t b = fb->buf[i];
      uint8_t g = fb->buf[i + 1];
      uint8_t r = fb->buf[i + 2];
      double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      model_input_buffer[i] = (uint8_t)(l + 0.5) - 128;
    }

    xTaskNotify(processTaskHandle, 1, eSetValueWithOverwrite);

    if (wifiConnected && RemoteClient.connected())
    {
      // Send a packet
      RemoteClient.write(fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);
    fb = NULL;
  }
}

void processTask(void *param)
{
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    // pdTRUE - set value 0
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue == 1)
    {
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk)
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        // return false;
      }

      double a = output->data.int8[0];
      double b = output->data.int8[1];
      Serial.print(a);
      Serial.print(" ");
      Serial.print(b);
      Serial.println();
      softmax2(&a, &b);

      Serial.print("ER PROB: ");
      Serial.println(a);
      Serial.print("OK PROB: ");
      Serial.println(b);
      Serial.println();
    }
  }
}

void setup()
{
  setCpuFrequencyMhz(160);

  // pinMode(LAMP, OUTPUT);
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  connectToWiFi(WIFI_SSID, WIFI_PWD);
  setup_tflite();
  setup_camera();

  xTaskCreatePinnedToCore(captureTask, "Capture Task", 4096, NULL, 1, &captureTaskHandle, 0);
  xTaskCreatePinnedToCore(processTask, "Process Task", 4096, NULL, 1, &processTaskHandle, 1);
}

void loop()
{
  // put your main code here, to run repeatedly:
}

void setup_camera()
{
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB888;
  // config.pixel_format = PIXFORMAT_GRAYSCALE;

  config.frame_size = FRAMESIZE_96X96;
  config.fb_count = 1;

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID)
  {
    Serial.println("OV3660_PID");
    s->set_vflip(s, 1);       // flip it back
    s->set_brightness(s, 1);  // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
  }
  else if (s->id.PID == OV2640_PID)
  {
    Serial.println("OV2640_PID");
    s->set_brightness(s, 1);
    s->set_saturation(s, 0); // higher the saturation
  }
}

void setup_tflite()
{
  tensor_pool = (uint8_t *)heap_caps_calloc(tensor_pool_size, 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  // Load the sample sine model
  Serial.println("Loading Tensorflow model....");
  prod_model = tflite::GetModel(model_data);
  Serial.println("Model loaded!");

  error_reporter = &micro_error;

  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk)
  {
    return;
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      prod_model, micro_op_resolver, tensor_pool, tensor_pool_size, error_reporter);

  interpreter = &static_interpreter;

  // Allocate the the model's tensors in the memory pool that was created.
  Serial.println("Allocating tensors to memory pool");
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("There was an error allocating the memory...ooof");
    return;
  }

  // Define input and output nodes
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("input nbytes: ");
  Serial.println(input->bytes);
  Serial.print("output nbytes: ");
  Serial.println(output->bytes);

  model_input_buffer = input->data.int8;

  Serial.println("Starting inferences...! ");
}

void connectToWiFi(const char *ssid, const char *pwd)
{
  Serial.println("Connecting to WiFi network: " + String(ssid));

  // delete old config
  WiFi.disconnect(true);

  //register event handler
  WiFi.onEvent(WiFiEvent);

  //Initiate connection
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
}

//wifi event handler
void WiFiEvent(WiFiEvent_t event)
{
  switch (event)
  {
  case SYSTEM_EVENT_STA_GOT_IP:
    // When connected set
    Serial.print("WiFi connected! IP address: ");
    Serial.println(WiFi.localIP());
    wifiConnected = true;
    Server.begin();
    break;
  case SYSTEM_EVENT_STA_DISCONNECTED:
    Serial.println("WiFi lost connection");
    wifiConnected = false;
    RemoteClient.stop();
    Server.end();
    WiFi.removeEvent(WiFiEvent);
    connectToWiFi(WIFI_SSID, WIFI_PWD);
    break;
  }
}

void CheckForConnections()
{
  if (Server.hasClient())
  {
    // If we are already connected to another computer,
    // then reject the new connection. Otherwise accept
    // the connection.
    if (RemoteClient.connected())
    {
      Serial.println("Connection rejected");
      Server.available().stop();
    }
    else
    {
      Serial.println("Connection accepted");
      RemoteClient = Server.available();
    }
  }
}

void softmax2(double *a, double *b)
{
  double m, sum, constant;

  m = max(*a, *b);

  sum = 0.0;
  sum += exp(*a - m);
  sum += exp(*b - m);

  constant = m + log(sum);
  *a = exp(*a - constant);
  *b = exp(*b - constant);
}