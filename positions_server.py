import rpi_ws281x
from rpi_ws281x import PixelStrip
from rpi_ws281x import Color
import socket
import atexit

# LED strip configuration:
LED_COUNT = 400         # Number of LED pixels.
LED_PIN = 18            # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10            # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000    # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10            # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 128     # Set to 0 for darkest and 255 for brightest
LED_INVERT = False      # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0         # set to '1' for GPIOs 13, 19, 41, 45 or 53
STRIP_TYPE = rpi_ws281x.ws.WS2811_STRIP_GRB

# socket configuration
HOST = '192.168.1.141'
PORT = 5000


def one_hot(strip, on_index=0):
    for i in range(strip.numPixels()):
        if i == on_index:
            strip.setPixelColor(i, int(Color(255, 255, 255)))
        else:
            strip.setPixelColor(i, int(Color(0, 0, 0)))
    strip.show()


def main():
    # Create PixelStrip object with appropriate configuration.
    strip = PixelStrip(
        num=LED_COUNT,
        pin=LED_PIN,
        freq_hz=LED_FREQ_HZ,
        dma=LED_DMA,
        invert=LED_INVERT,
        brightness=LED_BRIGHTNESS,
        channel=LED_CHANNEL,
        strip_type=STRIP_TYPE
    )

    # Initialize the library (must be called once before other functions).
    strip.begin()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # close properly in case of ctrl-c
        atexit.register(s.close)

        # the usual
        s.bind((HOST, PORT))
        s.listen()

        while True:
            print('awaiting new connection')
            conn, addr = s.accept()

            # after a connection, get data
            while True:
                data = conn.recv(1024)
                print(data)

                if data == b'':
                    break

                try:
                    # change the light
                    light_index = int(data)
                    one_hot(strip, light_index)

                    # tell the client we are done changing the light.
                    conn.sendall(bytes("done", 'utf-8'))
                    pass
                except ValueError as e:
                    print(e)



if __name__ == '__main__':
    main()
    pass
