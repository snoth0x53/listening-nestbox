# The Listening Nest Box

A camera and a microphone in an ordinary tit nest box, a Raspberry Pi 5 that watches both, and a phone that buzzes when somebody moves in. Everything runs at home, nothing is sent to the cloud.

**Read the project page:** https://snoth0x53.github.io/listening-nestbox/

## What it does

- **Sees** into the box with an infrared-sensitive camera (Camera Module 3 Wide NoIR) and a small 850 nm IR light.
- **Listens** outside with a USB microphone and names the birds that sing nearby ([BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi)).
- **Notices** movement in the box, checks it with a small object-detection model (YOLOv8n) and tells Home Assistant.
- **Tells me** with a push message, a "bird present" tile on the dashboard and a live picture on iPhone and Apple TV.

## What is in this repository

| File | Content |
|---|---|
| `index.html` and the other `.html` files | The project page: overview, hardware, live stream, bird sounds, detection, Home Assistant, results |
| `watch.py` | The motion and bird-detection script that runs on the Pi |
| `style.css`, `img/` | Styling and photos of the page |

The page is plain HTML and CSS: no build step, no tracking, no cookies, no external scripts or fonts. To look at it locally:

```
python3 -m http.server 8000
```

then open http://localhost:8000.

## Privacy and safety

- The camera stream stays inside the home network. Do not forward ports to it.
- The script switches off usage statistics of the libraries it uses.
- Keep the Home Assistant webhook address secret (put it in a service drop-in file, never in the script).
- Mains voltage outdoors belongs in the hands of a qualified electrician.

## Contact

[contact e-mail to be added]
