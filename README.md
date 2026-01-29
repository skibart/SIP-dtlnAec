# SIP-dtlnAec — DTLN Acoustic Echo Cancellation with SIP.js

Small proof-of-concept project showing how to integrate **DTLN (deep-learning) Acoustic Echo Cancellation** into a browser **SIP/WebRTC** call using **SIP.js**.

The app captures microphone audio, processes it in real time with DTLN AEC (TFLite models), and uses the filtered stream as the outgoing audio track. Remote audio can be used as the reference signal for echo cancellation.

## Key features
- SIP/WebRTC calling via **SIP.js**
- Real-time audio processing with **WebCodecs** (`TrackProcessor/TrackGenerator`)
- DTLN AEC inference using two `.tflite` models
- Designed for experimental/learning purposes

## Requirements
- Chrome/Edge (WebCodecs support)
- SIP server with WebSocket transport (`wss://`)
- Model files in `./assets/`:
  - `dtln_aec_128_1.tflite`
  - `dtln_aec_128_2.tflite`

## License
[SIP.js](https://sipjs.com/)
[DTLN model](https://github.com/breizhn/DTLN-aec)
